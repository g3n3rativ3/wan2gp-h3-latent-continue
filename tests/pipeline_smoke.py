"""Execute the actual generate() body with tiny deterministic CPU stand-ins.
This tests control flow/capture/replay, NOT H3 quality or CUDA/MMGP integration.
Usage: python tests/pipeline_smoke.py /path/to/Wan2GP
"""
import ast
import functools
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]; upstream=Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location('h3smoke',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
pkg=importlib.util.module_from_spec(spec);sys.modules['h3smoke']=pkg;spec.loader.exec_module(pkg)
from h3smoke.latent_runtime import LatentMixin, JOB
from h3smoke.checkpoint import atomic_save, load_checkpoint, token_start
from h3smoke.packing import patchify_video_latents
from h3smoke.seams import audio_window
import types
for name, path in [('shared',upstream/'shared'),('shared.utils',upstream/'shared/utils')]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
from h3smoke.progress_bridge import generation_progress, control_video_encoding
import tempfile

namespace=dict(torch=torch,np=np,F=F,math=math,os=os,functools=functools,hashlib=hashlib,
               generation_progress=generation_progress,control_video_encoding=control_video_encoding,
               LatentMixin=LatentMixin,GenerationInterrupted=RuntimeError,
               H3_DIALOGUE_GENERATION=False,VISUAL_COND_TIMESTEP=0.999,
               offload=SimpleNamespace(set_step_no_for_lora=lambda *a:None),
               tqdm=lambda it,**kw:it)
exec((upstream/'models/minimax_h3/constants.py').read_text(),namespace)
exec((upstream/'shared/utils/frame_scheduler.py').read_text(),namespace)
namespace['patchify_video']=patchify_video_latents
namespace['pack_audio']=lambda latent:latent[0].permute(1,2,0).reshape(-1,32).contiguous()
namespace['unpack_audio']=lambda rows:rows.reshape(2,-1,32).permute(2,0,1).unsqueeze(0)
native_path=upstream/'models/minimax_h3/pipeline.py'
tree=ast.parse(native_path.read_text())
tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
exec(compile(tree,str(native_path),'exec'),namespace)
# Prompt representation preprocessing is independent of the latent injection being tested.
namespace['_qwen_frames']=lambda v:v
Base=namespace['MiniMaxH3Pipeline']
from h3smoke.native_bridge import compile_edit, dispatch_install
import json
edits=json.loads((ROOT/'native_edits.json').read_text())
# Simulate another plugin wrapping native generate before latent installation.
seen_wrappers=[]
original_generate=Base.generate
@functools.wraps(original_generate)
def other_plugin(self,*args,**kwargs):
    seen_wrappers.append('called')
    return original_generate(self,*args,**kwargs)
def closure_plugin(previous):
    def wrapped(self,*args,**kwargs):
        seen_wrappers.append('closure-before')
        result=previous(self,*args,**kwargs)
        seen_wrappers.append('closure-after')
        return result
    return wrapped
closure_wrapper=closure_plugin(other_plugin)
Base.generate=closure_wrapper
# Install against native packing and transformer layout as well as generate.
for name,path in [('models',upstream/'models'),('models.minimax_h3',upstream/'models/minimax_h3'),
                  ('models.minimax_h3.components',upstream/'models/minimax_h3/components')]:
    module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
spec_pack=importlib.util.spec_from_file_location('models.minimax_h3.components.packing',upstream/'models/minimax_h3/components/packing.py')
native_pack=importlib.util.module_from_spec(spec_pack);sys.modules[spec_pack.name]=native_pack;spec_pack.loader.exec_module(native_pack)
transform_path=upstream/'models/minimax_h3/transformer.py'
transform_tree=ast.parse(transform_path.read_text())
cls=next(n for n in transform_tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3Model')
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_layout'];cls.bases=[]
transform_tree.body=[cls]
module=types.ModuleType('models.minimax_h3.transformer')
module.__dict__.update(torch=torch,build_packed_sequence=native_pack.build_packed_sequence,
                       build_ref2va_packed_sequence=native_pack.build_ref2va_packed_sequence)
exec(compile(transform_tree,str(transform_path),'exec'),module.__dict__)
sys.modules[module.__name__]=module
native_model=module.MiniMaxH3Model
module=types.ModuleType('models.minimax_h3.pipeline');module.MiniMaxH3Pipeline=Base;sys.modules[module.__name__]=module
from h3smoke.native_bridge import ensure_hooks, install_loader
ensure_hooks()
ensure_hooks()  # Idempotent, no duplicate edits.
assert Base.generate is closure_wrapper


class VideoVAE:
    _model_dtype=torch.float32
    def __init__(self): self.encode_calls=0
    def enable_tiling(self,**kw): pass
    def encode_condition(self,video,keep_all_latents=False):
        self.encode_calls+=1
        t=video.shape[2]
        n=math.ceil(t/17)*5 if keep_all_latents else (1 if t==1 else namespace['video_latent_frames'](t))
        return torch.zeros(1,24,n,2,2)
    def decode(self,video):
        return F.interpolate(video[:,:3],size=(token_start(video.shape[2]),32,32),mode='nearest').tanh()

class AudioVAE:
    def __init__(self): self.encode_calls=0
    def encode(self,audio):
        self.encode_calls+=1
        return torch.zeros(1,32,2,round(audio.shape[-1]/32000*40))
    def decode(self,audio): return torch.zeros(1,2,round(audio.shape[-1]/40*32000))

class Transformer:
    pdd_num_steps=None;cache=None;patch_size=(1,2,2)
    def parameters(self): yield torch.zeros(1)
    def preprocess_text_embeds(self,context): return context
    def __call__(self,video,audio,*args,**kwargs): return -video*0.5,-audio*0.5

class TestPipeline(Base):
    def __init__(self):
        self.transformer=Transformer();self.vae=VideoVAE();self.audio_vae=AudioVAE()
        self.audio_only=False;self.reference_mode=False;self.fixed_prompt=None
        self.text_encoder=None;self.latent_upscaler=None;self.dialogue_whisper=None
        self.dtype=torch.float32;self._abort=False;self._early_stop=False
        self._private_offloadobj=None;self._shared_offloadobj=None
        self._lc_identity={'architecture':'test'}
    def _prepare_audio_references(self,sources):
        assert not sources
        return []
    def _encode_prompt(self,prompt,presentation): return torch.zeros(1),torch.zeros(3,dtype=torch.long)

pipe=TestPipeline()
def run(opts,**kw):
    job={'enabled':opts['save'] or opts['continue'],'options':opts,'pending':None}
    token=JOB.set(job)
    try:
        out=pipe.generate('test',frame_num=124,width=32,height=32,sampling_steps=2,seed=19,**kw)
        assert out is not None and out['x'].shape==(3,124,32,32)
        return out,job
    finally: JOB.reset(token)

out,job=run({'save':True,'continue':False,'latent_path':''})
assert pipe.vae.encode_calls==0
pending=job['pending'];pending['tensors']['last_frame']=out['x'][:,-1:].clone()
with tempfile.TemporaryDirectory() as td:
    cp=Path(td)/'clip.safetensors';atomic_save(cp,pending['tensors'],pending['info'])
    source=out['x'][:,-18:]
    baseline,_=run({'save':False,'continue':False,'latent_path':''},input_video=source,prefix_frames_count=18,
                   input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
    assert pipe.vae.encode_calls==2 and pipe.audio_vae.encode_calls==1
    continued,newjob=run({'save':True,'continue':True,'latent_path':str(cp)},input_video=source,prefix_frames_count=18,
                        input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
    assert pipe.vae.encode_calls==2,'Latent continuation unexpectedly called the video encoder'
    assert pipe.audio_vae.encode_calls==1,'Latent continuation unexpectedly called the audio encoder'
    assert newjob['pending']['info']['target_frames']==107
    assert newjob['pending']['info']['continuation_mode']=='latent'
    short,shortjob=run({'save':True,'continue':True,'latent_path':str(cp)},
                       input_video=source[:,-1:],prefix_frames_count=1,
                       input_waveform=torch.zeros(2,1333),input_waveform_sample_rate=32000)
    assert shortjob['pending']['info']['target_frames']==124, 'Context changed assembly duration'
    assert shortjob['pending']['info']['effective_context']['video_latent_indices']==[31,32,33,34,35,36]
    assert shortjob['pending']['info']['effective_context']['native_overlap_frames']==1
    assert pipe.vae.encode_calls==2 and pipe.audio_vae.encode_calls==1
    md,ts=load_checkpoint(cp)
    assert torch.equal(ts['video'],pending['tensors']['video'])
    for mode in ('context','audio_prefix'):
        advanced={'save':True,'continue':True,'latent_path':str(cp),'join_mode':mode,
                  'video_context_frames':35,'audio_context_seconds':1.0}
        result,advanced_job=run(advanced,input_video=source,prefix_frames_count=18,
                               input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
        assert result['audio'].shape==(round(124/24*32000),2)
        assert pipe.vae.encode_calls==2 and pipe.audio_vae.encode_calls==1
        advanced_pending=advanced_job['pending']
        if mode=='audio_prefix':
            start,stop,origin=audio_window(md,ts['audio'].shape[-1],1.0)
            assert torch.equal(advanced_pending['tensors']['audio'][...,:stop-start],ts['audio'][...,start:stop]), 'Frozen prefix changed during denoising'
            assert advanced_pending['info']['audio_origin_seconds']==origin
            advanced_pending['tensors']['last_frame']=result['x'][:,-1:].clone()
            cp2=Path(td)/'clip2.safetensors'
            atomic_save(cp2,advanced_pending['tensors'],advanced_pending['info'])
            advanced['latent_path']=str(cp2)
            third,third_job=run(advanced,input_video=result['x'][:,-18:],prefix_frames_count=18,
                               input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
            assert third['audio'].shape==result['audio'].shape
            assert -3 < third_job['pending']['info']['audio_origin_seconds'] < 0
print('PASS: extended context and frozen audio prefix, exact prefix preservation, sample counts, next-generation reload, no VAE re-encode.')
print('PASS: actual adapted generate() with CPU stand-ins: generation, baseline continuation, latent AV continuation, capture; no AV encoder calls in latent mode.')

assert seen_wrappers, 'Existing native wrapper was bypassed'
print('PASS: installed native generate source and existing plugin wrapper retained.')

# Native layout executes with explicit condition anchors and shifted target rows.
model=native_model();model.patch_size=(1,2,2)
payload={'fps':24,'audio_keyframes':[{'anchor':-20.,'latent_frame_count':10}],
         'lc_audio_offset':-39.,'target_audio_condition_latents':40}
token=JOB.set({'enabled':True})
try:
    layout=model._layout(torch.zeros(3,dtype=torch.long),37,2,2,240,payload)
    positions=layout.position_ids.clone()
    assert model._layout(torch.zeros(3,dtype=torch.long),37,2,2,240,payload) is layout
    assert torch.equal(positions,layout.position_ids), 'Cached audio shift applied twice'
finally: JOB.reset(token)
# Native loader wrapper preserves pipeline class and other plugins' properties.
class Handler:
    @staticmethod
    def load_model(base_model_type,model_filename,model_def):
        pipe.other_plugin_marker=True
        return pipe, 'offload'
module=types.ModuleType('models.minimax_h3.minimax_h3_handler');module.family_handler=Handler;sys.modules[module.__name__]=module
module=types.ModuleType('models.minimax_h3.minimax_h3_main');module.VIDEO_VAE_FILE='video';module.AUDIO_VAE_FILE='audio';sys.modules[module.__name__]=module
before=type(pipe)
install_loader();install_loader()
loaded,_=Handler.load_model('minimax_h3_fl2va','/weights/model.safetensors',{})
assert loaded is pipe and type(loaded) is before and loaded.other_plugin_marker
assert loaded._lc_identity['architecture']=='minimax_h3_fl2va'
print('PASS: native layout/packing hooks, cache, idempotence, original loader and class preserved.')

assert 'closure-before' in seen_wrappers and 'closure-after' in seen_wrappers
print('PASS: undecorated closure wrapper retained around native latent generation.')
