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
tree=ast.parse((ROOT/'pipeline.py').read_text())
tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
exec(compile(tree,str(ROOT/'pipeline.py'),'exec'),namespace)
# Prompt representation preprocessing is independent of the latent injection being tested.
namespace['_qwen_frames']=lambda v:v
Base=namespace['MiniMaxH3Pipeline']

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
