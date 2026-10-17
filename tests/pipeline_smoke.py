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
# Wan2GP 18.x: generate() is also wrapped by Veda's geometry-release context.
namespace['veda_generation_context']=lambda method:functools.wraps(method)(lambda self,*a,**k:method(self,*a,**k))
exec((upstream/'models/minimax_h3/constants.py').read_text(),namespace)
exec((upstream/'shared/utils/frame_scheduler.py').read_text(),namespace)
# Native latent normalization constants, used around the two-phase latent upscaler.
vae_tree=ast.parse((upstream/'models/minimax_h3/video_vae.py').read_text())
vae_tree.body=[n for n in vae_tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('LATENTS_MEAN','LATENTS_STD') for t in n.targets)]
exec(compile(vae_tree,str(upstream/'models/minimax_h3/video_vae.py'),'exec'),namespace)
# Wan2GP 18.x imports these native helpers from shared.utils.utils; load the real bodies.
utils_tree=ast.parse((upstream/'shared/utils/utils.py').read_text())
utils_tree.body=[n for n in utils_tree.body if isinstance(n,ast.FunctionDef) and n.name in
                 ('guide_to_float','guide_mask_to_float','convert_video_tensor_to_uint8_chunked')]
exec(compile(utils_tree,str(upstream/'shared/utils/utils.py'),'exec'),namespace)
# Native cooperative checkpoint has no active controller in this CPU harness.
try:
    from shared.utils.media_control import inference_checkpoint
except ImportError:
    inference_checkpoint = lambda: None
namespace['inference_checkpoint'] = inference_checkpoint
excerpts_path=upstream/'models/minimax_h3/excerpts.py'
if excerpts_path.exists():
    exec(excerpts_path.read_text(), namespace)
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
    def encode_condition(self,video,device=None,keep_all_latents=False):
        self.encode_calls+=1
        t=video.shape[2]
        n=math.ceil(t/17)*5 if keep_all_latents else (1 if t==1 else namespace['video_latent_frames'](t))
        return torch.zeros(1,24,n,math.ceil(video.shape[-2]/16),math.ceil(video.shape[-1]/16))
    upsampling=False
    def decode(self,video,uint8_rounding=None):
        # Wan2GP 18.x: the VAE converts to uint8 itself ("truncate" = WGP's float conversion).
        self.decode_dtype=video.dtype
        decoded=F.interpolate(video[:,:3].float(),size=(token_start(video.shape[2]),video.shape[-2]*16,video.shape[-1]*16),mode='nearest').tanh().to(video.dtype)
        if uint8_rounding=='truncate':
            return decoded.float().sub_(-1).mul_(127.5).clamp_(0,255).to(torch.uint8)
        return decoded

class AudioVAE:
    def __init__(self): self.encode_calls=0
    def encode(self,audio):
        self.encode_calls+=1
        return torch.zeros(1,32,2,round(audio.shape[-1]/32000*40))
    def decode(self,audio): return torch.zeros(1,2,round(audio.shape[-1]/40*32000))

class Transformer:
    control_layers=None;pdd_num_steps=None;cache=None;patch_size=(1,2,2)
    def parameters(self): yield torch.zeros(1)
    def preprocess_text_embeds(self,context): return context
    layouts=0
    def __call__(self,video,audio,*args,**kwargs):
        payload=args[-1] if args and isinstance(args[-1],dict) else kwargs.get('payload')
        if payload is not None and not payload.get('refs'):
            # Build the actual native packed layout and check every condition row is placed.
            model=native_model();model.patch_size=self.patch_size
            layout=model._layout(payload['text_token_tags'],video.shape[2],video.shape[3],video.shape[4],audio.shape[-1],payload)
            rows=payload.get('cond_video_rows')
            assert layout.num_condition_video_rows==(0 if rows is None else rows.shape[0]), 'Condition video rows do not match the stage layout'
            rows=payload.get('cond_audio_rows')
            assert layout.num_condition_audio_rows==(0 if rows is None else rows.shape[0]), 'Condition audio rows do not match the layout'
            Transformer.layouts+=1
        return -video*0.5,-audio*0.5

class LatentUpscaler:
    # Stand-in for the native latent upscaler between the two H3 phases.
    def __init__(self): self.calls=0
    def __call__(self,video,scale,target_size=None,abort_callback=None,progress_callback=None):
        self.calls+=1
        return F.interpolate(video.float(),size=target_size,mode='nearest').to(video.dtype)

class TestPipeline(Base):
    def __init__(self):
        self.transformer=Transformer();self.vae=VideoVAE();self.audio_vae=AudioVAE()
        self.audio_only=False;self.reference_mode=False;self.fixed_prompt=None
        self.text_encoder=None;self.latent_upscaler=LatentUpscaler();self.dialogue_whisper=None
        self.presented=[]
        self.dtype=torch.float32;self._abort=False;self._early_stop=False
        self._private_offloadobj=None;self._shared_offloadobj=None
        self._lc_identity={'architecture':'test'}
    def _prepare_audio_references(self,sources):
        assert not sources
        return []
    def _encode_prompt(self,prompt,presentation):
        # Record the image sizes shown to the prompt encoder at each stage.
        self.presented.append([tuple(p['frames'].shape) for p in presentation if p.get('type')=='image'])
        return torch.zeros(1),torch.zeros(3,dtype=torch.long)

pipe=TestPipeline()
def last_frame(x):
    # Same conversion as integration.make_save_wrapper for native uint8 exports.
    last=x[:,-1:].detach().cpu().float()
    return last.div_(127.5).sub_(1.0) if x.dtype==torch.uint8 else last
def run(opts,size=32,**kw):
    job={'enabled':opts['save'] or opts['continue'],'options':opts,'pending':None}
    token=JOB.set(job)
    try:
        out=pipe.generate('test',frame_num=124,width=size,height=size,sampling_steps=2,seed=19,**kw)
        assert out is not None and out['x'].shape==(3,124,size,size)
        return out,job
    finally: JOB.reset(token)

out,job=run({'save':True,'continue':False,'latent_path':''})
assert pipe.vae.encode_calls==0
pending=job['pending'];pending['tensors']['last_frame']=last_frame(out['x'])
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
    # Wan2GP 18.x native audio repair (audio_conditioning_end before the history end)
    # moves the target audio start. A latent join must keep the saved-latent geometry.
    import contextlib, io
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        repaired,repaired_job=run({'save':True,'continue':True,'latent_path':str(cp)},input_video=source,prefix_frames_count=18,
                                  input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000,audio_conditioning_end=0.5)
    assert 'Native audio join repair' in log.getvalue(), 'Native audio repair offset was not neutralized'
    assert repaired_job['pending']['tensors']['audio'].shape==newjob['pending']['tensors']['audio'].shape
    # Same length as the native 18.x baseline continuation (native slices instead of padding).
    assert repaired['audio'].shape==continued['audio'].shape==baseline['audio'].shape
    assert pipe.vae.encode_calls==2 and pipe.audio_vae.encode_calls==1
    encodes=(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    native_repair,_=run({'save':False,'continue':False,'latent_path':''},input_video=source,prefix_frames_count=18,
                        input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000,audio_conditioning_end=0.5)
    assert pipe.audio_vae.encode_calls>encodes[1], 'Disabled options must use the original native path'
    assert native_repair['audio'].shape[0]>=baseline['audio'].shape[0], 'Native repair should regenerate earlier audio'
    pipe.vae.encode_calls,pipe.audio_vae.encode_calls=encodes
    print('PASS: native audio repair kept for native jobs, neutralized (and logged) for latent joins.')
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
            advanced_pending['tensors']['last_frame']=last_frame(result['x'])
            cp2=Path(td)/'clip2.safetensors'
            atomic_save(cp2,advanced_pending['tensors'],advanced_pending['info'])
            advanced['latent_path']=str(cp2)
            third,third_job=run(advanced,input_video=result['x'][:,-18:],prefix_frames_count=18,
                               input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
            assert third['audio'].shape==result['audio'].shape
            assert -3 < third_job['pending']['info']['audio_origin_seconds'] < 0
    # The reported model is Ref2VA: exercise its native branch separately.
    pipe.reference_mode=True
    ref_out,ref_job=run({'save':True,'continue':False,'latent_path':''})
    ref_pending=ref_job['pending']
    ref_pending['tensors']['last_frame']=last_frame(ref_out['x'])
    ref_cp=Path(td)/'ref.safetensors'
    atomic_save(ref_cp,ref_pending['tensors'],ref_pending['info'])
    encodes=(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    ref_cont,ref_cont_job=run({'save':True,'continue':True,'latent_path':str(ref_cp)},
                             input_video=ref_out['x'][:,-18:],prefix_frames_count=18)
    assert ref_cont_job['pending']['info']['continuation_mode']=='latent'
    assert encodes==(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    pipe.reference_mode=False
    pipe.transformer.control_layers=[0]
    try:
        run({'save':True,'continue':False,'latent_path':''})
    except ValueError as error:
        assert 'ControlNet-Union' in str(error)
    else:
        raise AssertionError('ControlNet latent capture must be rejected')
    finally:
        pipe.transformer.control_layers=None
# ---- Two-phase H3: low-resolution phase 1, latent upscale, high-resolution phase 2 ----
def save_cp(path,result,job):
    pending=job['pending'];pending['tensors']['last_frame']=last_frame(result['x'])
    atomic_save(path,pending['tensors'],pending['info'])
    return pending
with tempfile.TemporaryDirectory() as td:
    upscales=pipe.latent_upscaler.calls
    two,two_job=run({'save':True,'continue':False,'latent_path':''},size=64,guide_phases=2)
    assert pipe.latent_upscaler.calls==upscales+1, 'Native two-phase path was not used'
    tensors=two_job['pending']['tensors'];info=two_job['pending']['info']
    assert tuple(tensors['video'].shape[-2:])==(4,4) and tuple(tensors['video_phase1'].shape[-2:])==(2,2)
    assert tensors['video_phase1'].shape[2]==tensors['video'].shape[2]
    assert info['guide_phases']==2 and info['phase1']=={'width':32,'height':32} and (info['width'],info['height'])==(64,64)
    assert tensors['video'].dtype==torch.float32 and tensors['video_phase1'].dtype==torch.float32
    cp2=Path(td)/'two.safetensors';two_pending=save_cp(cp2,two,two_job)
    md2,ts2=load_checkpoint(cp2)
    assert torch.equal(ts2['video_phase1'],tensors['video_phase1']) and md2['phase1']=={'width':32,'height':32}
    source2=two['x'][:,-18:]
    encodes=(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    for mode in ('baseline','context','audio_prefix'):
        opts={'save':True,'continue':True,'latent_path':str(cp2),'join_mode':mode,'video_context_frames':35,'audio_context_seconds':1.0}
        pipe.presented=[]
        log=io.StringIO()
        with contextlib.redirect_stdout(log):
            cont,cont_job=run(opts,size=64,guide_phases=2,input_video=source2,prefix_frames_count=18,
                              input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
        assert encodes==(pipe.vae.encode_calls,pipe.audio_vae.encode_calls), f'{mode}: two-phase latent continuation re-encoded AV'
        out_log=log.getvalue()
        assert '(phase 1 latents)' in out_log and '(full resolution latents)' in out_log, out_log
        # The saved last frame is presented at each stage's size, as native does with the pixel frame.
        assert pipe.presented[0][0][-2:]==(32,32) and pipe.presented[1][0][-2:]==(64,64), pipe.presented  # C,T,H,W (identity _qwen_frames)
        pending=cont_job['pending']
        assert pending['info']['continuation_mode']=='latent' and pending['info']['guide_phases']==2
        assert tuple(pending['tensors']['video_phase1'].shape[-2:])==(2,2)
        if mode=='audio_prefix':
            start,stop,origin=audio_window(md2,ts2['audio'].shape[-1],1.0)
            assert torch.equal(pending['tensors']['audio'][...,:stop-start],ts2['audio'][...,start:stop]), 'Frozen prefix changed in two-phase'
            assert cont['audio'].shape==(round(124/24*32000),2)
        # Successive two-phase continuation from the new checkpoint.
        if mode=='baseline':
            cp3=Path(td)/'two-next.safetensors';save_cp(cp3,cont,cont_job)
            nxt,nxt_job=run({'save':True,'continue':True,'latent_path':str(cp3)},size=64,guide_phases=2,input_video=cont['x'][:,-18:],
                            prefix_frames_count=18,input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
            assert nxt_job['pending']['info']['continuation_mode']=='latent'
    # A two-phase checkpoint can be continued in one phase (full-resolution latents).
    one,one_job=run({'save':True,'continue':True,'latent_path':str(cp2)},size=64,input_video=source2,prefix_frames_count=18,
                    input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
    assert 'video_phase1' not in one_job['pending']['tensors'] and one_job['pending']['info']['guide_phases']==1
    assert 'phase1' not in one_job['pending']['info']
    # A single-phase checkpoint has no phase-1 latents: explicit refusal for a two-phase continuation.
    single,single_job=run({'save':True,'continue':False,'latent_path':''},size=64)
    cp1=Path(td)/'single.safetensors';save_cp(cp1,single,single_job)
    try:
        run({'save':False,'continue':True,'latent_path':str(cp1)},size=64,guide_phases=2,input_video=single['x'][:,-18:],
            prefix_frames_count=18,input_waveform=torch.zeros(2,24000),input_waveform_sample_rate=32000)
    except ValueError as error:
        assert 'single-phase render' in str(error), error
    else:
        raise AssertionError('Two-phase continuation of a single-phase checkpoint must be refused')
    # Phase-1 manifest/tensor consistency is validated on load.
    bad=dict(two_pending['info']);bad['phase1']={'width':48,'height':32}
    atomic_save(Path(td)/'bad.safetensors',two_pending['tensors'],bad)
    try: load_checkpoint(Path(td)/'bad.safetensors')
    except ValueError as error: assert 'phase-1' in str(error)
    else: raise AssertionError('Mismatched phase-1 manifest accepted')
    # Ref2VA two-phase save and continuation.
    pipe.reference_mode=True
    ref2,ref2_job=run({'save':True,'continue':False,'latent_path':''},size=64,guide_phases=2)
    cpr=Path(td)/'ref2.safetensors';save_cp(cpr,ref2,ref2_job)
    encodes=(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    _,refc_job=run({'save':True,'continue':True,'latent_path':str(cpr)},size=64,guide_phases=2,input_video=ref2['x'][:,-18:],prefix_frames_count=18)
    assert refc_job['pending']['info']['continuation_mode']=='latent'
    assert encodes==(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    pipe.reference_mode=False
    # Native two-phase render with latent options off still uses the original method.
    run({'save':False,'continue':False,'latent_path':''},size=64,guide_phases=2)
assert Transformer.layouts>0
print('PASS: two-phase save (phase-1 + final latents), continuation in all three modes with exact per-stage latents and no AV re-encode, '
      'successive two-phase joins, Ref2VA, two-phase -> one-phase, single-phase checkpoint refused for two phases, manifest validation.')
# Capture must stay before the native cast to the VAE dtype.
pipe.vae._model_dtype=torch.float16
cast_out,cast_job=run({'save':True,'continue':False,'latent_path':''})
assert cast_job['pending']['tensors']['video'].dtype==torch.float32
assert pipe.vae.decode_dtype==torch.float16, 'Native cast to the VAE dtype was bypassed'
assert cast_out['x'].dtype==torch.uint8  # 18.x decodes straight to WGP-truncated uint8
pipe.vae._model_dtype=torch.float32
print('PASS: pre-decode capture preserves float32 inference latents with an FP16 decoder.')
print('PASS: Ref2VA capture/continuation; ControlNet latent operation explicitly rejected.')
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
