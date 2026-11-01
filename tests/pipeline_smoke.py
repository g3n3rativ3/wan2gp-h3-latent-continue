"""Execute the actual generate() body with tiny deterministic CPU stand-ins.
This tests control flow/capture/replay, NOT H3 quality or CUDA/MMGP integration.
Usage: python tests/pipeline_smoke.py /path/to/Wan2GP [/path/to/wan2gp-h3-audio-refinement]
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

pipeline_module=types.ModuleType('models.minimax_h3.pipeline')  # globals live in a real module, as in WanGP
namespace=pipeline_module.__dict__
namespace.update(torch=torch,np=np,F=F,math=math,os=os,functools=functools,hashlib=hashlib,
               generation_progress=generation_progress,control_video_encoding=control_video_encoding,
               LatentMixin=LatentMixin,GenerationInterrupted=RuntimeError,
               H3_DIALOGUE_GENERATION=False,VISUAL_COND_TIMESTEP=0.999,
               offload=SimpleNamespace(set_step_no_for_lora=lambda *a:None,activate_loras=lambda *a,**k:None),
               tqdm=lambda it,**kw:it)
from PIL import Image as _PILImage
namespace['Image']=_PILImage  # native image references
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
# Wan2GP 19.x: Pika H3 Flex sampler (own noise layout and renoise steps), loaded from the real module.
pika_path=upstream/'models/minimax_h3/pika.py'
if pika_path.exists():
    pika_ns={}
    exec(compile(pika_path.read_text(),str(pika_path),'exec'),pika_ns)
    namespace.update(PIKA_SOLVER=pika_ns['PIKA_SOLVER'],PIKA_STEPS=pika_ns['PIKA_STEPS'],PIKA_VIDEO_SHIFT=pika_ns['PIKA_VIDEO_SHIFT'],
                     pika_noise=pika_ns['noise'],pika_renoise_=pika_ns['renoise_'],pika_sigma_schedule=pika_ns['sigma_schedule'])
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
pipeline_module.__file__=str(native_path)
refinement_plugin=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else None
if refinement_plugin is not None:
    # H3 Audio Refinement Controls is set up at WanGP start, before any latent job.
    ar_spec=importlib.util.spec_from_file_location('h3ar_core',refinement_plugin/'h3ar_core.py')
    ar_core=importlib.util.module_from_spec(ar_spec);sys.modules['h3ar_core']=ar_core;ar_spec.loader.exec_module(ar_core)
    assert ar_core.patch_pipeline(pipeline_module)
    assert Base._h3ar_exact_schedule and Base._h3ar_audio_refs, 'Audio refinement plugin rewrite not installed'
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
# Keep the native reference preparation so layouts with Ref2VA references are checked too.
transform_tree.body=[n for n in transform_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prepared_references']+[cls]
module=types.ModuleType('models.minimax_h3.transformer')
module.__dict__.update(torch=torch,build_packed_sequence=native_pack.build_packed_sequence,
                       MiniMaxH3PreparedReference=native_pack.MiniMaxH3PreparedReference,
                       build_ref2va_packed_sequence=native_pack.build_ref2va_packed_sequence)
exec(compile(transform_tree,str(transform_path),'exec'),module.__dict__)
sys.modules[module.__name__]=module
native_model=module.MiniMaxH3Model
sys.modules[pipeline_module.__name__]=pipeline_module
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
        # Latents that depend on the pixels (mean over the latent cell), so tests can tell which frames were encoded.
        pooled=F.adaptive_avg_pool3d(video.float().mean(1,keepdim=True),(n,math.ceil(video.shape[-2]/16),math.ceil(video.shape[-1]/16)))
        self.last_input,self.last_output=video.detach().clone(),pooled.repeat(1,24,1,1,1)
        return self.last_output.clone()
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
    refinement=[]
    audio_inputs=[]
    nan_audio=False
    def __call__(self,video,audio,*args,**kwargs):
        Transformer.audio_inputs.append(audio.detach().clone())
        if Transformer.nan_audio:  # model blow-up stand-in: invalid values in some audio latents
            out=-audio*0.5; out[...,3:6]=float('nan')
            return -video*0.5,out
        payload=args[-1] if args and isinstance(args[-1],dict) else kwargs.get('payload')
        if payload is not None and payload.get('keyframes') is None and payload.get('target_video_condition_frames')==video.shape[2]:
            # Audio Refinement Extra Phase: frozen video, audio re-denoised.
            Transformer.refinement.append({'sigma_audio':float(args[1].flatten()[0]),'audio_keyframes':payload.get('audio_keyframes'),
                                           'target_audio_condition_latents':payload.get('target_audio_condition_latents'),
                                           'lc_audio_offset':payload.get('lc_audio_offset',0.0),'audio':audio.detach().clone()})
        if payload is not None:
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
        self.ptd=False  # Wan2GP 19.x: PTD RefAsymOPD acceleration off (native default)
    def _prepare_audio_references(self,sources):
        # Waveform tensors are used as given (the native method also loads files and shares a 15 s budget).
        return [source for source in sources if source is not None]
    def _encode_prompt(self,prompt,presentation):
        # Record the image sizes shown to the prompt encoder at each stage.
        self.presented.append([tuple(p['frames'].shape) for p in presentation if p.get('type')=='image'])
        return torch.zeros(1),torch.zeros(3,dtype=torch.long)

pipe=TestPipeline()
# Wan2GP 18.19+: generate() returns audio running H3_AUDIO_TAIL_LATENTS past the last frame.
FULL_AUDIO=round(124/24*32000)+round(namespace.get('H3_AUDIO_TAIL_LATENTS',0)*32000/40)
def last_frame(x):
    # Same conversion as integration.make_save_wrapper for native uint8 exports.
    last=x[:,-1:].detach().cpu().float()
    return last.div_(127.5).sub_(1.0) if x.dtype==torch.uint8 else last
def run(opts,size=32,**kw):
    job={'enabled':opts['save'] or opts['continue'],'options':opts,'pending':None}
    token=JOB.set(job)
    try:
        steps=kw.pop('sampling_steps',2)
        out=pipe.generate('test',frame_num=124,width=size,height=size,sampling_steps=steps,seed=19,**kw)
        pin=getattr(pipe,'_lc_pinned',None) if opts.get('continue') and opts.get('join_mode')=='pinned' else None
        # Exact join generates the largest H3 length not above the request (the scene ends where the video ends).
        frames=124-(pin['requested_target_frames']-pin['target_frames'] if pin else 0)
        assert out is not None and out['x'].shape==(3,frames,size,size), (out['x'].shape if out else None, frames)
        return out,job
    finally: JOB.reset(token)

out,job=run({'save':True,'continue':False,'latent_path':''})
assert pipe.vae.encode_calls==0
pending=job['pending'];pending['tensors']['last_frame']=last_frame(out['x'])
with tempfile.TemporaryDirectory() as td:
    cp=Path(td)/'clip.safetensors';atomic_save(cp,pending['tensors'],pending['info'])
    source=out['x'][:,-18:]
    baseline,_=run({'save':False,'continue':False,'latent_path':''},input_video=source,prefix_frames_count=18,
                   input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
    assert pipe.vae.encode_calls==2 and pipe.audio_vae.encode_calls==1
    continued,newjob=run({'save':True,'continue':True,'latent_path':str(cp)},input_video=source,prefix_frames_count=18,
                        input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
    assert pipe.vae.encode_calls==2,'Latent continuation unexpectedly called the video encoder'
    assert pipe.audio_vae.encode_calls==1,'Latent continuation unexpectedly called the audio encoder'
    assert newjob['pending']['info']['target_frames']==107
    # Second-part export: WGP overlap and copied history recorded, with tail fingerprints.
    assert newjob['part2']['overlap']==18 and newjob['part2']['history']==17 and newjob['part2']['tail_signatures']
    assert job['part2'] is None, 'A render without Continue Video must not export a second part'
    assert newjob['pending']['info']['continuation_mode']=='latent'
    short,shortjob=run({'save':True,'continue':True,'latent_path':str(cp)},
                       input_video=source[:,-1:],prefix_frames_count=1,
                       input_waveform=torch.zeros(1333,2),input_waveform_sample_rate=32000)
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
                                  input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000,audio_conditioning_end=0.5)
    assert 'Native audio join repair' in log.getvalue(), 'Native audio repair offset was not neutralized'
    assert repaired_job['pending']['tensors']['audio'].shape==newjob['pending']['tensors']['audio'].shape
    # Same length as the native 18.x baseline continuation (native slices instead of padding).
    assert repaired['audio'].shape==continued['audio'].shape==baseline['audio'].shape
    assert pipe.vae.encode_calls==2 and pipe.audio_vae.encode_calls==1
    encodes=(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    native_repair,_=run({'save':False,'continue':False,'latent_path':''},input_video=source,prefix_frames_count=18,
                        input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000,audio_conditioning_end=0.5)
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
                               input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
        assert result['audio'].shape==(FULL_AUDIO,2)
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
                               input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
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
                              input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
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
            assert cont['audio'].shape==(FULL_AUDIO,2)
        # Successive two-phase continuation from the new checkpoint.
        if mode=='baseline':
            cp3=Path(td)/'two-next.safetensors';save_cp(cp3,cont,cont_job)
            nxt,nxt_job=run({'save':True,'continue':True,'latent_path':str(cp3)},size=64,guide_phases=2,input_video=cont['x'][:,-18:],
                            prefix_frames_count=18,input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
            assert nxt_job['pending']['info']['continuation_mode']=='latent'
    # A two-phase checkpoint can be continued in one phase (full-resolution latents).
    one,one_job=run({'save':True,'continue':True,'latent_path':str(cp2)},size=64,input_video=source2,prefix_frames_count=18,
                    input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
    assert 'video_phase1' not in one_job['pending']['tensors'] and one_job['pending']['info']['guide_phases']==1
    assert 'phase1' not in one_job['pending']['info']
    # A single-phase checkpoint has no phase-1 latents: explicit refusal for a two-phase continuation.
    single,single_job=run({'save':True,'continue':False,'latent_path':''},size=64)
    cp1=Path(td)/'single.safetensors';save_cp(cp1,single,single_job)
    try:
        run({'save':False,'continue':True,'latent_path':str(cp1)},size=64,guide_phases=2,input_video=single['x'][:,-18:],
            prefix_frames_count=18,input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
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
# ---- Phase-1 size changed between the source render and the continuation ----
with tempfile.TemporaryDirectory() as td:
    scale=namespace['H3_TWO_PHASE_SCALE']
    try:
        src,src_job=run({'save':True,'continue':False,'latent_path':''},size=128,guide_phases=2)  # phase 1 64x64
        assert tuple(src_job['pending']['tensors']['video_phase1'].shape[-2:])==(4,4)
        cps=Path(td)/'scaled.safetensors';src_job['pending']['tensors']['last_frame']=last_frame(src['x'])
        atomic_save(cps,src_job['pending']['tensors'],src_job['pending']['info'])
        namespace['H3_TWO_PHASE_SCALE']=1.5  # e.g. another plugin or setting: phase 1 now 96x96
        log=io.StringIO()
        with contextlib.redirect_stdout(log):
            resized,resized_job=run({'save':True,'continue':True,'latent_path':str(cps)},size=128,guide_phases=2,input_video=src['x'][:,-18:],
                                    prefix_frames_count=18,input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
        assert 'resampled' in log.getvalue(), log.getvalue()
        assert tuple(resized_job['pending']['tensors']['video_phase1'].shape[-2:])==(6,6)
        assert resized_job['pending']['info']['effective_context'].get('phase1_resampled_from')==[64,64]
    finally:
        namespace['H3_TWO_PHASE_SCALE']=scale
print('PASS: phase-1 size changed since the source render -> saved phase-1 latents resampled for phase 1, exact latents in phase 2.')

# ---- Exact join: new window on a VAE chunk boundary, first latents pinned to the saved ones ----
exact={'save':True,'continue':True,'join_mode':'pinned','video_context_frames':35,'audio_context_seconds':1.0}
with tempfile.TemporaryDirectory() as td:
    for phases,size in ((1,32),(2,64)):
        src,src_job=run({'save':True,'continue':False,'latent_path':''},size=size,guide_phases=phases)
        cpp=Path(td)/f'exact{phases}.safetensors';src_job['pending']['tensors']['last_frame']=last_frame(src['x'])
        atomic_save(cpp,src_job['pending']['tensors'],src_job['pending']['info'])
        mdp,tsp=load_checkpoint(cpp)
        start,stop,origin=audio_window(mdp,tsp['audio'].shape[-1],1.0)
        # er_sde is not checked: native er_sde already returns NaN from sigma 1 with these stand-ins (division by 1 - sigma).
        for solver in (('euler','res_multistep','ralston_2s') if phases==1 else ('euler',)):
            encodes=(pipe.vae.encode_calls+(phases==2),pipe.audio_vae.encode_calls)  # two phases: phase 1 re-encodes the pinned frames
            log=io.StringIO()
            with contextlib.redirect_stdout(log):
                # Form strengths below 1 must not release the pinned frames.
                cont,cj=run(dict(exact,latent_path=str(cpp)),size=size,guide_phases=phases,input_video=src['x'][:,-18:],prefix_frames_count=18,
                            input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000,sample_solver=solver,
                            masking_strength=0.5,denoising_strength=0.7)
            tensors,info=cj['pending']['tensors'],cj['pending']['info']
            assert torch.equal(tensors['video'][:,:,:2],tsp['video'][:,:,35:37]), f"{solver}: pinned latents not exact, max diff {float((tensors['video'][:,:,:2]-tsp['video'][:,:,35:37]).abs().max())}, per latent {[float((tensors['video'][:,:,i]-tsp['video'][:,:,35+i]).abs().max()) for i in range(2)]}"
            assert not torch.equal(tensors['video'][:,:,2:4],tsp['video'][:,:,35:37]), 'new latents must be generated'
            if phases==2:
                # Phase 1 follows the final video: the last 5 source frames, resized to phase 1 (32x32), natively encoded and pinned.
                assert torch.equal(pipe.vae.last_input[0],namespace['_resize_video'](src['x'][:,-5:],32,32).to(pipe.vae.last_input))
                assert torch.equal(tensors['video_phase1'][:,:,:2],pipe.vae.last_output[:,:,:2].to(tensors['video_phase1'])), 'phase-1 pinned latents not the re-encode'
                assert 'phase 1 follows the final video' in log.getvalue()
            assert torch.equal(tensors['audio'][...,:stop-start],tsp['audio'][...,start:stop]), f'{solver}: frozen audio changed'
            assert abs(info['audio_origin_seconds']-(origin+4/24))<1e-9 and info['audio_origin_seconds']<0
            assert {k:info['effective_context']['pinned'][k] for k in ('frames','latents','first_latent','clip_start','overlap')}=={'frames':5,'latents':2,'first_latent':35,'clip_start':119,'overlap':18}
            assert all(p<0 for p in info['effective_context']['video_frame_positions'])
            # 124-frame window, 13 history copies: 111 requested -> 107 generated (5+17*6), 4 frames shorter, nothing cut.
            assert info['target_frames']==107 and info['effective_context']['pinned'].get('requested_target_frames')==111
            assert 'generating 107 so the scene ends where the video ends (the new part is 4 frame(s) shorter' in log.getvalue()
            assert cont['x'].shape==(3,120,size,size) and cont['audio'].shape==(FULL_AUDIO-round(4/24*32000),2)
            assert cj['part2']['overlap']==18 and cj['part2']['history']==13
            assert encodes==(pipe.vae.encode_calls,pipe.audio_vae.encode_calls), 'exact join re-encoded AV beyond the phase-1 pinned frames'
            assert 'exact join: Windows Frames Overlap 18 -> window starts on source frame 119' in log.getvalue()
            assert 'seamless end skipped: set WanGP "Windows Frames Overlap" (Sliding Window) to 35 or more' in log.getvalue() and 'seam' not in cj
        # Successive exact joins (the new checkpoint is continued again).
        cpn=Path(td)/f'exact{phases}-next.safetensors';cj['pending']['tensors']['last_frame']=last_frame(cont['x'])
        atomic_save(cpn,cj['pending']['tensors'],cj['pending']['info'])
        _,nj=run(dict(exact,latent_path=str(cpn)),size=size,guide_phases=phases,input_video=cont['x'][:,-18:],prefix_frames_count=18,
                 input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
        mdn,tsn=load_checkpoint(cpn)
        from h3smoke.seams import pinned_geometry
        geo=pinned_geometry(mdn['target_frames'],tsn['video'].shape[2],18)  # 107 delivered frames -> 5 pinned, latents 30..31
        assert (mdn['target_frames'],geo['frames'],geo['first_latent'],geo['latents'])==(107,5,30,2), (mdn['target_frames'],geo)
        assert torch.equal(nj['pending']['tensors']['video'][:,:,:2],tsn['video'][:,:,30:32])
        assert nj['pending']['info']['effective_context']['pinned']['clip_start']==102
    # Two phases, phase 1 from the saved phase-1 draft (option off): exact draft latents, no encode at all.
    encodes=(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    _,dj=run(dict(exact,latent_path=str(Path(td)/'exact2.safetensors'),phase1_from_final=False),size=64,guide_phases=2,
             input_video=src['x'][:,-18:],prefix_frames_count=18,input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
    md2,ts2=load_checkpoint(Path(td)/'exact2.safetensors')
    assert torch.equal(dj['pending']['tensors']['video_phase1'][:,:,:2],ts2['video_phase1'][:,:,35:37])
    assert torch.equal(dj['pending']['tensors']['video'][:,:,:2],ts2['video'][:,:,35:37])
    assert encodes==(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
    # A single-phase checkpoint can now be continued in two phases with Exact join (phase 1 from the final video).
    one,one_job=run({'save':True,'continue':False,'latent_path':''},size=64)
    cps=Path(td)/'single64.safetensors';one_job['pending']['tensors']['last_frame']=last_frame(one['x'])
    atomic_save(cps,one_job['pending']['tensors'],one_job['pending']['info'])
    mds,tss=load_checkpoint(cps)
    _,sj=run(dict(exact,latent_path=str(cps)),size=64,guide_phases=2,input_video=one['x'][:,-18:],prefix_frames_count=18,
             input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
    assert torch.equal(sj['pending']['tensors']['video'][:,:,:2],tss['video'][:,:,35:37]) and 'video_phase1' in sj['pending']['tensors']
    # Longer overlap pins more frames (35 -> 22 frames, 7 latents).
    src1,src1_job=run({'save':True,'continue':False,'latent_path':''})
    cp1=Path(td)/'exact-long.safetensors';src1_job['pending']['tensors']['last_frame']=last_frame(src1['x'])
    atomic_save(cp1,src1_job['pending']['tensors'],src1_job['pending']['info'])
    md1,ts1=load_checkpoint(cp1)
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        lout,lj=run(dict(exact,latent_path=str(cp1)),input_video=src1['x'][:,-35:],prefix_frames_count=35,
                    input_waveform=torch.zeros(48000,2),input_waveform_sample_rate=32000)
    assert torch.equal(lj['pending']['tensors']['video'][:,:,:7],ts1['video'][:,:,30:37]) and lj['part2']['history']==13
    # Seamless end (on by default): the re-decoded last 5 source frames + the switch frame are kept for the export,
    # i.e. window output frames [13 + 22 - 5 - 1, 13 + 22).
    assert 'seamless end: the last 5 source frames' in log.getvalue()
    assert lj['seam']['count']==5 and torch.equal(lj['seam']['frames'],lout['x'][:,29:35])
    _,off=run(dict(exact,latent_path=str(cp1),seamless_end=False),input_video=src1['x'][:,-35:],prefix_frames_count=35,
              input_waveform=torch.zeros(48000,2),input_waveform_sample_rate=32000)
    assert 'seam' not in off, 'Seamless end must be off when unticked'
    # Refusals: overlap too short, audio context shorter than the pinned frames.
    for kw,message in (({'prefix_frames_count':1,'input_video':src1['x'][:,-1:]},'too short'),
                       ({'prefix_frames_count':35,'input_video':src1['x'][:,-35:],'_seconds':0.5},'audio context')):
        opts=dict(exact,latent_path=str(cp1),audio_context_seconds=kw.pop('_seconds',1.0))
        try: run(opts,input_waveform=torch.zeros(48000,2),input_waveform_sample_rate=32000,**kw)
        except ValueError as error: assert message in str(error), error
        else: raise AssertionError(f'exact join should refuse: {message}')
print('PASS: exact join: window on a VAE chunk boundary, last source latents pinned bit-exact (euler, res_multistep, '
      'ralston_2s; one and two phases; form strengths < 1), frozen audio shifted with the window, 22-frame pin with a 35 overlap, '
      'two phases with phase 1 from the final video (pinned frames re-encoded at phase-1 size) or from the saved draft, '
      'single-phase checkpoint continued in two phases, successive joins, WGP overlap bookkeeping, explicit refusals, no AV re-encode.')

# ---- Wan2GP 19.x Pika H3 Flex sampler (6 fixed steps, own noise and renoise) with latent operations ----
if 'PIKA_SOLVER' in namespace:
    pika=dict(sample_solver=namespace['PIKA_SOLVER'],sampling_steps=namespace['PIKA_STEPS'],shift=namespace['PIKA_VIDEO_SHIFT'])
    with tempfile.TemporaryDirectory() as td:
        psrc,psrc_job=run({'save':True,'continue':False,'latent_path':''},**pika)
        assert torch.isfinite(psrc_job['pending']['tensors']['video']).all()
        cpk=Path(td)/'pika.safetensors';psrc_job['pending']['tensors']['last_frame']=last_frame(psrc['x'])
        atomic_save(cpk,psrc_job['pending']['tensors'],psrc_job['pending']['info'])
        mdk,tsk=load_checkpoint(cpk)
        for mode in ('baseline','context','audio_prefix','pinned'):
            log=io.StringIO()
            with contextlib.redirect_stdout(log):
                pout,pj=run({'save':True,'continue':True,'latent_path':str(cpk),'join_mode':mode,'video_context_frames':35,
                             'audio_context_seconds':1.0},input_video=psrc['x'][:,-18:],prefix_frames_count=18,
                            input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000,**pika)
            tensors=pj['pending']['tensors']
            if mode=='pinned':
                assert torch.equal(tensors['video'][:,:,:2],tsk['video'][:,:,35:37]), \
                    f"pika: pinned latents not exact, max diff {float((tensors['video'][:,:,:2]-tsk['video'][:,:,35:37]).abs().max())}"
            if mode in ('audio_prefix','pinned'):
                start,stop,_=audio_window(mdk,tsk['audio'].shape[-1],1.0)
                assert torch.equal(tensors['audio'][...,:stop-start],tsk['audio'][...,start:stop]), f'pika {mode}: frozen audio changed'
    print('PASS: Pika H3 Flex sampler (19.x): save, continuation in all four join modes, exact join pinned latents and frozen audio exact.')

# ---- Ref2VA with an image reference and a voice (audio guide) reference: all join modes ----
from PIL import Image as _Image
ref_image=_Image.new('RGB',(32,32),(120,40,40))
pipe.reference_mode=True
with tempfile.TemporaryDirectory() as td:
    voice=dict(input_ref_images=[ref_image],audio_guide=torch.zeros(1,2,32000),audio_prompt_type='A')
    srcr,srcr_job=run({'save':True,'continue':False,'latent_path':''},**voice)
    cpv=Path(td)/'voice.safetensors';srcr_job['pending']['tensors']['last_frame']=last_frame(srcr['x'])
    atomic_save(cpv,srcr_job['pending']['tensors'],srcr_job['pending']['info'])
    mdv,tsv=load_checkpoint(cpv)
    layouts=Transformer.layouts
    for mode in ('baseline','context','audio_prefix','pinned'):
        _,vj=run({'save':True,'continue':True,'latent_path':str(cpv),'join_mode':mode,'video_context_frames':35,'audio_context_seconds':1.0},
                 input_video=srcr['x'][:,-18:],prefix_frames_count=18,input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000,**voice)
        if mode=='pinned':
            assert torch.equal(vj['pending']['tensors']['video'][:,:,:2],tsv['video'][:,:,35:37])
        if mode in ('audio_prefix','pinned'):
            start,stop,_=audio_window(mdv,tsv['audio'].shape[-1],1.0)
            assert torch.equal(vj['pending']['tensors']['audio'][...,:stop-start],tsv['audio'][...,start:stop])
    assert Transformer.layouts>layouts
pipe.reference_mode=False
# FL2VA: an audio guide that drives the generated soundtrack conflicts with the frozen audio -> clear refusal.
with tempfile.TemporaryDirectory() as td:
    srcf,srcf_job=run({'save':True,'continue':False,'latent_path':''})
    cpf=Path(td)/'fl.safetensors';srcf_job['pending']['tensors']['last_frame']=last_frame(srcf['x'])
    atomic_save(cpf,srcf_job['pending']['tensors'],srcf_job['pending']['info'])
    try:
        run({'save':False,'continue':True,'latent_path':str(cpf),'join_mode':'pinned','video_context_frames':35,'audio_context_seconds':1.0},
            input_video=srcf['x'][:,-18:],prefix_frames_count=18,input_waveform=torch.zeros(24000,2)+0.1,input_waveform_sample_rate=32000,
            audio_prompt_type='A')
    except ValueError as error:
        assert 'soundtrack that drives' in str(error), error
    else:
        raise AssertionError('FL2VA soundtrack conditioning with frozen audio must be refused')
print('PASS: Ref2VA with image + voice references: all four join modes run, layouts with references checked, '
      'pinned video and frozen audio exact.')

# ---- Audio Refinement Extra Phase (native, or H3 Audio Refinement Controls if supplied) ----
import contextlib, io
refine_value='enabled:2:0.3' if refinement_plugin is not None else 'enabled'
refine_steps,refine_denoise=(2,0.3) if refinement_plugin is not None else (6,0.5)
expected_sigma=3.0*refine_denoise/(1.0+2.0*refine_denoise)  # audio shift 3 applied to the start of the schedule
def refined(opts,**kw):
    Transformer.refinement=[]
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        result,result_job=run(opts,custom_settings={'audio_refinement':refine_value},**kw)
    passes=list(Transformer.refinement)
    assert len(passes)==refine_steps, f'{len(passes)} refinement steps, expected {refine_steps}'
    assert abs(passes[0]['sigma_audio']-expected_sigma)<1e-5, (passes[0]['sigma_audio'],expected_sigma)
    return result,result_job,passes,log.getvalue()
with tempfile.TemporaryDirectory() as td:
    # Native path (latent options off) keeps the refinement exactly as installed.
    refined({'save':False,'continue':False,'latent_path':''})
    base,base_job,passes,_=refined({'save':True,'continue':False,'latent_path':''})
    assert passes[0]['audio_keyframes'] is None, 'A render without latent join must refine natively'
    cpr=Path(td)/'refined.safetensors';base_job['pending']['tensors']['last_frame']=last_frame(base['x'])
    atomic_save(cpr,base_job['pending']['tensors'],base_job['pending']['info'])
    mdr,tsr=load_checkpoint(cpr)
    # The checkpoint stores the refined audio (capture happens after the refinement).
    assert torch.equal(tsr['audio'],base_job['pending']['tensors']['audio'])
    source=base['x'][:,-18:]
    for mode in ('baseline','context','audio_prefix'):
        opts={'save':True,'continue':True,'latent_path':str(cpr),'join_mode':mode,'video_context_frames':35,'audio_context_seconds':1.0}
        encodes=(pipe.vae.encode_calls,pipe.audio_vae.encode_calls)
        cont,cont_job,passes,log=refined(opts,input_video=source,prefix_frames_count=18,
                                         input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
        assert encodes==(pipe.vae.encode_calls,pipe.audio_vae.encode_calls), f'{mode}: refinement re-encoded AV'
        assert 'Audio refinement keeps the join' in log, log
        audio=cont_job['pending']['tensors']['audio']
        if mode=='audio_prefix':
            start,stop,origin=audio_window(mdr,tsr['audio'].shape[-1],1.0)
            assert all(p['target_audio_condition_latents']==stop-start for p in passes)
            assert all(abs(p['lc_audio_offset']-origin*40)<1e-9 and p['lc_audio_offset']<0 for p in passes)
            assert torch.equal(audio[...,:stop-start],tsr['audio'][...,start:stop]), 'Frozen prefix changed by the refinement'
            assert cont['audio'].shape==(FULL_AUDIO,2)
        else:
            assert passes[0]['audio_keyframes'], f'{mode}: join audio conditions dropped by the refinement'
            assert all(p['lc_audio_offset']==0.0 for p in passes)
    # Exact join + refinement: pinned video stays exact (video frozen), frozen audio restored, shifted offset kept.
    start,stop,origin=audio_window(mdr,tsr['audio'].shape[-1],1.0)
    _,pj,passes,log=refined({'save':True,'continue':True,'latent_path':str(cpr),'join_mode':'pinned','video_context_frames':35,
                             'audio_context_seconds':1.0},input_video=source,prefix_frames_count=18,
                            input_waveform=torch.zeros(24000,2),input_waveform_sample_rate=32000)
    assert torch.equal(pj['pending']['tensors']['video'][:,:,:2],tsr['video'][:,:,35:37])
    assert torch.equal(pj['pending']['tensors']['audio'][...,:stop-start],tsr['audio'][...,start:stop])
    assert all(p['target_audio_condition_latents']==stop-start and abs(p['lc_audio_offset']-(origin+4/24)*40)<1e-9 for p in passes)
    # Two phases + refinement.
    two,two_job,passes,_=refined({'save':True,'continue':False,'latent_path':''},size=64,guide_phases=2)
    assert 'video_phase1' in two_job['pending']['tensors']
print(f'PASS: audio refinement ({refine_value}{" via H3 Audio Refinement Controls" if refinement_plugin else ", native"}): '
      f'{refine_steps} steps from sigma {expected_sigma:.4f}; native when no latent join; refined audio saved; join conditions, '
      'frozen prefix and its time offset kept in all three modes and the exact join; two phases; no AV re-encode.')

# ---- Audio noise as WanGP 18.18 (seed reproduction of renders made before the 18.19 audio tail) ----
TAIL=int(namespace.get('H3_AUDIO_TAIL_LATENTS',0))
def first_audio(opts,**kw):
    Transformer.audio_inputs=[]
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        out,out_job=run(opts,**kw)
    return Transformer.audio_inputs[0],out,out_job,log.getvalue()
def seeded_audio(audio_latents):
    # Native draw order (no references, no keyframes): video noise, then channel-major audio rows.
    generator=torch.Generator(device='cpu').manual_seed(19)
    torch.randn((1,24,namespace['video_latent_frames'](124),2,2),generator=generator)
    return namespace['unpack_audio'](torch.randn((audio_latents*2,32),generator=generator))
native_audio,native_out,native_job,_=first_audio({'save':True,'continue':False,'latent_path':''})
assert torch.equal(native_audio,seeded_audio(native_audio.shape[-1])), 'harness draw order differs from native'
old_audio,old_out,old_job,log=first_audio({'save':True,'continue':False,'latent_path':'','audio_noise_1818':True})
assert old_audio.shape==native_audio.shape, 'the audio tail must still be generated'
assert old_job['pending']['info']['settings']['audio_noise']==('wangp-18.18')
assert native_job['pending']['info']['settings']['audio_noise']=='native'
if TAIL:
    legacy=native_audio.shape[-1]-TAIL
    assert 'Audio noise as WanGP 18.18: seeded noise drawn for' in log, log
    assert torch.equal(old_audio[...,:legacy],seeded_audio(legacy)), 'audio noise differs from the WanGP 18.18 draw'
    assert not torch.equal(native_audio[:,:,1,:legacy],old_audio[:,:,1,:legacy]), 'the 18.19 draw should differ on channel 1'
    assert torch.equal(native_audio[:,:,0,:legacy],old_audio[:,:,0,:legacy])  # channel 0 rows come first: unchanged
    assert old_audio[...,legacy:].std()>0.5 and torch.isfinite(old_audio).all()
    # Same video noise and same seeded generator use: video identical.
    assert torch.equal(native_out['x'],old_out['x'])
else:
    assert torch.equal(native_audio,old_audio) and 'no audio tail in this WanGP version' in log
# The option alone (no Save, no Continue) applies to plain renders: Text Only, Start with Image, later windows.
def noise_only(**kw):
    Transformer.audio_inputs=[]
    token=JOB.set({'enabled':True,'noise_only':True,'options':{'save':False,'continue':False,'latent_path':'','audio_noise_1818':True},
                   'pending':None,'part2':None,'wgp_overlap':0,'output_seen':False,'saved':[],'source':''})
    log=io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            out=pipe.generate('test',frame_num=124,width=32,height=32,sampling_steps=2,seed=19,**kw)
    finally: JOB.reset(token)
    return Transformer.audio_inputs[0],out,log.getvalue()
def native_first(**kw):
    Transformer.audio_inputs=[]
    out=pipe.generate('test',frame_num=124,width=32,height=32,sampling_steps=2,seed=19,**kw)
    return Transformer.audio_inputs[0],out
plain_audio,plain_out,plain_log=noise_only()
assert torch.equal(plain_audio,old_audio), 'Text Only: noise differs from the 18.18 draw'
assert torch.equal(plain_out['x'],native_out['x'])
start=torch.linspace(-1,1,3*32*32).view(3,1,32,32)
image_audio,image_out,_=noise_only(image_start=start)
native_image_audio,native_image_out=native_first(image_start=start)
if TAIL:
    legacy=native_audio.shape[-1]-TAIL
    assert 'Audio noise as WanGP 18.18: seeded noise drawn for' in plain_log
    assert torch.equal(image_audio[:,:,0,:legacy],native_image_audio[:,:,0,:legacy]) and not torch.equal(image_audio[:,:,1,:legacy],native_image_audio[:,:,1,:legacy])
assert torch.equal(image_out['x'],native_image_out['x']), 'Start with Image: video must be unchanged'
_,later,_=noise_only(window_no=3)  # sliding windows: later windows are plain renders, no latent restriction
assert later is not None
# Refinement re-noise and er_sde step noise use the same rule (run through the edited lines).
with tempfile.TemporaryDirectory() as td:
    Transformer.refinement=[]
    with contextlib.redirect_stdout(io.StringIO()):
        run({'save':True,'continue':False,'latent_path':'','audio_noise_1818':True},custom_settings={'audio_refinement':refine_value})
    assert Transformer.refinement, 'refinement did not run'
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            run({'save':True,'continue':False,'latent_path':'','audio_noise_1818':True},sample_solver='er_sde')
        except ValueError as error:
            # native er_sde returns NaN with these stand-ins (now reported explicitly); only the noise shapes are exercised here
            assert 'invalid values (NaN/inf)' in str(error), error
print(f'PASS: audio noise as WanGP 18.18: seeded audio noise = the pre-18.19 draw on {"the first %d latents" % (native_audio.shape[-1]-TAIL) if TAIL else "all latents (no tail)"}, '
      'tail from its own generator, video unchanged, recorded in the checkpoint; also alone for Text Only, Start with Image and later windows; refinement and er_sde paths run.')

# ---- NaN produced by the model: explicit stop before decoding (not an ambiguous output tail) ----
for phases,stage in ((1,'at the end of denoising'),(2,'at the end of phase 1')):
    Transformer.nan_audio=True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run({'save':True,'continue':False,'latent_path':''},size=64 if phases==2 else 32,guide_phases=phases)
    except ValueError as error:
        message=str(error)
    else:
        raise AssertionError('NaN latents not reported')
    finally:
        Transformer.nan_audio=False
    assert 'invalid values (NaN/inf) '+stage in message and 'audio:' in message and 'audio latents 3..5' in message, message
print('PASS: NaN/inf from the model stops the render before decoding with the stage and the affected latents named.')

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
