"""CPU tests; run: python -m unittest discover -s tests -v"""
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h3test',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
pkg=importlib.util.module_from_spec(spec);sys.modules['h3test']=pkg;spec.loader.exec_module(pkg)
from h3test.checkpoint import atomic_save, load_checkpoint, inspect_checkpoint, sha256_file, tail_indices, token_start
from h3test.latent_runtime import LatentMixin, JOB, frame_hash
from h3test.integration import make_generation_wrapper, make_save_wrapper, make_record_wrapper, make_prepare_wrapper, preflight, options, state_mapping
from h3test import packing
from h3test.integration import TASK_KEY, KEY, make_queue_wrapper, remember_options
from h3test.export_frames import export_tail


def tensors():
    return {'video':torch.randn(1,24,37,2,2), 'audio':torch.randn(1,32,2,207),
            'last_frame':torch.zeros(3,1,32,32)}


def info():
    return {'target_frames':124,'width':32,'height':32,'fps':24.0,'identity':{'architecture':'test'}}


class Checkpoints(unittest.TestCase):
    def test_legacy_v1_and_v2_audio_origin(self):
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'legacy.safetensors'
            save_file(tensors(),str(path),metadata={'format':'wan2gp.h3.latent-continuation','version':'1','manifest':json.dumps(info())})
            md,loaded=load_checkpoint(path)
            self.assertEqual(md.get('audio_origin_seconds',0),0)
            self.assertEqual(loaded['audio'].shape[-1],207)
            bad=info();bad['audio_origin_seconds']=float('nan')
            path2=pathlib.Path(td)/'bad.safetensors'
            atomic_save(path2,tensors(),bad)
            with self.assertRaisesRegex(ValueError,'timeline origin'): load_checkpoint(path2)

    def test_lossless_roundtrip_all_dtypes(self):
        with tempfile.TemporaryDirectory() as td:
            for dtype in (torch.float32,torch.bfloat16,torch.float16):
                ts={k:v.to(dtype) for k,v in tensors().items()}
                path=pathlib.Path(td)/f'{dtype}.safetensors'
                atomic_save(path,ts,info())
                md,loaded=load_checkpoint(path)
                for key in ts:
                    self.assertEqual(loaded[key].dtype,dtype)
                    self.assertTrue(torch.equal(loaded[key],ts[key]))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'clip.safetensors'
            atomic_save(path,tensors(),info()); previous=path.read_bytes()
            with self.assertRaises(FileExistsError): atomic_save(path,tensors(),info())
            self.assertEqual(previous,path.read_bytes())

    def test_lora_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'lora.safetensors';save_file({'lora.up':torch.zeros(2,2)},str(path))
            with self.assertRaises(ValueError): inspect_checkpoint(path)

    def test_bad_geometry_and_nan(self):
        with tempfile.TemporaryDirectory() as td:
            for index, change in enumerate(('geometry','nan')):
                ts=tensors();md=info()
                if change=='geometry': md['width']=64
                else: ts['video'][0,0,0,0,0]=float('nan')
                path=pathlib.Path(td)/f'{index}.safetensors';atomic_save(path,ts,md)
                with self.assertRaises(ValueError): load_checkpoint(path)

    def test_video_pair_binding(self):
        with tempfile.TemporaryDirectory() as td:
            video=pathlib.Path(td)/'a.mp4';video.write_bytes(b'original test file')
            path=pathlib.Path(td)/'a.safetensors';md=info();md['video_sha256']=sha256_file(video)
            atomic_save(path,tensors(),md)
            params={'guidance_phases':1,'image_prompt_type':'V','video_source':str(video)}
            opts={'save':False,'continue':True,'latent_path':str(path)}
            preflight(params,opts)
            video.write_bytes(b'changed file')
            with self.assertRaises(ValueError): preflight(params,opts)


class Timing(unittest.TestCase):
    def test_native_grid(self):
        self.assertEqual([token_start(i) for i in range(7)],[0,1,5,9,13,17,18])
        self.assertEqual(token_start(37),124)

    def test_boundary_phase_kept(self):
        items=tail_indices(37,124,18)
        self.assertEqual(items,[(31,-20),(32,-16),(33,-12),(34,-8),(35,-4),(36,-3)])
        self.assertEqual(tail_indices(37,124,1),[(36,-3)])

    def test_trimmed_endpoint_does_not_use_future_tokens(self):
        for frames in range(90,125):
            tail=tail_indices(37,frames,min(frames,18))
            self.assertTrue(all(token_start(i)<frames for i,_ in tail))
            for i,offset in tail: self.assertEqual(offset,token_start(i)-(frames-1))

    def test_video_conditions_preserve_values(self):
        mix=LatentMixin();mix._lc_source=tensors();mix._lc_manifest=info()
        latents,anchors=[],[];mix._lc_video_conditions(latents,anchors,18)
        for latent,anchor,(index,offset) in zip(latents,anchors,tail_indices(37,124,18)):
            self.assertTrue(torch.equal(latent,mix._lc_source['video'][:,:,index:index+1]))
            self.assertEqual(anchor['frame_index'],offset)

    def test_audio_end_alignment(self):
        mix=LatentMixin();mix._lc_source=tensors();mix._lc_manifest=info()
        latents,anchors=[],[];mix._lc_audio_conditions(latents,anchors,18,24)
        self.assertEqual(anchors[0]['anchor'],176-205)
        self.assertTrue(torch.equal(latents[0],mix._lc_source['audio'][...,176:207]))
        positions=torch.zeros(62,3,dtype=torch.float64)
        packing._fill_audio_condition_positions(positions,0,[(anchors[0]['anchor'],31)],10,100,torch.tensor([0.,1.]))
        self.assertEqual(positions[0,0].item(),71)
        self.assertEqual(positions[30,0].item(),101)

    def test_packing_accepts_negative_frame_anchors(self):
        # Native layout functions must work both with keyframes and Ref2VA references.
        for builder in (packing.build_packed_sequence,packing.build_ref2va_packed_sequence):
            args=[torch.zeros(3,dtype=torch.long)]
            if builder is packing.build_ref2va_packed_sequence: args.append([])
            args.extend([37,2,2,207,(1,2,2)])
            layout=builder(*args,keyframe_anchors=(('frame',1,-3),),audio_condition_anchors=((-29.,31),))
            self.assertTrue(torch.isfinite(layout.position_ids).all())


class Integration(unittest.TestCase):
    def test_export_chunk_tail_and_count(self):
        video=torch.arange(3*20*32*32,dtype=torch.int32).remainder(256).to(torch.uint8).reshape(3,20,32,32)
        for chunks in ([video], (video[:,:15],None,video[:,15:18],video[:,18:]),
                       [video[:,:17].permute(1,2,3,0),video[:,17:].permute(1,2,3,0)]):
            tail,count=export_tail(chunks)
            self.assertEqual(count,20)
            self.assertTrue(torch.equal(tail,video[:,-8:]))
        for invalid in ([],[None],[torch.zeros(3,1,32,32)],
                        [video,torch.zeros(3,1,64,64,dtype=torch.uint8)],
                        [torch.zeros(3,0,32,32,dtype=torch.uint8)]):
            with self.assertRaises(ValueError): export_tail(invalid)

    def test_chunk_export_and_final_mux_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary=pathlib.Path(directory)/'render_tmp.mp4'
            final=pathlib.Path(directory)/'render.mp4'
            decoded=torch.rand(3,124,32,32)*2-1
            pending={'tensors':{k:v for k,v in tensors().items() if k!='last_frame'},'info':info()}
            job={'pending':pending,'source':'','saved':[]}
            token=JOB.set(job)
            try:
                LatentMixin()._lc_decoded(decoded)
                output=decoded[:,:122].half().add(1).mul(127.5).clamp(0,255).to(torch.uint8)
                chunks=[torch.zeros(3,9,32,32,dtype=torch.uint8),output[:,:119],output[:,119:]]
                def encode(tensor,save_file,fps):
                    self.assertIs(tensor,chunks)
                    pathlib.Path(save_file).write_bytes(b'encoded')
                def metadata(video_path,configs):
                    with open(video_path,'ab') as stream:stream.write(b'metadata')
                make_save_wrapper(encode)(chunks,str(temporary),24)
                temporary.replace(final)  # Stand-in for native final audio mux.
                make_record_wrapper(metadata)(str(final),{})
                md,loaded=load_checkpoint(final.with_suffix('.safetensors'))
                self.assertEqual(md['output_frames'],131)
                self.assertEqual(md['target_frames'],122)
                self.assertEqual(md['video_sha256'],sha256_file(final))
                self.assertFalse(temporary.with_suffix('.safetensors').exists())
                self.assertTrue(torch.equal(loaded['video'],pending['tensors']['video']))
            finally: JOB.reset(token)

    def test_missing_task_options_stop_before_render(self):
        def render(task,model_type,plugin_data=None):
            self.fail('No GPU work may start without explicit options')
        wrapped=make_generation_wrapper(render,lambda x:x)
        with self.assertRaisesRegex(RuntimeError,'lost its options'):
            wrapped({'params':{}},'h3_latent_fl2va',{})

    def test_kwargs_wrapper_and_frozen_task_options(self):
        seen=[]
        def render(task, **kwargs):
            seen.append(JOB.get()['options'].copy())
            return False  # cancelled: no output expected
        wrap=make_generation_wrapper(render,lambda x:x)
        wrap({},model_type='h3_latent_fl2va',plugin_data={KEY:{'save':True}})
        task={'params':{TASK_KEY:{'model_type':'h3_latent_fl2va','options':{'save':False,'continue':False,'latent_path':''}}}}
        wrap(task,model_type='h3_latent_fl2va',plugin_data={KEY:{'save':True}})
        self.assertEqual([x['save'] for x in seen],[True,False])
        with self.assertRaisesRegex(ValueError,'mismatch'):
            wrap(task,model_type='h3_latent_ref2va',plugin_data={})

    def test_enqueue_recovers_from_session_and_freezes_options(self):
        state={'model_type':'h3_latent_fl2va'}
        remember_options(state,state['model_type'],{KEY:{'save':True}})
        queued=[]
        enqueue=make_queue_wrapper(lambda **inputs:queued.append(inputs),lambda x:x,lambda st:st['model_type'])
        enqueue(state=state,model_type=state['model_type'],plugin_data={'other':7})
        remember_options(state,state['model_type'],{KEY:{'save':False}})
        self.assertTrue(queued[0][TASK_KEY]['options']['save'])
        self.assertEqual(queued[0]['plugin_data']['other'],7)
        with self.assertRaisesRegex(RuntimeError,'no captured options'):
            enqueue(state={'model_type':state['model_type']},model_type=state['model_type'])

    def test_options_malformed_payloads_default_disabled(self):
        for payload in (None, [], 4, 'wrong', {'h3_latent_prototype':None}, {'h3_latent_prototype':[]}):
            with self.subTest(payload=payload):
                self.assertEqual(options(payload),{'save':False,'continue':False,'latent_path':''})

    def test_mapping_preserves_unrelated_plugin_keys(self):
        source={'another_plugin':{'setting':9},'h3_latent_prototype':{'save':True}}
        result=state_mapping(source)
        self.assertEqual(result,source)
        self.assertIsNot(result,source)
        self.assertTrue(options(result)['save'])

    def test_form_options_survive_native_cleaner(self):
        def native(target,inputs,model_type=None,model_filename=None):
            inputs.pop('state');inputs.pop('plugin_data',None)
            return inputs
        wrap=make_prepare_wrapper(native,lambda x:x,lambda st:st['model_type'])
        options={'h3_latent_prototype':{'save':True,'continue':False,'latent_path':''}}
        result=wrap('state',{'state':{'model_type':'h3_latent_fl2va'},'plugin_data':options})
        self.assertEqual(result['plugin_data'],options)
        options['h3_latent_prototype']['save']=False
        self.assertTrue(result['plugin_data']['h3_latent_prototype']['save'])
        with self.assertRaises(ValueError):
            wrap('state',{'state':{'model_type':'h3_latent_fl2va'},'guidance_phases':2})

    def test_two_phases_and_unsupported_rejected(self):
        opts={'save':True,'continue':False,'latent_path':''}
        for field,value in [('guidance_phases',2),('spatial_upsampling','x2'),('temporal_upsampling','x2'),
                            ('batch_size',2),('film_grain_intensity',0.2),('keep_frames_video_source','-10')]:
            with self.subTest(field=field),self.assertRaises(ValueError): preflight({field:value},opts)
        with self.assertRaises(ValueError): preflight({}, {'save':False,'continue':True,'latent_path':''})

    def test_job_options_are_isolated_and_reset_on_error(self):
        seen=[]
        def original(task,model_type,plugin_data=None,guidance_phases=1):
            seen.append(JOB.get()['options'].copy())
            raise RuntimeError('cancelled')
        wrapped=make_generation_wrapper(original,lambda x:x)
        for use in (True,False):
            with self.assertRaises(RuntimeError):
                wrapped({},'h3_latent_fl2va',{'h3_latent_prototype':{'save':use}})
            self.assertIsNone(JOB.get())
        self.assertEqual([x['save'] for x in seen],[True,False])

    def test_native_model_bypass(self):
        def original(task,model_type,plugin_data=None):
            self.assertIsNone(JOB.get()); return 'native'
        self.assertEqual(make_generation_wrapper(original,lambda x:x)({},'native'), 'native')

    def test_sidecar_final_name_metadata_and_trim(self):
        with tempfile.TemporaryDirectory() as td:
            video=pathlib.Path(td)/'final_name.mp4'
            pending={'tensors':{k:v for k,v in tensors().items() if k!='last_frame'},'info':info()}
            decoded=torch.rand(3,124,32,32)*2-1
            mix=LatentMixin()
            job={'pending':pending,'source':'','saved':[]}
            token=JOB.set(job)
            try:
                mix._lc_decoded(decoded)
                def save_video(tensor,save_file,fps=24): pathlib.Path(save_file).write_bytes(b'encoded')
                def record_file_metadata(video_path,configs):
                    with open(video_path,'ab') as f: f.write(b'metadata')
                # Exact native fp16-to-uint8 conversion, trimming two frames at the end.
                output=decoded[:,:122].half().add(1).mul(127.5).clamp(0,255).to(torch.uint8)
                make_save_wrapper(save_video)(output,str(video),24)
                make_record_wrapper(record_file_metadata)(str(video),{})
                md,ts=load_checkpoint(video.with_suffix('.safetensors'))
                self.assertEqual(md['target_frames'],122)
                self.assertEqual(md['output_frames'],122)
                self.assertEqual(md['video_sha256'],sha256_file(video))
                self.assertTrue(torch.equal(ts['video'],pending['tensors']['video']))
                self.assertIsNone(job['pending'])
            finally: JOB.reset(token)

    def test_ambiguous_static_tail_refused(self):
        token=JOB.set({'pending':{'info':info(),'tensors':{}}})
        try:
            LatentMixin()._lc_decoded(torch.zeros(3,124,32,32))
            def save_video(tensor,fps=24): self.fail('Ambiguous tail must not be accepted')
            with self.assertRaisesRegex(ValueError,'Ambiguous'):
                make_save_wrapper(save_video)(torch.zeros(3,122,32,32))
        finally: JOB.reset(token)

    def test_changed_tail_refused_before_export(self):
        pending={'info':info(),'tail_signatures':{('never',):[124]},'tensors':{}}
        token=JOB.set({'pending':pending})
        try:
            def save_video(tensor,fps=24): self.fail('Should reject before encoding')
            with self.assertRaises(ValueError): make_save_wrapper(save_video)(torch.zeros(3,124,32,32))
        finally: JOB.reset(token)

if __name__=='__main__': unittest.main()
