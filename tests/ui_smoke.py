"""Real Gradio/native plugin API smoke test; no model weights or GPU required.
Usage: python tests/ui_smoke.py /path/to/Wan2GP
"""
import importlib.util
import ast
import os
from pathlib import Path
import sys
import types
import inspect
import typing
import asyncio
import tempfile
import torch
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
ROOT=Path(__file__).resolve().parents[1]
wangp=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(wangp))
import gradio as gr
# Isolate package initializers importing unrelated inference engines; the plugin API
# and its insertion method below are loaded unchanged from the supplied source.
for name, path in [('shared',wangp/'shared'),('shared.utils',wangp/'shared/utils')]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
# Avoid the config migration import, which eagerly imports SeedVC/torchaudio.
# No plugin API body is changed and no installation/migration method is invoked.
tree=ast.parse((wangp/'shared/utils/plugins.py').read_text())
tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.module=='shared.utils.wgp_config_migration')]
api=types.ModuleType('shared.utils.plugins');sys.modules[api.__name__]=api
exec(compile(tree,str(wangp/'shared/utils/plugins.py'),'exec'),api.__dict__)
PluginManager=api.PluginManager
spec=importlib.util.spec_from_file_location('h3uitest',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
pkg=importlib.util.module_from_spec(spec);sys.modules['h3uitest']=pkg;spec.loader.exec_module(pkg)
from h3uitest.plugin import H3LatentPlugin

plugin=H3LatentPlugin();plugin.setup_ui()
assert not plugin.tabs
plugin.get_base_model_type=lambda name:name
plugin.get_state_model_type=lambda state:state['model_type']
plugin.generate_media=lambda task,model_type,plugin_data=None,guidance_phases=1:True
plugin.save_video=lambda tensor,save_file=None,fps=24:None
plugin.record_file_metadata=lambda video_path,configs:None
plugin.prepare_inputs_dict=lambda target,inputs,model_type=None,model_filename=None:inputs
plugin.get_model_settings=lambda state,model_type:state.get('all_settings',{}).get(model_type)
installed={}
plugin._set_wgp_global_func=lambda name,fn:installed.__setitem__(name,fn)
# Execute the actual native form-save and queue-building bodies, without loading
# the GPU application. Only unrelated field-cleaning/preview dependencies are stubbed.
native_names={'save_inputs','get_function_arguments','get_model_settings','set_model_settings',
              'get_state_model_type','add_video_task'}
native_tree=ast.parse((wangp/'wgp.py').read_text())
native_tree.body=[node for node in native_tree.body if isinstance(node,ast.FunctionDef) and node.name in native_names]
native={'inspect':inspect,'typing':typing,'gr':gr,'task_id':0,
        'get_gen_info':lambda state:state.setdefault('gen',{'queue':[]}),
        'get_preview_images':lambda inputs:(None,None,[],[])}
# Load the actual new state container/helper without importing the whole Deepy app.
hybrid=wangp/'shared/deepy/hybrid.py'
StateClass=dict
if hybrid.exists():
    from copy import deepcopy
    hybrid_tree=ast.parse(hybrid.read_text())
    hybrid_tree.body=[node for node in hybrid_tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))
                      and node.name in ('SharedState','service_for')]
    native['deepcopy']=deepcopy
    exec(compile(hybrid_tree,str(hybrid),'exec'),native)
    StateClass=native['SharedState']
exec(compile(native_tree,str(wangp/'wgp.py'),'exec'),native)
plugin.add_video_task=native['add_video_task']
native['prepare_inputs_dict']=lambda *args:installed['prepare_inputs_dict'](*args)
manager=PluginManager.__new__(PluginManager);manager.plugins={'prototype':plugin}
manager.plugins_dir=str(ROOT.parent)
manager._plugin_metadata_cache={}
extensions=manager.discover_plugin_model_extensions([ROOT.name])
assert len(extensions)==1, 'Model plugin must be discovered, not just its UI'
assert Path(extensions[0].defaults_root).is_dir()
assert Path(extensions[0].profiles_root).is_dir()
assert extensions[0].model_handlers==[ROOT.name+'.handler']
with gr.Blocks() as app:
    state=gr.State(StateClass(model_type='h3_latent_fl2va'))
    plugin_data=gr.State({})
    image_prompt_type=gr.Textbox(value='V',visible=False)
    model_choice=gr.Dropdown(choices=['h3_latent_fl2va','native'],value='h3_latent_fl2va')
    with gr.Column() as parent:
        video_source=gr.Video()
        following=gr.Textbox(label='Original following control')
    native_inputs=[]
    for name in inspect.signature(native['save_inputs']).parameters:
        component = state if name=='state' else plugin_data if name=='plugin_data' else gr.State(
            'state' if name=='target' else 0 if name=='image_mode' else 1 if name=='guidance_phases' else None)
        native_inputs.append(component)
    generate=gr.Button('Generate')
    generate.click(native['save_inputs'],inputs=native_inputs,outputs=None)
    components=dict(state=state,plugin_data=plugin_data,image_prompt_type=image_prompt_type,
                    model_choice=model_choice,video_source=video_source)
    manager.run_component_insertion_and_setup(components)
config=app.get_config_file()
checkboxes=[x for x in config['components'] if x['type']=='checkbox']
assert len(checkboxes)==2,checkboxes
assert all(x['props']['value'] is False for x in checkboxes)
assert not any(x['type']=='tabitem' for x in config['components'])
assert any(x['type']=='file' and x['props']['file_types']==['.safetensors'] for x in config['components'])
assert parent.children[0] is video_source
assert type(parent.children[1]).__name__=='Column'
assert following in getattr(parent.children[-1],'children',[])
assert set(installed)=={'generate_media','save_video','record_file_metadata','prepare_inputs_dict','get_model_settings','add_video_task'}
# Updating backend state cannot recursively retrigger the user-only option handlers.
updates=[dep for dep in config['dependencies'] if any(event in ('input','upload','clear') for _,event in dep['targets'])]
assert len(updates)==1
print('PASS: native insertion, Gradio '+gr.__version__+', two unchecked options, latent upload, no new tab, 6 lifecycle bridges.')

# Reproduce the reported callback input using REAL Gradio State objects.
callbacks={fn.fn.__name__:fn.fn for fn in app.fns.values() if fn.fn is not None}
from h3uitest.integration import KEY, options, make_prepare_wrapper
wrapped=gr.State({KEY:{'save':True,'continue':True,'latent_path':''},'another_plugin':{'keep':7}})
assert callbacks['restore'](wrapped)==(True,True,None,'baseline',35,1.0)
assert callbacks['restore'](gr.State())==(gr.skip(),)*6
assert options(gr.State(gr.State({KEY:{'save':True}})))['save'] is True
session={'model_type':'h3_latent_fl2va','all_settings':{'h3_latent_fl2va':{'plugin_data':{'another_plugin':{'keep':7}}}}}
changed=callbacks['update_data'](False,True,None,wrapped,session)
assert changed['another_plugin']=={'keep':7}
assert wrapped.value[KEY]['save'] is True
assert changed[KEY]['save'] is False
# A checkbox alone, with no native field edit, must reach task settings.
callbacks['update_data'](True,False,None,wrapped,session)
queued=installed['get_model_settings'](session,'h3_latent_fl2va')
assert queued['plugin_data'][KEY]['save'] is True
assert queued['plugin_data']['another_plugin']=={'keep':7}
assert KEY not in session['all_settings']['h3_latent_fl2va']['plugin_data']
# A stale native form snapshot cannot erase the latest explicit checkbox choice.
prepared=installed['prepare_inputs_dict']('state',{'state':session,'plugin_data':{}})
assert prepared['plugin_data'][KEY]['save'] is True
callbacks['update_data'](False,False,None,wrapped,session)
assert installed['get_model_settings'](session,'h3_latent_fl2va')['plugin_data'][KEY]['save'] is False
other={'all_settings':{'h3_latent_fl2va':{}}}
assert not options(installed['get_model_settings'](other,'h3_latent_fl2va')['plugin_data'])['save']
assert callbacks['visibility'](gr.State({'model_type':'h3_latent_fl2va'}),'V')[0]['visible'] is True
for malformed in (None,[],42,gr.State([]),{KEY:None},{KEY:'invalid'}):
    expected=(False,False,None,'baseline',35,1.0) if isinstance(malformed,dict) and KEY in malformed else (gr.skip(),)*6
    assert callbacks['restore'](malformed)==expected
def clean(target,inputs,model_type=None,model_filename=None):
    inputs.pop('plugin_data',None)
    return inputs
cleaned=make_prepare_wrapper(clean,lambda x:x,lambda st:st['model_type'])(
    'state',{'state':{'model_type':'h3_latent_fl2va'},'plugin_data':wrapped})
assert isinstance(cleaned['plugin_data'],dict)
assert cleaned['plugin_data'][KEY]['save'] is True
print('PASS: native model discovery + real State restoration/update/visibility/queue normalization; other plugin data preserved.')

# Unlike the older tests, do NOT invoke update_data first. Browser checkbox
# values must win over stale plugin_data through actual Gradio processing.
from gradio.state_holder import SessionState
save_event=next(event for event in app.fns.values() if getattr(event.fn,'_h3_form_bound',False))
assert len(save_event.inputs)==len(native_inputs)+6
async def exercise_submission():
    session=SessionState(app)
    session[state._id]=StateClass(model_type='h3_latent_fl2va')
    session[plugin_data._id]={KEY:{'save':False},'another_plugin':{'keep':7}}
    await app.process_api(save_event,[None]*len(native_inputs)+[True,False,None,'baseline',35,1.0],state=session)
    st=session[state._id]
    settings=installed['get_model_settings'](st,'h3_latent_fl2va')
    assert options(settings['plugin_data'])['save'] is True
    assert settings['plugin_data']['another_plugin']=={'keep':7}
    settings.update(state=st,model_type='h3_latent_fl2va')
    # Reproduce shared plugin_data disappearing AFTER successful form capture.
    settings.pop('plugin_data')
    installed['add_video_task'](**settings)
    task=st['gen']['queue'][0]
    assert options(task['plugin_data'])['save'] is True
    # A later checkbox change must not modify an already queued task.
    await app.process_api(save_event,[None]*len(native_inputs)+[False,False,None,'baseline',35,1.0],state=session)
    assert options(installed['get_model_settings'](st,'h3_latent_fl2va')['plugin_data'])['save'] is False
    assert options(task['plugin_data'])['save'] is True
    from h3uitest.integration import make_generation_wrapper,make_save_wrapper,make_record_wrapper
    from h3uitest.latent_runtime import LatentMixin,JOB
    from h3uitest.checkpoint import load_checkpoint,sha256_file
    with tempfile.TemporaryDirectory() as directory:
        destination=Path(directory)/'custom_output'/'render.mp4'
        destination.parent.mkdir()
        original_video=torch.randn(1,24,37,2,2)
        original_audio=torch.randn(1,32,2,207)
        def render(task,model_type,plugin_data=None,guidance_phases=1):
            job=JOB.get()
            assert job['options']['save'] is True
            job.update(identity={'architecture':'test'},settings={})
            mix=LatentMixin();mix._lc_source=None
            mix._lc_capture(original_video,original_audio,124,24,32,32)
            decoded=torch.rand(3,124,32,32)*2-1
            mix._lc_decoded(decoded)
            # Run the actual native save_video list branch. Only the imageio
            # writer/codec selection are replaced; record every encoded frame.
            written=[]
            class Writer:
                def append_data(self,frame): written.append(torch.from_numpy(frame.copy()))
                def close(self): destination.write_bytes(b'fake-encoded-video')
            export_tree=ast.parse((wangp/'shared/utils/audio_video.py').read_text())
            export_tree.body=[node for node in export_tree.body if isinstance(node,ast.FunctionDef)
                              and node.name in ('save_video','_video_frame_tensor_to_uint8')]
            export_globals={'torch':torch,'osp':os.path,
                            '_validate_video_save_settings':lambda *args:None,
                            '_get_codec_params':lambda *args:{},
                            'imageio':types.SimpleNamespace(get_writer=lambda *args,**kwargs:Writer())}
            exec(compile(export_tree,str(wangp/'shared/utils/audio_video.py'),'exec'),export_globals)
            def metadata(video_path,configs):
                with open(video_path,'ab') as stream: stream.write(b'metadata')
            output=decoded.half().add(1).mul(127.5).clamp(0,255).to(torch.uint8)
            chunks=[output[:,:119],output[:,119:]]
            make_save_wrapper(export_globals['save_video'])(chunks,str(destination),24,nrow=1)
            assert torch.equal(torch.stack(written),output.permute(1,2,3,0))
            make_record_wrapper(metadata)(str(destination),{})
            return True
        # Native worker pops the shared payload. A wrapper can also replace it:
        # execution must depend on frozen task options, never on live UI state.
        task.pop('plugin_data')
        make_generation_wrapper(render,lambda model:model)(task,'h3_latent_fl2va',plugin_data={})
        manifest,loaded=load_checkpoint(destination.with_suffix('.safetensors'))
        assert torch.equal(loaded['video'],original_video)
        assert torch.equal(loaded['audio'],original_audio)
        assert manifest['video_sha256']==sha256_file(destination)
        assert JOB.get() is None
    # New controls must take the same proven form -> queue route.
    await app.process_api(save_event,[None]*len(native_inputs)+[True,True,None,'audio_prefix',52,2.0],state=session)
    experimental=installed['get_model_settings'](st,'h3_latent_fl2va')
    experimental.update(state=st,model_type='h3_latent_fl2va')
    installed['add_video_task'](**experimental)
    from h3uitest.integration import TASK_KEY
    choice=st['gen']['queue'][-1]['params'][TASK_KEY]['options']
    assert (choice['join_mode'],choice['video_context_frames'],choice['audio_context_seconds'])==('audio_prefix',52,2.0)
asyncio.run(exercise_submission())
print('PASS: actual Gradio process_api -> native save_inputs -> settings -> native add_video_task; stale data, disable, queued snapshot.')
print('PASS: plugin_data lost before enqueue AND before execution -> frozen task -> capture -> safetensors, exact tensors and final video hash (CPU/fake encoder).')
