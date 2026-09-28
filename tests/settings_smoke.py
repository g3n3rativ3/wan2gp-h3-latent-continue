"""WanGP 13.14 real name validator/enqueue/import + plugin transport regression.
Usage: python tests/settings_smoke.py /path/to/Wan2GP
No model loading or GPU required.
"""
import ast
import copy
import importlib.util
import inspect
from pathlib import Path
import sys
import types

root=Path(__file__).resolve().parents[1]
upstream=Path(sys.argv[1]).resolve()
def load(name,path,package=False):
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)] if package else None)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
load('h3transport',root/'__init__.py',True)
from h3transport.integration import (TASK_KEY,KEY,make_queue_wrapper,make_settings_wrapper,
    make_generation_wrapper,make_validation_wrapper,make_runtime_task_wrapper,make_edit_wrapper,preflight)
validator=load('native_names',upstream/'shared/utils/setting_names.py')
model='minimax_h3_fl2va_pruned_fused_turbo'
base=lambda m:'minimax_h3_fl2va_pruned' if m==model else m
get_model=lambda s:s['model_type']
seen=[]
class ReachedModelValidation(Exception):pass
def model_def(m):raise ReachedModelValidation()
ns=dict(task_id=0,get_gen_info=lambda st:st.setdefault('gen',{'queue':[]}),
        get_preview_images=lambda p:(None,None,[],[]),
        unknown_settings_error=validator.unknown_settings_error,
        gr=types.SimpleNamespace(Info=seen.append),get_model_def=model_def)
tree=ast.parse((upstream/'wgp.py').read_text())
tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
           ('add_video_task','_build_runtime_task','validate_settings')]
exec(compile(tree,str(upstream/'wgp.py'),'exec'),ns)
legacy={TASK_KEY:{'model_type':model,'options':{'save':False,'continue':False,'latent_path':''}}}
assert 'Unknown setting: '+TASK_KEY in ns['validate_settings']({},model,True,legacy)[-1]
validate=make_validation_wrapper(ns['validate_settings'])
try:validate({},model,True,legacy)
except ReachedModelValidation:pass
else:raise AssertionError('Did not reach native model validation')
assert TASK_KEY not in legacy
assert not validator.unknown_settings_error(legacy)
assert validator.unknown_settings_error({'unrelated_typo':1}) # validator not relaxed

for save in (False,True):
    st={'model_type':model,'gen':{'queue':[]}}
    opts={'save':save,'continue':False,'latent_path':''}
    settings={'plugin_data':{KEY:opts,TASK_KEY:{'model_type':model,'options':copy.deepcopy(opts)},'foreign':{'keep':7}}}
    assert not validator.unknown_settings_error(settings)
    enqueue=make_queue_wrapper(ns['add_video_task'],base,get_model)
    enqueue(state=st,model_type=model,**settings)
    task=st['gen']['queue'][0]
    assert TASK_KEY not in task['params'] and 'plugin_data' not in task['params']
    assert task[TASK_KEY]['options']['save'] is save
    opts['save']=not save
    assert task[TASK_KEY]['options']['save'] is save
    assert task['plugin_data']['foreign']=={'keep':7}
    # Actual worker call convention: explicit plugin_data + filtered params.
    def render(task,model_type,state,plugin_data=None):return 'native'
    if not save:
        wrapped=make_generation_wrapper(render,base)
        params={k:v for k,v in task['params'].items() if k in inspect.signature(wrapped).parameters}
        assert wrapped(task,plugin_data=task.pop('plugin_data'),**params)=='native'

# Imported legacy params are migrated to the task envelope before dispatch.
build=make_runtime_task_wrapper(ns['_build_runtime_task'])
task=build(9,dict(model_type=model,**{TASK_KEY:{'model_type':model,'options':{'save':True,'continue':False}}}),{'foreign':7})
assert TASK_KEY not in task['params'] and 'plugin_data' not in task['params']
assert task[TASK_KEY]['options']['save'] is True and task['plugin_data']['foreign']==7
# Editing must update the independent snapshot and never duplicate plugin_data kwargs.
st={'editing_task_id':9,'gen':{'queue':[task]},'edit_state':{'model_type':model,'plugin_data':{TASK_KEY:{'model_type':model,'options':{'save':False,'continue':False}}}}}
def edit(st):task['params']=st.pop('edit_state');return 'edited'
assert make_edit_wrapper(edit)(st)=='edited'
assert not task[TASK_KEY]['options']['save'] and 'plugin_data' not in task['params']
for flags in ('AS','KS'):
    preflight({'audio_prompt_type':flags},{'save':False,'continue':False})
    try:preflight({'audio_prompt_type':flags},{'save':True,'continue':False})
    except ValueError as e:assert 'soundtrack' in str(e)
    else:raise AssertionError('Kept soundtrack accepted for latent capture')
print('PASS: original Unknown setting reproduced; migration, disabled fused variant, native queue/import, immutable snapshot, edits, native worker calling convention, soundtrack guards; unrelated unknown keys still rejected.')
