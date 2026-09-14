"""Local latent edits to installed H3 methods, preserving their wrapper chains."""
import ast
import functools
import inspect
import json
from pathlib import Path
import textwrap
import types
from .latent_runtime import JOB, LatentMixin
from .seams import shift_target_audio

SUPPORTED = frozenset(('minimax_h3_fl2va', 'minimax_h3_ref2va',
                       'minimax_h3_fl2va_pruned', 'minimax_h3_ref2va_pruned'))
_ready = False


def active():
    job = JOB.get()
    return bool(job and job.get('enabled'))


def reachable_functions(function):
    """Follow captured callables without executing wrappers or changing their cells."""
    pending, seen = [function], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, types.MethodType):
            pending.append(value.__func__)
        elif isinstance(value, functools.partial):
            pending.append(value.func)
        elif isinstance(value, types.FunctionType):
            yield value
            wrapped = getattr(value, '__wrapped__', None)
            if wrapped is not None:
                pending.append(wrapped)
            pending.extend(value.__defaults__ or ())
            pending.extend((value.__kwdefaults__ or {}).values())
            for cell in value.__closure__ or ():
                try:
                    pending.append(cell.cell_contents)
                except ValueError:
                    pass  # Empty closure cell.


def edited_source(raw, edits):
    source = inspect.getsource(raw)
    for old, new in edits:
        if not old.startswith('\n'):
            old, new = '\n' + old, '\n' + new
        if source.count(old) != 1:
            return None
        source = source.replace(old, new, 1)
    return source


def compile_edit(function, edits, additions=None):
    """Find the actual native body inside wrappers, including undecorated closures."""
    candidates = []
    for candidate in reachable_functions(function):
        # Wrappers stay intact: only a reachable native body is edited.
        if candidate.__closure__ or hasattr(candidate, '__wrapped__'):
            continue
        try:
            source = edited_source(candidate, edits)
        except (OSError, TypeError):
            continue
        if source is not None:
            candidates.append((candidate, source))
    if len(candidates) != 1:
        raise RuntimeError('H3 Latent: ' + ('no reachable native method contains the required latent insertion points'
                           if not candidates else 'multiple reachable methods contain the latent insertion points')
                           + '. Existing wrappers were left unchanged. Native rendering remains available '
                           'with latent options disabled; include the installed H3 plugin sources for adaptation.')
    raw, source = candidates[0]
    tree = ast.parse(textwrap.dedent(source))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    node.decorator_list = []
    env = dict(raw.__globals__)
    env.update(additions or {})
    exec(compile(tree, raw.__code__.co_filename, 'exec'), env)
    env[node.name]._h3_added_globals = additions or {}
    return raw, env[node.name]


def dispatch_install(raw, edited):
    # Mutate the referenced function in place: wrappers holding it keep running.
    original = types.FunctionType(raw.__code__, raw.__globals__, raw.__name__, raw.__defaults__)
    original.__kwdefaults__ = raw.__kwdefaults__
    raw.__globals__.update(edited._h3_added_globals)
    # Keep native global lookups live, including other plugins' later changes.
    edited = types.FunctionType(edited.__code__, raw.__globals__, raw.__name__, raw.__defaults__)
    edited.__kwdefaults__ = raw.__kwdefaults__
    signature = inspect.signature(raw)
    key = '_h3_latent_dispatch_' + raw.__name__
    def invoke(*args, **kwargs):
        return (edited if active() else original)(*args, **kwargs)
    raw.__globals__[key] = invoke
    env = {}
    exec('def dispatch(*args, **kwargs):\n    return ' + key + '(*args, **kwargs)', env)
    raw.__code__ = env['dispatch'].__code__
    raw.__signature__ = signature


def ensure_hooks():
    global _ready
    if _ready:
        return
    from models.minimax_h3.pipeline import MiniMaxH3Pipeline
    from models.minimax_h3.transformer import MiniMaxH3Model
    from models.minimax_h3.components import packing
    edits = json.loads(Path(__file__).with_name('native_edits.json').read_text())
    # Prepare all changes before installing any of them.
    generated = compile_edit(MiniMaxH3Pipeline.generate, [(e['old'],e['new']) for e in edits])
    packed = compile_edit(packing._fill_audio_condition_positions, [(
        '        elif anchor == "first":\n            origin = target_origin\n',
        '        elif anchor == "first":\n            origin = target_origin\n'
        '        elif isinstance(anchor, (int, float)) and not isinstance(anchor, bool):\n'
        '            origin = target_origin + float(anchor)\n')])
    layout = compile_edit(MiniMaxH3Model._layout, [(
        '                     payload["fps"],\n',
        '                     payload["fps"],\n                     payload.get("lc_audio_offset", 0.0),\n'), (
        '        payload["layout_signature"], payload["layout"] = signature, layout\n',
        '        _h3_shift_audio(layout, payload.get("lc_audio_offset", 0.0))\n'
        '        payload["layout_signature"], payload["layout"] = signature, layout\n')],
        {'_h3_shift_audio':shift_target_audio})
    for name, method in vars(LatentMixin).items():
        if name.startswith('_lc_'):
            setattr(MiniMaxH3Pipeline, name, method)
    for raw, edited in (generated, packed, layout):
        dispatch_install(raw, edited)
    _ready = True
    print('[H3 Latent] Native H3 latent hooks installed; existing wrappers preserved.')


def install_loader():
    """Attach checkpoint identity to native instances without replacing their class."""
    from models.minimax_h3.minimax_h3_handler import family_handler
    original = family_handler.load_model
    if getattr(original, '_h3_latent_loader', False):
        return
    signature = inspect.signature(original)
    @functools.wraps(original)
    def load(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        result = original(*args, **kwargs)
        base = bound.arguments.get('base_model_type')
        if base in SUPPORTED:
            from models.minimax_h3.minimax_h3_main import VIDEO_VAE_FILE, AUDIO_VAE_FILE
            pipe, _ = result
            filename = bound.arguments.get('model_filename', '')
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            definition = bound.arguments.get('model_def') or {}
            pipe._lc_identity = {'architecture':base, 'checkpoint_name':Path(str(filename)).name,
                'video_vae':str(definition.get('video_vae_file', VIDEO_VAE_FILE)),
                'audio_vae':str(definition.get('audio_vae_file', AUDIO_VAE_FILE)),
                'normalization':'WanGP-H3-LATENTS_MEAN_STD-v1'}
        return result
    load._h3_latent_loader = True
    # family_handler is a class in native Wan2GP; avoid adding implicit self.
    family_handler.load_model = staticmethod(load) if isinstance(family_handler, type) else load
