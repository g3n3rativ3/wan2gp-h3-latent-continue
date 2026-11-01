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


def edit_alternatives(edit):
    """An edit is (old, new), or a list of (old, new) variants for different WanGP versions."""
    return [edit] if isinstance(edit, tuple) else list(edit)


def apply_edits(source, edits):
    for edit in edits:
        for old, new in edit_alternatives(edit):
            if not old.startswith('\n'):
                old, new = '\n' + old, '\n' + new
            if source.count(old) == 1:
                source = source.replace(old, new, 1)
                break
        else:
            return None
    return source


def edited_source(raw, edits):
    return apply_edits(inspect.getsource(raw), edits)


def load_generate_edits():
    """native_edits.json entries: {"old", "new"}, or {"variants": [{"old", "new"}, ...]} (first unique match wins)."""
    entries = json.loads(Path(__file__).with_name('native_edits.json').read_text())
    return [[(v['old'], v['new']) for v in e['variants']] if 'variants' in e else (e['old'], e['new']) for e in entries]


def _function_node(source):
    tree = ast.parse(textwrap.dedent(source))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    node.decorator_list = []
    return tree, node


def _code_signature(code):
    """Bytecode identity of a function body, independent of names, files and line numbers."""
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
            tuple(_code_signature(c) if isinstance(c, types.CodeType) else (type(c).__name__, repr(c))
                  for c in code.co_consts))


def compiled_signature(source, namespace, filename='<h3-latent-check>'):
    tree, node = _function_node(source)
    env = dict(namespace)  # defaults are evaluated against the module globals
    exec(compile(tree, filename, 'exec'), env)
    return _code_signature(env[node.name].__code__)


def foreign_generate_rewrites(raw):
    """In-place body rewrites other plugins installed on native generate, as source transforms.

    H3 Audio Refinement Controls (wan2gp-h3-audio-refinement) recompiles generate from the
    WanGP source with an exact refinement schedule and, optionally, kept audio references.
    The latent version is also built from the source, so the same rewrites are re-applied
    to it; the installed body is then checked by bytecode in compile_edit.
    """
    hooks = raw.__globals__.get('_h3ar_hooks')
    owner = raw.__globals__.get('MiniMaxH3Pipeline')
    if hooks is None or owner is None:
        return []
    transforms = []
    if getattr(owner, '_h3ar_exact_schedule', False):
        def schedule(source):
            new, count = hooks._OLD_SCHEDULE.subn(lambda m: (
                f"{m['indent']}full_steps = refinement_steps\n"
                f"{m['indent']}refinement_base_sigmas = torch.linspace(float(H3_AUDIO_REFINEMENT_DENOISE), 0.0, refinement_steps + 1, {m['kw']})\n"), source)
            return new if count == 1 else None
        transforms.append(('H3 Audio Refinement exact schedule', schedule))
    if getattr(owner, '_h3ar_audio_refs', False):
        def references(source):
            old_prompt, old_loras = hooks._OLD_REFINEMENT_PROMPT, hooks._OLD_REFINEMENT_LORAS
            if source.count(old_prompt) != 1 or source.count(old_loras) != 1 or source.index(old_prompt) > source.index(old_loras):
                return None
            return source.replace(old_prompt, hooks._NEW_REFINEMENT_PROMPT).replace(old_loras, hooks._NEW_REFINEMENT_LORAS)
        transforms.append(('H3 Audio Refinement audio references', references))
    return transforms


def apply_rewrites(source, rewrites):
    for name, transform in rewrites:
        changed = transform(source)
        if changed is None:
            raise RuntimeError(f'H3 Latent: cannot combine the latent edits with "{name}". Existing wrappers were left unchanged; '
                               'disable the latent options or send both plugin versions for adaptation.')
        source = changed
    return source


def compile_edit(function, edits, additions=None, rewrites_for=None):
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
    rewrites = rewrites_for(raw) if rewrites_for else []
    if rewrites_for:
        # The installed body must be WanGP's source plus the known rewrites; otherwise another plugin
        # changed it in place and its change would be missing from the latent version.
        expected = apply_rewrites(inspect.getsource(raw), rewrites)
        if compiled_signature(expected, raw.__globals__) != _code_signature(raw.__code__):
            print('[H3 Latent] WARNING: another plugin rewrote MiniMaxH3Pipeline.generate in place (not as a wrapper). '
                  'With latent options on, that in-place change is not applied; wrappers are unaffected. '
                  'Known and combined: H3 Audio Refinement Controls.')
        elif rewrites:
            print('[H3 Latent] Combined with ' + ', '.join(name for name, _ in rewrites) + ' (installed body verified).')
    source = apply_rewrites(source, rewrites)
    tree, node = _function_node(source)
    # Report native file lines in tracebacks (lines after an inserted edit shift by its added lines).
    ast.increment_lineno(tree, raw.__code__.co_firstlineno - 1)
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
    edits = load_generate_edits()
    # Prepare all changes before installing any of them.
    generated = compile_edit(MiniMaxH3Pipeline.generate, edits,
                             rewrites_for=foreign_generate_rewrites)
    packed = compile_edit(packing._fill_audio_condition_positions, [(
        '        elif anchor == "first":\n            origin = target_origin\n',
        '        elif anchor == "first":\n            origin = target_origin\n'
        '        elif isinstance(anchor, (int, float)) and not isinstance(anchor, bool):\n'
        '            origin = target_origin + float(anchor)\n')])
    layout = compile_edit(MiniMaxH3Model._layout, [(
        '                     payload["fps"], payload.get("audio_time_offset", 0.0),\n',
        '                     payload["fps"], payload.get("audio_time_offset", 0.0),\n'
        '                     payload.get("lc_audio_offset", 0.0),\n'), (
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
