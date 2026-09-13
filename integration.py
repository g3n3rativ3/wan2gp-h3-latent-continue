"""Narrow bridge to WanGP's job and output lifecycle (no core file edits)."""
from collections.abc import Mapping
import functools
import inspect
from pathlib import Path
import torch
from .checkpoint import atomic_save, inspect_checkpoint, sha256_file
from .latent_runtime import JOB, frame_hash
from .export_frames import export_tail
from .seams import seam_settings

KEY = 'h3_latent_prototype'
UI_KEY = '_h3_latent_ui_options'
TASK_KEY = '_h3_latent_task_options_v1'


def make_queue_wrapper(original, get_base, get_state):
    """Freeze the captured selection in task params before native enqueue.

    plugin_data is mutable shared plumbing and is popped by queue_worker_func.
    This independent, JSON-safe record belongs to the queued task, not the UI.
    """
    import copy
    @functools.wraps(original)
    def enqueue(**inputs):
        model = inputs.get('model_type')
        if not model or not str(get_base(model)).startswith('h3_latent_'):
            return original(**inputs)
        state = state_mapping(inputs.get('state'))
        ui_model = get_state(state) if state else model
        stored = inputs.get(TASK_KEY)
        captured = stored.get('options') if isinstance(stored,dict) and stored.get('model_type') == model else None
        if captured is None:
            captured = state.get(UI_KEY, {}).get(ui_model)
        if captured is None:
            captured = state.get(UI_KEY, {}).get(model)
        if captured is None:
            payload = state_mapping(inputs.get('plugin_data'))
            if KEY not in payload:
                raise RuntimeError('H3 Latent: no captured options at enqueue. Refresh the page and create a new task; rendering was not started.')
            captured = options(payload)
        frozen = copy.deepcopy(captured)
        inputs[TASK_KEY] = {'model_type':model,'options':frozen}
        payload = copy.deepcopy(state_mapping(inputs.get('plugin_data')))
        payload[KEY] = copy.deepcopy(frozen)
        inputs['plugin_data'] = payload
        print(f'[H3 Latent] Queued model={model}: save={frozen["save"]}, continue={frozen["continue"]}')
        return original(**inputs)
    return enqueue


def bound_parameters(signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    result = dict(bound.arguments)
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            result.update(result.pop(name, {}))
    return result


def remember_options(state, model, data):
    # Session-local and model-specific; never share UI options between users.
    for _ in range(8):
        if type(state).__name__ == 'State' and type(state).__module__.startswith('gradio.'):
            state = state.value
        else:
            break
    if isinstance(state, dict):
        state.setdefault(UI_KEY, {})[model] = options(data)


def merge_ui_options(state, model, data):
    import copy
    result = copy.deepcopy(state_mapping(data))
    snapshot = state_mapping(state).get(UI_KEY, {}).get(model)
    if snapshot is not None:
        result[KEY] = copy.deepcopy(snapshot)
    return result


def make_settings_wrapper(original, get_base):
    @functools.wraps(original)
    def settings(state, model_type):
        result = original(state, model_type)
        if result is not None and str(get_base(model_type)).startswith('h3_latent_'):
            result = dict(result)
            result['plugin_data'] = merge_ui_options(state, model_type, result.get('plugin_data'))
        return result
    return settings


def state_mapping(data):
    """Unwrap Gradio State refresh values; keep ordinary plugin mappings intact.

    A form refresh can pass gr.State(...) as a stateful value. Only unwrap
    actual Gradio State classes, with a depth bound for malformed nesting.
    Unexpected/empty payloads are treated as disabled default options.
    """
    for _ in range(8):
        if isinstance(data, Mapping):
            return dict(data)
        cls = type(data)
        if cls.__name__ != 'State' or not cls.__module__.startswith('gradio.'):
            return {}
        data = data.value
    return {}


def options(data):
    raw = state_mapping(state_mapping(data).get(KEY))
    result = {'save': raw.get('save') is True, 'continue': raw.get('continue') is True,
              'latent_path': str(raw.get('latent_path') or '')}
    result.update({k:raw[k] for k in ('join_mode','video_context_frames','audio_context_seconds') if k in raw})
    return result


def preflight(params, opts):
    if not (opts['save'] or opts['continue']): return
    join_mode, _, _ = seam_settings(opts)
    if join_mode != 'baseline' and not opts['continue']:
        raise ValueError('H3 Latent: experimental join modes require Continue with latent; use baseline for the initial clip.')
    if join_mode == 'audio_prefix' and params.get('skip_steps_cache_type') not in (None, '', 'none', 'disabled'):
        raise ValueError('H3 Latent: audio-prefix mode requires No Skipping (Spectrum and other step caches unsupported).')
    if int(params.get('guidance_phases', 1)) != 1:
        raise ValueError('H3 Latent Continue supports SINGLE-PHASE rendering only. Two phases are not supported.')
    if int(params.get('batch_size', 1)) != 1:
        raise ValueError('H3 Latent Continue: batch size must be 1.')
    for key in ('spatial_upsampling','temporal_upsampling','postprocess_audio','replace_voice_method',
                'self_refiner_setting','keep_frames_video_source'):
        if params.get(key) not in (None, '', 0, False):
            raise ValueError(f'H3 Latent Continue: disable {key}; it can invalidate latent/video correspondence.')
    for key in ('film_grain_intensity','sliding_window_discard_last_frames','sliding_window_trim_first_frames',
                'sliding_window_color_correction_strength'):
        if float(params.get(key) or 0) != 0:
            raise ValueError(f'H3 Latent Continue: {key} must be zero.')
    if params.get('video_mask') is not None or 'G' in str(params.get('video_prompt_type') or ''):
        raise ValueError('H3 Latent Continue: masked/control-video editing is not supported.')
    if (params.get('frame_scheduler') or {}).get('active'):
        raise ValueError('H3 Latent Continue: custom frame scheduling is unsupported.')
    if '&' in str(params.get('video_prompt_type') or '') or '~' in str(params.get('video_prompt_type') or ''):
        raise ValueError('H3 Latent Continue: HDR and phase-two tiling are unsupported.')
    if (params.get('custom_settings') or {}).get('audio_refinement', 'none') != 'none':
        raise ValueError('H3 Latent Continue: disable extra audio refinement.')
    if opts['continue']:
        if 'V' not in str(params.get('image_prompt_type') or '') or not params.get('video_source'):
            raise ValueError('Select Continue Video and upload its original source video.')
        if not opts['latent_path']:
            raise ValueError('Continue with latent is checked, but no .safetensors file was supplied.')
        info = inspect_checkpoint(opts['latent_path'])
        source = params['video_source']
        if not isinstance(source, (str, Path)) or not Path(source).is_file():
            raise ValueError('Latent continuation requires a local source video file.')
        if sha256_file(source) != info.get('video_sha256'):
            raise ValueError('The video does not match this latent checkpoint. Use the original saved video, without editing or transcoding.')


def make_generation_wrapper(original, get_base):
    sig = inspect.signature(original)
    @functools.wraps(original)
    def generate(*args, **kwargs):
        params = bound_parameters(sig, args, kwargs)
        model = params.get('model_type') or params.get('task', {}).get('params', {}).get('model_type')
        prototype = str(get_base(model)).startswith('h3_latent_') if model else False
        if not prototype:
            return original(*args, **kwargs)
        task_params = state_mapping(state_mapping(params.get('task')).get('params'))
        snapshot = task_params.get(TASK_KEY)
        if snapshot is not None:
            if not isinstance(snapshot, dict) or snapshot.get('model_type') != model:
                raise ValueError('H3 Latent: task options/model mismatch; recreate this task.')
            raw = snapshot.get('options')
            if not isinstance(raw,dict) or any(type(raw.get(k)) is not bool for k in ('save','continue')):
                raise ValueError('H3 Latent: invalid captured task options; recreate this task.')
            opts = options({KEY:raw})
        else:
            payload = state_mapping(params.get('plugin_data'))
            if KEY not in payload:
                raise RuntimeError('H3 Latent: task has lost its options. Rendering stopped before loading weights. Refresh and create a new task; include Form captured and Queued lines in the report.')
            opts = options(payload)
        print('[H3 Latent] Job options: save=' + str(opts['save'])
              + ', continue=' + str(opts['continue']))
        # Two phases are never supported by these model entries, including baseline mode.
        if int(params.get('guidance_phases',1)) != 1:
            raise ValueError('H3 Latent Continue supports SINGLE-PHASE rendering only.')
        preflight(params, opts)
        job = {'enabled':opts['save'] or opts['continue'],'options':opts,'pending':None,
               'output_seen':False,'saved':[], 'source': str(params.get('video_source') or '')}
        token = JOB.set(job)
        try:
            result = original(*args, **kwargs)
            state = state_mapping(params.get('state'))
            generation = state_mapping(state.get('gen'))
            aborted = generation.get('abort') is True
            if aborted and opts['save'] and not job['saved']:
                print('[H3 Latent] Generation aborted by user; pending latent capture discarded.')
            if opts['save'] and result and not job['saved'] and not aborted:
                raise RuntimeError('Video generation returned without a saved latent checkpoint. See console; no latent success is claimed.')
            return result
        finally:
            job['pending'] = None
            JOB.reset(token)
    return generate


def make_save_wrapper(original):
    sig = inspect.signature(original)
    @functools.wraps(original)
    def save(*args, **kwargs):
        job = JOB.get()
        if job and job.get('pending'):
            arguments = bound_parameters(sig, args, kwargs)
            tensor, total_frames = export_tail(arguments['tensor'])
            pending = job['pending']; info = pending['info']
            last = tensor[:,-1:]
            signature = tuple(frame_hash(tensor[:,i:i+1]) for i in range(max(0,tensor.shape[1]-8),tensor.shape[1]))
            matches = pending['tail_signatures'].get(signature, [])
            if len(matches) > 1:
                raise ValueError('Ambiguous output tail (repeated identical frames); refusing to guess the latent endpoint.')
            frame_end = matches[0] if matches else None
            if frame_end is None:
                raise ValueError('Output tail no longer matches the decoded latent: postprocessing or trimming is unsupported. No mismatched sidecar will be written.')
            if tuple(tensor.shape[-2:]) != (info['height'],info['width']) or abs(float(arguments['fps'])-info['fps'])>1e-6:
                raise ValueError('Output resolution or FPS changed after sampling.')
            info['target_frames'] = frame_end
            info['output_frames'] = total_frames
            # Text-encoder presentation cache, not a re-encoded video conditioning latent.
            pending['tensors']['last_frame'] = last.detach().cpu().float()
            if tensor.dtype == torch.uint8:
                pending['tensors']['last_frame'].div_(127.5).sub_(1.0)
            job['output_seen'] = True
        return original(*args, **kwargs)
    return save


def make_record_wrapper(original):
    sig = inspect.signature(original)
    @functools.wraps(original)
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        job = JOB.get()
        if not job or not job.get('pending'): return result
        arguments = bound_parameters(sig, args, kwargs)
        if not job.get('output_seen'):
            raise RuntimeError('Latent capture was not matched to WanGP video export.')
        path = Path(arguments['video_path'])
        pending = job['pending']; info = pending['info']
        info['video_filename'] = path.name
        # Native metadata may rewrite the MP4: bind AFTER record_file_metadata has finished.
        info['video_sha256'] = sha256_file(path)
        configs = arguments.get('configs', {})
        info['loras'] = {k:configs.get(k) for k in ('transformer_loras_filenames','transformer_loras_multipliers')}
        info['source_video'] = Path(job['source']).name if job['source'] else None
        destination = path.with_suffix('.safetensors')
        atomic_save(destination, pending['tensors'], info)
        print(f'[H3 Latent] Saved {destination} (final segment, {info["target_frames"]} frames, original tensor dtypes).')
        job['saved'].append(str(destination))
        job['pending'] = None; job['output_seen'] = False
        return result
    return record


def make_prepare_wrapper(original, get_base, get_state):
    """WanGP's normal form cleaner removes plugin_data before storing task settings.
    Retain a detached copy for prototype jobs so UI selections actually reach the queue.
    """
    import copy
    @functools.wraps(original)
    def prepare(target, inputs, model_type=None, model_filename=None):
        model = model_type or get_state(inputs['state'])
        prototype = str(get_base(model)).startswith('h3_latent_')
        data = merge_ui_options(inputs.get('state'), model, inputs.get('plugin_data'))
        if prototype and int(inputs.get('guidance_phases',1)) != 1:
            raise ValueError('H3 Latent Continue supports SINGLE-PHASE rendering only; change the source settings explicitly.')
        result = original(target, inputs, model_type, model_filename)
        if prototype and target in ('state','edit_state','settings'):
            result['plugin_data'] = data
        return result
    return prepare
