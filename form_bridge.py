"""Read the inline controls in the SAME event that saves the native form.

Wan2GP registers save_inputs events before inserting plugin components. Extend
only those existing events belonging to this form, without copying its fields.
"""
import functools
import inspect
from .integration import KEY, TASK_KEY, state_mapping, remember_options


def bind_form_events(context, data_component, controls, get_model):
    count = 0
    for event in list(context.fns.values()):
        original = event.fn
        if original is None or getattr(original, '__name__', '') != 'save_inputs':
            continue
        if data_component not in event.inputs or getattr(original, '_h3_form_bound', False):
            continue
        if event.inputs_as_dict:
            raise RuntimeError('H3 Latent: unsupported dictionary-style native form event.')
        signature = inspect.signature(original)
        if not {'state', 'plugin_data'}.issubset(signature.parameters):
            raise RuntimeError('H3 Latent: unsupported save_inputs signature.')
        length = len(event.inputs)

        def build(fn, sig, n):
            @functools.wraps(fn)
            def save_form(*values):
                bound = sig.bind(*values[:n])
                saved, enabled, path, experiment, video_frames, audio_seconds = values[n:]
                payload = state_mapping(bound.arguments['plugin_data'])
                payload[KEY] = {'save': bool(saved), 'continue': bool(enabled), 'latent_path': path or '',
                                'join_mode':experiment,'video_context_frames':video_frames,'audio_context_seconds':audio_seconds}
                captured = dict(payload[KEY])
                state = bound.arguments['state']
                model = get_model(state_mapping(state))
                remember_options(state, model, payload)
                bound.arguments['plugin_data'] = payload
                print(f'[H3 Latent] Form captured: save={bool(saved)}, continue={bool(enabled)}, model={model}')
                result = fn(*bound.args, **bound.kwargs)
                # Native task editing replaces params without add_video_task.
                target = bound.arguments.get('target')
                current = state_mapping(state)
                settings = current.get('edit_state') if target == 'edit_state' else current.get('all_settings', {}).get(model)
                if target in ('state','edit_state') and isinstance(settings,dict):
                    settings[TASK_KEY] = {'model_type':model,'options':captured}
                return result
            save_form._h3_form_bound = True
            return save_form

        event.fn = build(original, signature, length)
        event.inputs = list(event.inputs) + list(controls)
        count += 1
    if count == 0:
        raise RuntimeError('H3 Latent: no native save_inputs event found for this form; controls cannot be connected.')
    return count
