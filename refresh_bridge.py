"""Notify UI restoration without subscribing to shared Gradio State changes."""
import functools
import inspect
import uuid


def bind_refresh_events(context, data_component, signal, exclude=()):
    """Append a scalar notification to every existing writer of this form's data.

    Never hash, serialize, traverse or sanitize another plugin's payload. Preserve
    sync/async/generator callbacks, component-keyed results and skip updates.
    """
    count = 0
    for event in list(context.fns.values()):
        if event.fn is None or event.fn in exclude or data_component not in event.outputs:
            continue
        if getattr(event.fn, '_h3_refresh_bound', False):
            continue
        output_count = len(event.outputs)

        def build(fn, n):
            def append(value):
                token = uuid.uuid4().hex
                if isinstance(value, dict) and value and all(hasattr(k, '_id') for k in value):
                    return {**value, signal: token}
                if n == 1:
                    return [value, token]
                if isinstance(value, (list, tuple)):
                    return [*value, token]
                # Gradio permits a single skip update for all outputs.
                if isinstance(value, dict) and value.get('__type__') == 'update' and len(value) == 1:
                    return [*[value] * n, token]
                raise RuntimeError('H3 Latent: unexpected form refresh result shape.')

            if inspect.isasyncgenfunction(fn):
                @functools.wraps(fn)
                async def wrapped(*args, **kwargs):
                    async for value in fn(*args, **kwargs):
                        yield append(value)
            elif inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def wrapped(*args, **kwargs):
                    return append(await fn(*args, **kwargs))
            elif inspect.isgeneratorfunction(fn):
                @functools.wraps(fn)
                def wrapped(*args, **kwargs):
                    for value in fn(*args, **kwargs):
                        yield append(value)
            else:
                @functools.wraps(fn)
                def wrapped(*args, **kwargs):
                    return append(fn(*args, **kwargs))
            wrapped._h3_refresh_bound = True
            return wrapped

        event.fn = build(event.fn, output_count)
        event.outputs = list(event.outputs) + [signal]
        count += 1
    return count
