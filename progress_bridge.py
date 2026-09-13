"""Optional Wan2GP 13 progress UI; no source/version compatibility gate."""
from contextlib import nullcontext
import importlib

try:
    _native = importlib.import_module('shared.utils.phase_progress')
except ModuleNotFoundError as error:
    if error.name != 'shared.utils.phase_progress':
        raise
    _native = None

generation_progress = getattr(_native, 'generation_progress', lambda method: method)
control_video_encoding = getattr(_native, 'control_video_encoding', lambda enabled=True: nullcontext())
