"""Versioned, non-pickle H3 checkpoint format and pure timing helpers."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

FORMAT = 'wan2gp.h3.latent-continuation'
VERSION = 2
MAX_BYTES = 2 * 1024**3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def token_start(index):
    return (index // 5) * 17 + (0, 1, 5, 9, 13)[index % 5]


def tail_indices(length, delivered_frames, overlap):
    """Include blocks intersecting the requested tail; coordinates retain source phase."""
    boundary = delivered_frames - 1
    left = max(0, delivered_frames - overlap)
    result = []
    for i in range(length):
        start, end = token_start(i), token_start(i + 1)
        if start < delivered_frames and end > left:
            result.append((i, start - boundary))
    if not result:
        raise ValueError('No video latent covers the requested continuation boundary.')
    return result


def inspect_checkpoint(path):
    from safetensors import safe_open
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('Missing checkpoint or file exceeds the prototype 2 GiB limit.')
    with safe_open(str(path), framework='pt', device='cpu') as f:
        md = f.metadata() or {}
        if md.get('format') != FORMAT or md.get('version') not in ('1', str(VERSION)):
            raise ValueError('Not an H3 Latent Continue checkpoint (a LoRA is not a checkpoint).')
        info = json.loads(md['manifest'])
        origin = info.get('audio_origin_seconds', 0.0)
        if not isinstance(origin, (int,float)) or not math.isfinite(origin) or not -3 <= origin <= 0:
            raise ValueError('Invalid checkpoint audio timeline origin.')
        if set(f.keys()) != {'video', 'audio', 'last_frame'}:
            raise ValueError('Unexpected checkpoint tensor keys.')
        shapes = {k: f.get_slice(k).get_shape() for k in f.keys()}
    v, a, last = shapes['video'], shapes['audio'], shapes['last_frame']
    if len(v) != 5 or v[0] != 1 or v[1] != 24 or v[2] < 2 or v[2] > 512 or min(v[3:]) < 2:
        raise ValueError('Invalid H3 video latent shape.')
    if len(a) != 4 or a[:3] != [1, 32, 2] or not 1 <= a[3] <= 2000:
        raise ValueError('Invalid H3 audio latent shape.')
    if last != [3, 1, v[3]*16, v[4]*16]:
        raise ValueError('Invalid last-frame dimensions.')
    frames = info.get('target_frames', 0)
    fps = info.get('fps', 0)
    if not isinstance(frames, int) or not 1 <= frames <= token_start(v[2]):
        raise ValueError('Invalid delivered-frame mapping.')
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError('Invalid checkpoint FPS.')
    if info.get('width') != v[4]*16 or info.get('height') != v[3]*16:
        raise ValueError('Manifest dimensions differ from latent dimensions.')
    return info


def load_checkpoint(path):
    from safetensors.torch import load_file
    import torch
    info = inspect_checkpoint(path)
    tensors = load_file(str(path), device='cpu')
    for key, tensor in tensors.items():
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'Non-finite or non-floating tensor: {key}')
    return info, tensors


def atomic_save(path, tensors, info):
    from safetensors.torch import save_file
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Latent checkpoint already exists: {path}. Choose a new output filename.')
    fd, tmp = tempfile.mkstemp(prefix='.h3-latent-', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        save_file({k: v.detach().cpu().contiguous() for k,v in tensors.items()}, tmp,
                  metadata={'format': FORMAT, 'version': str(VERSION), 'manifest': json.dumps(info, ensure_ascii=False)})
        # Exclusive destination reservation avoids overwriting a checkpoint from another job.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        try:
            os.replace(tmp, path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    finally:
        Path(tmp).unlink(missing_ok=True)
