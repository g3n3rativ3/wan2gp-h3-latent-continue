"""Inspect native RGB export chunks without joining the entire video."""
import torch


def export_tail(value):
    """Return at most 8 final frames (C,T,H,W) and the total frame count.

    Wan2GP's SDR render converts frames to uint8, then passes a list of chunks
    to save_video. A list is a temporal sequence, not a batch of videos.
    """
    if torch.is_tensor(value):
        if value.ndim == 5:
            if value.shape[0] != 1:
                raise ValueError('Latent export requires batch size 1.')
            value = value[0]
        if value.ndim != 4 or value.shape[0] != 3 or value.shape[1] == 0:
            raise ValueError('Unexpected or empty video tensor layout at export.')
        return value[:, -8:], value.shape[1]
    if not isinstance(value, (list, tuple)) or not value or not torch.is_tensor(value[0]):
        raise ValueError('Latent export requires a tensor or a nonempty list of RGB uint8 tensor chunks.')
    chunks = []
    total = 0
    geometry = None
    for chunk in value:
        if chunk is None:
            continue  # Same rule as native save_video.
        if not torch.is_tensor(chunk) or chunk.ndim != 4 or chunk.dtype != torch.uint8:
            raise ValueError('Latent export: list chunks must be 4D uint8 tensors, as produced by Wan2GP SDR export.')
        # Native save_video gives channel-last precedence for list chunks.
        if chunk.shape[-1] in (1, 3, 4):
            chunk = chunk.permute(3, 0, 1, 2)
        if chunk.shape[0] != 3:
            raise ValueError('Latent export supports RGB chunks only.')
        size = tuple(chunk.shape[-2:])
        if min(size) <= 0 or (geometry is not None and geometry != size):
            raise ValueError('Latent export: chunk resolution changes or is empty.')
        geometry = size
        if chunk.shape[1]:
            chunks.append(chunk)
            total += chunk.shape[1]
    if total == 0:
        raise ValueError('Latent export contains no video frames.')
    tail = []
    remaining = 8
    for chunk in reversed(chunks):
        take = min(remaining, chunk.shape[1])
        tail.append(chunk[:, -take:].detach().cpu())
        remaining -= take
        if remaining == 0:
            break
    return torch.cat(list(reversed(tail)), dim=1), total
