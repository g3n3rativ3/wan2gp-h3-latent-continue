"""Experimental context selection and sample-accurate audio pre-roll geometry."""
import math

MODES = ('baseline', 'context', 'audio_prefix', 'pinned')


def reference_context_frames(delivered_frames, native_overlap):
    """Keep one H3 history interval even when assembly overlaps a single frame."""
    return min(int(delivered_frames), max(18, int(native_overlap)))


def seam_settings(options):
    mode = options.get('join_mode', 'baseline')
    if mode not in MODES:
        raise ValueError('H3 Latent: unknown join mode.')
    frames = options.get('video_context_frames', 35)
    seconds = options.get('audio_context_seconds', 1.0)
    if mode != 'baseline':
        if frames not in (18, 35, 52):
            raise ValueError('H3 Latent: video context must be 18, 35 or 52 frames.')
        if seconds not in (0.5, 1.0, 2.0):
            raise ValueError('H3 Latent: audio context must be 0.5, 1 or 2 seconds.')
    return mode, int(frames), float(seconds)


def audio_window(info, length, seconds):
    # Target frame zero is the last delivered source frame, NOT the end of it.
    boundary = (info['target_frames'] - 1) / info['fps']
    delivered_end = info['target_frames'] / info['fps']
    source_origin = float(info.get('audio_origin_seconds', 0.0))
    start = max(0, math.floor((delivered_end - seconds - source_origin) * 40 + 1e-8))
    stop = min(length, math.ceil((delivered_end - source_origin) * 40 - 1e-8))
    if stop <= start or source_origin + stop / 40 < boundary - 1e-6:
        raise ValueError('H3 Latent: checkpoint audio does not cover the continuation boundary.')
    origin = source_origin + start / 40 - boundary
    return start, stop, origin


def shift_target_audio(layout, offset):
    if offset:
        rows = layout.audio_indices[layout.num_condition_audio_rows:]
        layout.position_ids[rows, 0] += offset
    return layout


# H3 VAE: each 17-frame chunk is encoded on its own into latents covering [1, 4, 4, 4, 4] frames.
_CHUNK_TOKEN_STARTS = {0: 0, 1: 1, 5: 2, 9: 3, 13: 4}


def latents_for_frames(frames):
    """Latents covering the first `frames` frames of a chunk-aligned sequence (frames on a latent boundary)."""
    return 5 * (frames // 17) + _CHUNK_TOKEN_STARTS[frames % 17]


def pinned_geometry(delivered, latent_count, overlap):
    """Exact join: the new window starts on a VAE chunk boundary of the saved segment, so its first latents
    ARE the saved latents of the last `frames` delivered frames (pinned), and the new frames follow them in the
    same latent sequence. `frames` <= overlap: WanGP drops the overlap, i.e. the pinned frames and what precedes."""
    delivered, latent_count, overlap = int(delivered), int(latent_count), int(overlap)
    remainder = delivered % 17
    if remainder not in _CHUNK_TOKEN_STARTS:
        raise ValueError(f'H3 Latent exact join: the saved segment has {delivered} frames, which does not end on a VAE latent '
                         'boundary (WanGP trimmed its tail). Render the source with a 5+17n frame length (e.g. 124, 243).')
    candidates = [frames for frames in range(remainder or 17, min(overlap, delivered) + 1, 17)]
    if not candidates:
        raise ValueError(f'H3 Latent exact join: the overlap ({overlap} frames) is too short; it needs at least '
                         f'{remainder or 17} frames for this source length.')
    frames = max(candidates)
    first = latents_for_frames(delivered - frames)
    count = latents_for_frames(frames)
    if first + count != latents_for_frames(delivered) or first + count > latent_count:
        raise ValueError('H3 Latent exact join: saved latents do not cover the delivered frames.')
    return {'frames': frames, 'latents': count, 'first_latent': first, 'clip_start': delivered - frames}
