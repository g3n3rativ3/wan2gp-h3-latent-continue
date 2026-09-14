"""Experimental context selection and sample-accurate audio pre-roll geometry."""
import math

MODES = ('baseline', 'context', 'audio_prefix')


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
