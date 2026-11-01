"""Continue Video "second part" export with the real native save_video/mux helpers and ffmpeg.
Usage: python tests/part2_smoke.py /path/to/Wan2GP
Simulates the WGP window assembly (source + window minus overlap, trimmed tail); no model.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
upstream = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(upstream))
from shared.utils import audio_video as av  # actual native encoder and muxer
import soundfile as sf
import cv2

spec = importlib.util.spec_from_file_location('h3part2', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec); sys.modules['h3part2'] = pkg; spec.loader.exec_module(pkg)
from h3part2.latent_runtime import LatentMixin, JOB
from h3part2.integration import make_save_wrapper, make_record_wrapper, frames_from

FPS, H, W = 24, 64, 64
SOURCE, OVERLAP, HISTORY, TARGET, TRIM = 40, 18, 17, 107, 7
NEW = HISTORY + TARGET - OVERLAP - TRIM  # frames created by the render that WGP keeps


def frame(color, index):
    value = torch.zeros(3, 1, H, W, dtype=torch.uint8)
    value[:, 0] = torch.tensor(color, dtype=torch.uint8).view(3, 1, 1)
    value[:, 0, :4, :4] = index % 251  # unique frames for the exact tail fingerprints
    return value


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                          'stream=codec_type,nb_read_frames,duration', '-of', 'json', str(path)],
                         capture_output=True, text=True, check=True)
    return {s['codec_type']: s for s in json.loads(out.stdout)['streams']}


def first_frame_color(path):
    capture = cv2.VideoCapture(str(path)); ok, image = capture.read(); capture.release()
    assert ok
    return image[H//2, W//2][::-1]  # BGR -> RGB


def render(td, with_audio=True, tamper=False, continue_video=True, audio_tail=0.0, seam=None):
    source = torch.cat([frame((200, 0, 0), i) for i in range(SOURCE)], dim=1)
    window = torch.cat([frame((200, 0, 0), 1000 + i) if i < OVERLAP else frame((0, 200, 0), 2000 + i)
                        for i in range(HISTORY + TARGET)], dim=1)
    decoded = window[:, HISTORY:]  # decoded target frames; the first HISTORY window frames are copies
    output = torch.cat([source, window[:, OVERLAP:HISTORY + TARGET - TRIM]], dim=1)  # WGP assembly
    if tamper:
        output = output.clone(); output[:, -1] ^= 1  # postprocessing changed the tail
    assert output.shape[1] == SOURCE + NEW
    job = {'enabled': True, 'options': {'save': False, 'continue': True}, 'pending': None, 'saved': [],
           'source': '', 'phase1': None, 'part2': None, 'wgp_overlap': OVERLAP if continue_video else 0}
    token = JOB.set(job)
    try:
        dummy = types.SimpleNamespace()
        LatentMixin._lc_capture(dummy, torch.zeros(1), torch.zeros(1), TARGET, FPS, W, H, HISTORY)
        LatentMixin._lc_decoded(dummy, decoded)
        if seam is not None:  # Exact join seamless end: switch frame + re-decoded last source frames
            job['seam'] = {'frames': seam(source), 'count': seam(source).shape[1] - 1}
        save = make_save_wrapper(av.save_video)
        record = make_record_wrapper(lambda video_path, configs, is_image, audio_only, gen: None, lambda: {'audio_output_codec': 'aac_128'})
        final = Path(td) / 'clip.mp4'
        if with_audio:
            temporary = Path(td) / 'clip_tmp.mp4'
            save(tensor=[output], save_file=str(temporary), fps=FPS, nrow=1, normalize=True, value_range=(-1, 1),
                 codec_type='libx264_8', container='mp4')
            # Source part silent, new part a loud tone: the second part must start loud.
            total = (SOURCE + NEW) / FPS
            t = np.arange(round(total * 32000)) / 32000
            audio = np.where(t >= SOURCE / FPS, 0.5 * np.sin(2 * np.pi * 440 * t), 0.0).astype(np.float32)
            wav = Path(td) / 'audio.wav'; sf.write(wav, np.stack([audio, audio], 1), 32000)
            if audio_tail:  # WanGP 18.19+: the saved audio runs audio_tail seconds past the last frame
                t = np.arange(round((total + audio_tail) * 32000)) / 32000
                audio = np.where(t >= SOURCE / FPS, 0.5 * np.sin(2 * np.pi * 440 * t), 0.0).astype(np.float32)
                sf.write(wav, np.stack([audio, audio], 1), 32000)
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(temporary), '-i', str(wav), '-map', '0:v', '-map', '1:a',
                                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', str(final)], check=True)
            else:
                av.combine_video_with_audio_tracks(str(temporary), [str(wav)], str(final))
            temporary.unlink()
        else:
            save(tensor=[output], save_file=str(final), fps=FPS, nrow=1, normalize=True, value_range=(-1, 1),
                 codec_type='libx264_8', container='mp4')
        before = final.read_bytes()
        record(str(final), {}, False, False, {})
        assert final.read_bytes() == before, 'Complete video was modified'
        assert not list(Path(td).glob('*h3part2_tmp*')), 'Temporary second-part video left behind'
        return final, job
    finally:
        JOB.reset(token)


with tempfile.TemporaryDirectory() as td:
    final, job = render(td)
    part2 = final.with_name('clip_part2.mp4')
    assert job.get('part2_saved') == [str(part2)] and part2.is_file()
    streams = probe(part2)
    assert int(streams['video']['nb_read_frames']) == NEW, streams
    assert abs(float(streams['audio']['duration']) - NEW / FPS) < 0.06, streams
    assert int(probe(final)['video']['nb_read_frames']) == SOURCE + NEW
    color = first_frame_color(part2)
    assert color[1] > 150 and color[0] < 60, f'Second part does not start on the first new frame: {color}'
    wav = part2.with_suffix('.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(part2), str(wav)], check=True)
    audio, rate = sf.read(wav, always_2d=True)
    head = audio[:round(0.04 * rate)]
    assert np.sqrt(np.mean(head ** 2)) > 0.15, 'Second-part audio does not start at the new part (source silence included)'
    print(f'PASS: second part = {NEW} new frames (source {SOURCE}, overlap {OVERLAP}, trimmed tail {TRIM}); '
          'first frame is new, audio aligned, complete video unchanged.')

K = 5
def redecode(source):
    # switch frame = the source frame itself (identical decode); the last K frames slightly different (R 200 -> 214)
    last = torch.cat([frame((214, 0, 0), 3000 + i) for i in range(K)], dim=1)
    return torch.cat([source[:, SOURCE - K - 1:SOURCE - K], last], dim=1)

def red(path, index):
    capture = cv2.VideoCapture(str(path)); capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, image = capture.read(); capture.release()
    assert ok
    return float(image[H//2, W//2][2])

with tempfile.TemporaryDirectory() as td:
    final, job = render(td, seam=redecode)
    part2 = final.with_name('clip_part2.mp4')
    assert job['seam_applied']['frames'] == K and job['seam_applied']['switch_difference'] < 0.01
    assert int(probe(final)['video']['nb_read_frames']) == SOURCE + NEW
    reds = [red(final, i) for i in range(SOURCE - K - 2, SOURCE + 1)]
    assert all(abs(r - 200) < 4 for r in reds[:2]) and all(abs(r - 214) < 4 for r in reds[2:2 + K]), reds
    streams = probe(part2)
    assert int(streams['video']['nb_read_frames']) == NEW + K, streams
    assert abs(red(part2, 0) - 214) < 4
    wav = part2.with_suffix('.wav'); subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(part2), str(wav)], check=True)
    audio, rate = sf.read(wav, always_2d=True)
    quiet, loud = audio[:round(0.15 * rate)], audio[round((K / FPS + 0.02) * rate):round((K / FPS + 0.06) * rate)]
    assert np.sqrt(np.mean(quiet ** 2)) < 0.05 and np.sqrt(np.mean(loud ** 2)) > 0.15, 'second-part audio misaligned with the earlier start'
    print(f'PASS: seamless end: last {K} source frames replaced in the complete video, switch frame identical, '
          f'second part starts {K} frames earlier with its audio aligned.')

with tempfile.TemporaryDirectory() as td:
    final, job = render(td, seam=lambda source: torch.cat([frame((0, 200, 0), 4000 + i) for i in range(K + 1)], dim=1))
    assert 'seam_applied' not in job and int(probe(final.with_name('clip_part2.mp4'))['video']['nb_read_frames']) == NEW
    assert abs(red(final, SOURCE - 1) - 200) < 4
    print('PASS: inconsistent re-decode refused (source end kept, second part unchanged).')

with tempfile.TemporaryDirectory() as td:
    final, job = render(td, audio_tail=0.1)
    streams = probe(final.with_name('clip_part2.mp4'))
    full = probe(final)
    assert int(streams['video']['nb_read_frames']) == NEW
    assert abs(float(streams['audio']['duration']) - (NEW / FPS + 0.1)) < 0.06, streams
    assert abs((float(full['audio']['duration']) - float(full['video']['duration'])) -
               (float(streams['audio']['duration']) - float(streams['video']['duration']))) < 0.04
    print('PASS: audio tail past the last frame (WanGP 18.19+) kept in the second part, as in the complete video.')

with tempfile.TemporaryDirectory() as td:
    final, job = render(td, with_audio=False)
    part2 = final.with_name('clip_part2.mp4')
    streams = probe(part2)
    assert int(streams['video']['nb_read_frames']) == NEW and 'audio' not in streams
    print('PASS: second part without audio when the complete video has none.')

with tempfile.TemporaryDirectory() as td:
    final, job = render(td, tamper=True)
    assert not final.with_name('clip_part2.mp4').exists() and not job.get('part2_saved')
    print('PASS: altered output tail -> second part skipped with a warning; render not failed.')

with tempfile.TemporaryDirectory() as td:
    final, job = render(td, continue_video=False)
    assert not final.with_name('clip_part2.mp4').exists() and job['part2'] is None
    print('PASS: no second part outside Continue Video.')

# Chunk slicing keeps each layout (channel-first and channel-last chunks, tensors).
chunks = [torch.arange(3*5*2*2).view(3, 5, 2, 2), torch.zeros(4, 2, 2, 3)]
sliced = frames_from(chunks, 3)
assert sliced[0].shape == (3, 2, 2, 2) and sliced[1].shape == (4, 2, 2, 3)
assert torch.equal(sliced[0], chunks[0][:, 3:])
assert frames_from(chunks, 6)[0].shape == (3, 2, 2, 3)
assert frames_from(torch.zeros(1, 3, 9, 2, 2), 4).shape == (1, 3, 5, 2, 2)
print('PASS: frame slicing for native save_video inputs.')
