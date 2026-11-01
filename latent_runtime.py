"""Per-job capture/replay. No inference tensors or user choices in process-global slots."""
from contextvars import ContextVar
import hashlib
import math
import torch
from .checkpoint import load_checkpoint, tail_indices
from .seams import seam_settings, audio_window, reference_context_frames, pinned_geometry
from .checkpoint import token_start

JOB = ContextVar('h3_latent_job', default=None)


def frame_bytes(frame):
    if frame.dtype != torch.uint8:
        frame = frame.to(torch.float16).add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
    return frame.cpu().contiguous().numpy().tobytes()


def frame_hash(frame): return hashlib.sha256(frame_bytes(frame)).hexdigest()


def check_finite(stage, video=None, audio=None):
    """Stop with an explicit message when denoising produced NaN/inf (instead of a blank video and a tail mismatch)."""
    job = JOB.get()
    if job is not None and job.get('noise_only'):
        return  # plain render: native behaviour
    problems = []
    for name, tensor, axis, unit in (('video', video, 2, 'latent frames'), ('audio', audio, -1, 'audio latents')):
        if tensor is None or not torch.is_tensor(tensor):
            continue
        bad = ~torch.isfinite(tensor.detach())
        count = int(bad.sum())
        if count:
            dims = [d for d in range(bad.ndim) if d != axis % bad.ndim]
            where = bad.any(dim=dims[0], keepdim=True)
            for d in dims[1:]:
                where = where.any(dim=d, keepdim=True)
            where = where.flatten().nonzero().flatten()
            problems.append(f'{name}: {count} of {tensor.numel()} values, {unit} {int(where[0])}..{int(where[-1])} of {tensor.shape[axis]}')
    if problems:
        raise ValueError(f'H3 Latent: the model produced invalid values (NaN/inf) {stage} ({"; ".join(problems)}). '
                           'The render was stopped before decoding (the video would be blank); nothing was saved. '
                           'The source checkpoint is valid (it is checked when loaded). Retry the job; if it repeats, '
                           'try again with "Audio noise as WanGP 18.18" unchecked, then without the multi-GPU denoising '
                           'split, to find which one triggers it.')


class LatentMixin:
    def _lc_begin(self, args):
        self._lc_source = None
        self._lc_noise_1818 = False
        self._lc_manifest = None
        self._lc_audio_prefix = None
        self._lc_audio_origin = 0.0
        self._lc_join_mode = 'baseline'
        self._lc_context_info = {}
        self._lc_pinned = None
        self._lc_seam = None
        self._lc_phase1_from_final = False
        self._lc_phase1_pin = None
        job = JOB.get()
        if int(args.get('guide_phases', 1) or 1) not in (1, 2):
            raise ValueError('H3 Latent Continue: use one or two generation phases.')
        if not job or not job['enabled']: return
        if job.get('noise_only'):
            # Audio noise as WanGP 18.18 alone: native render otherwise (latent state above stays reset).
            self._lc_noise_1818 = True
            return
        if getattr(self.transformer, 'control_layers', None):
            raise ValueError('H3 Latent Continue: ControlNet-Union is unsupported with latent operations; disable both latent options to use native ControlNet rendering.')
        if args.get('refinement_mode') or self.audio_only or self.fixed_prompt is not None or self.transformer.pdd_num_steps is not None:
            raise ValueError('H3 Latent Continue: refinement, PDD, audio-only and fixed-prompt models are unsupported.')
        if int(args.get('kwargs', {}).get('window_no', 1)) > 1:
            raise ValueError('H3 Latent Continue: one generated window per job; continue the saved result in a new job.')
        if args.get('image_start') is not None and job['options']['continue']:
            raise ValueError('H3 Latent Continue: select Continue Video, without a Start Image.')
        job['pending'] = None
        job['phase1'] = None
        # Continue Video: WGP drops this many leading frames of the window, then keeps the source video.
        job['part2'] = None
        job['wgp_overlap'] = (int(args.get('prefix_frames_count') or 0)
                              if args.get('input_video') is not None and args.get('image_start') is None else 0)
        if job['options']['continue']:
            if args.get('input_video') is None or int(args.get('prefix_frames_count',0)) < 1:
                raise ValueError('H3 Latent Continue: Continue Video and a source video are required.')
            info, tensors = load_checkpoint(job['options']['latent_path'])
            if info['identity'] != self._lc_identity:
                raise ValueError('H3 Latent Continue: model/checkpoint/VAE identity differs. Select the original model and quantization.')
            if (info['width'],info['height']) != (int(args['width']),int(args['height'])) or abs(info['fps']-float(args['fps']))>1e-6:
                raise ValueError('H3 Latent Continue: source and target resolution/FPS must match exactly.')
            if int(args['prefix_frames_count']) > info['target_frames']:
                raise ValueError('H3 Latent Continue: overlap exceeds the saved final generated segment; reduce overlap.')
            self._lc_source, self._lc_manifest = tensors, info
            self._lc_join_mode, self._lc_video_context, self._lc_audio_seconds = seam_settings(job['options'])
            if self._lc_join_mode in ('audio_prefix', 'pinned'):
                if self.transformer.cache is not None:
                    raise ValueError('H3 Latent audio-prefix / exact join: disable all step caches (No Skipping).')
                # Audio guides used as references (Ref2VA voices, RefMods) are separate condition rows and are fine.
                # A guide that drives the target soundtrack itself is refused when the frozen prefix is placed.
                start, stop, self._lc_audio_origin = audio_window(info, tensors['audio'].shape[-1], self._lc_audio_seconds)
                self._lc_audio_prefix = tensors['audio'][...,start:stop].clone()
            if self._lc_join_mode == 'pinned':
                pin = pinned_geometry(info['target_frames'], tensors['video'].shape[2], job['wgp_overlap'])
                pin['overlap'] = int(job['wgp_overlap'])
                # The new window starts `frames - 1` frames before the last delivered frame: move the audio origin with it.
                self._lc_audio_origin += (pin['frames'] - 1) / info['fps']
                if self._lc_audio_origin > 1e-9:
                    raise ValueError(f'H3 Latent exact join: the audio context ({self._lc_audio_seconds:g} s) must cover the '
                                     f'{pin["frames"]} pinned frames; choose a longer audio context.')
                self._lc_pinned = pin
                # Two phases: phase 1 follows the FINAL video (as native H3, which re-encodes the source frames at the
                # phase-1 size) instead of the source's phase-1 draft, which phase 2 later redrew.
                self._lc_phase1_from_final = bool(job['options'].get('phase1_from_final', True))
                # Seamless end: the source's last (partial) VAE chunk was decoded without its future. When the pin
                # includes the full chunk before it, the new window decodes that chunk identically and re-decodes
                # the last frames with the real future: they replace the source's last frames in the complete video.
                tail = info['target_frames'] - 17 * ((info['target_frames'] - 1) // 17)
                if job['options'].get('seamless_end', True):
                    if pin['frames'] >= tail + 17:
                        self._lc_seam = {'frames': int(tail)}
                        print(f'[H3 Join] seamless end: the last {tail} source frames will be replaced by their re-decode.')
                    else:
                        needed = 1 + 17 * math.ceil((tail + 16) / 17)  # next value of H3's overlap slider (1, 18, 35, 52...)
                        print(f'[H3 Join] seamless end skipped: set WanGP "Windows Frames Overlap" (Sliding Window) to {needed} or more '
                              f'(current {pin["overlap"]}; "Video latent context" is a different setting).')
                print(f'[H3 Join] exact join: Windows Frames Overlap {pin["overlap"]} -> window starts on source frame {pin["clip_start"]}; {pin["frames"]} frames pinned '
                      f'to saved latents {pin["first_latent"]}..{pin["first_latent"] + pin["latents"] - 1}; '
                      f'audio prefix from {self._lc_audio_origin:.6f}s.')
                print(f'[H3 Join] frozen audio={stop-start} tokens, origin={self._lc_audio_origin:.6f}s, trim={round(-self._lc_audio_origin*32000)} samples')
            if self._lc_join_mode != 'baseline':
                print(f'[H3 Join] mode={self._lc_join_mode}, video context={self._lc_video_context} frames, audio context={self._lc_audio_seconds}s')
        job['identity'] = self._lc_identity
        job['settings'] = {k: args.get(k) for k in ('input_prompt','seed','sampling_steps','shift','sample_solver','guide_phases',
                                                    'sampling_steps2','switch_threshold')}
        self._lc_noise_1818 = bool(job['options'].get('audio_noise_1818', False))
        job['settings']['audio_noise'] = 'wangp-18.18' if self._lc_noise_1818 else 'native'

    def _lc_resample(self, latents, size):
        """Spatial resample of a latent sequence (bilinear, antialiased when shrinking), cached per source tensor."""
        cache = self._lc_source.setdefault('_resampled', {})
        key = (id(latents), size)
        if key not in cache:
            frames = latents.permute(0, 2, 1, 3, 4).flatten(0, 1).float()  # (T, C, H, W): spatial only
            shrink = size[0] < latents.shape[-2] or size[1] < latents.shape[-1]
            frames = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False, antialias=shrink)
            cache[key] = frames.unflatten(0, (latents.shape[0], latents.shape[2])).permute(0, 2, 1, 3, 4).to(latents.dtype).contiguous()
        return cache[key]

    def _lc_stage_video(self, stage_height, stage_width):
        """Saved latents at this stage's size: final (full) or phase-1 (two-phase low resolution)."""
        if stage_height is None or stage_width is None:
            return self._lc_source['video'], 'single'
        size = (math.ceil(int(stage_height) / 16), math.ceil(int(stage_width) / 16))
        if getattr(self, '_lc_pinned', None) is not None and getattr(self, '_lc_phase1_from_final', False) \
                and size != tuple(self._lc_source['video'].shape[-2:]):
            return self._lc_resample(self._lc_source['video'], size), 'final video reduced to phase 1'
        for key, label in (('video', 'full resolution'), ('video_phase1', 'phase 1')):
            tensor = self._lc_source.get(key)
            if tensor is not None and tuple(tensor.shape[-2:]) == size:
                return tensor, label
        phase1 = self._lc_source.get('video_phase1')
        if phase1 is None:
            raise ValueError(f'H3 Latent Continue: this checkpoint has no {stage_width}x{stage_height} phase-1 latents: it was saved '
                             'from a single-phase render. Continue it with one phase, or render the source clip with two phases '
                             'and Save latent checkpoint enabled.')
        # The target is checked at start, so only phase 1 can differ: its size follows the two-phase scale
        # and rounding (H3_TWO_PHASE_SCALE, which another plugin or setting may change between renders).
        saved = (int(phase1.shape[-1]) * 16, int(phase1.shape[-2]) * 16)
        cache = self._lc_source.setdefault('_phase1_resampled', {})
        if size not in cache:
            cache[size] = self._lc_resample(phase1, size)
            print(f'[H3 Latent] Phase 1 is {stage_width}x{stage_height} but the checkpoint phase-1 latents are {saved[0]}x{saved[1]} '
                  '(two-phase scale or resolution rounding changed since the source render): phase-1 latents resampled for phase 1 '
                  '(approximate); phase 2 still uses the exact full-resolution latents.')
        if not hasattr(self, '_lc_context_info'):
            self._lc_context_info = {}
        self._lc_context_info['phase1_resampled_from'] = [saved[0], saved[1]]
        return cache[size], 'phase 1 resampled'

    def _lc_pinned_target(self, requested):
        """Exact join: largest H3 length (5+17n) not above the requested target frames."""
        requested = int(requested)
        target = requested if requested <= 5 else 5 + 17 * ((requested - 5) // 17)
        pin = self._lc_pinned
        pin['requested_target_frames'], pin['target_frames'] = requested, target
        if target != requested:
            print(f'[H3 Join] exact join: {requested} new-window frames is not an H3 length (5+17n); generating {target} so the '
                  f'scene ends where the video ends (the new part is {requested - target} frame(s) shorter than requested).')
        return target

    def _lc_pin_history(self, continuation):
        """Exact join: pixel frames before the pinned region (output prefix, dropped by WanGP with the overlap)."""
        pin = self._lc_pinned
        overlap, frames = pin['overlap'], pin['frames']
        if continuation is None or continuation.shape[1] < overlap:
            raise ValueError('H3 Latent exact join: the source overlap frames are missing.')
        history = overlap - frames
        return (continuation[:, -overlap:-frames] if history else None), history

    def _lc_presented_frame(self, continuation):
        """Image shown to the prompt encoder for target frame 0."""
        pin = getattr(self, '_lc_pinned', None)
        if pin is None:
            return self._lc_source['last_frame']
        return continuation[:, -pin['frames']:continuation.shape[1] - pin['frames'] + 1]

    def _lc_noise_tail(self, tail, audio_t):
        """Audio tail latents (WanGP 18.19+) left out of the seeded audio noise, as before 18.19; 0 = native draw."""
        tail = int(tail or 0)
        if not getattr(self, '_lc_noise_1818', False):
            return 0
        if not 0 < tail < int(audio_t):
            print('[H3 Latent] Audio noise as WanGP 18.18: no audio tail in this WanGP version; native noise is already identical.')
            return 0
        print(f'[H3 Latent] Audio noise as WanGP 18.18: seeded noise drawn for {int(audio_t) - tail} audio latents, '
              f'the {tail} tail latents from a separate generator.')
        return tail

    def _lc_with_tail(self, noise, tail, seed, salt=0):
        """Append independent noise for the audio tail without consuming the seeded generator."""
        if not tail:
            return noise
        generator = torch.Generator(device='cpu').manual_seed((int(seed) * 1000003 + 0x7A11 + 7919 * int(salt)) % (2 ** 63))
        extra = torch.randn(tuple(noise.shape[:-1]) + (int(tail),), generator=generator, dtype=torch.float32, device='cpu')
        return torch.cat((noise, extra.to(device=noise.device, dtype=noise.dtype)), dim=-1)

    def _lc_prefix_tokens(self, prefix):
        """Audio tokens added before target frame 0 for the frozen audio prefix."""
        if getattr(self, '_lc_pinned', None) is None:
            return prefix.shape[-1]
        return math.ceil(-self._lc_audio_origin * 40 - 1e-9)

    def _lc_pin_latents(self, video, noise, sigma):
        """Pin the first target latents to the saved ones with WanGP's masked denoising (prediction blended,
        exact at the last step), for this stage's resolution. Returns source, noise, buffer, editable mask."""
        pin = self._lc_pinned
        first, count = pin['first_latent'], pin['latents']
        if count >= video.shape[2]:
            raise ValueError('H3 Latent exact join: no frame left to generate after the pinned frames.')
        encoded = getattr(self, '_lc_phase1_pin', None)
        if encoded is not None and tuple(encoded.shape[-2:]) == tuple(video.shape[-2:]):
            known, stage = encoded.to(device=video.device, dtype=video.dtype), 'final video re-encoded at phase-1 size'
        else:
            saved, stage = self._lc_stage_video(video.shape[-2] * 16, video.shape[-1] * 16)
            known = saved[:, :, first:first + count].to(device=video.device, dtype=video.dtype)
        source = torch.zeros_like(video)
        source[:, :, :count] = known
        noise = video.detach().clone() if noise is None else noise.to(video).clone()
        mask = torch.ones((1, 1) + tuple(video.shape[2:]), device=video.device, dtype=video.dtype)
        mask[:, :, :count] = 0
        video[:, :, :count] = torch.lerp(known, noise[:, :, :count], float(sigma))
        self._lc_context_info.setdefault('pinned_stages', []).append(stage)
        print(f'[H3 Join] {count} target latents pinned to the saved {stage} latents.')
        return source, noise, torch.empty_like(source), mask

    def _lc_video_conditions(self, latents, keyframes, overlap, stage_height=None, stage_width=None, continuation=None, stage_video=None):
        src, stage = self._lc_stage_video(stage_height, stage_width)
        native_overlap = overlap
        if getattr(self, '_lc_pinned', None) is not None:
            pin = self._lc_pinned
            if stage == 'final video reduced to phase 1':
                # Pinned frames: the final source frames (pixels of the overlap) encoded at the phase-1 size by the
                # native VAE, on the window's own chunk grid (it starts on a chunk boundary).
                if continuation is None or stage_video is None or continuation.shape[1] < pin['frames']:
                    raise ValueError('H3 Latent exact join: the overlap frames are needed for phase 1.')
                # Every chunk's latents are kept (causal encoder: the last chunk's padding only affects latents after the
                # pinned frames), then the first `latents` ones cover exactly the pinned frames for any 1/5/9/13 + 17n count.
                encoded = self._encode_video(stage_video(continuation[:, -pin['frames']:]), keep_all_latents=True)[:, :, :pin['latents']]
                if encoded.shape[2] != pin['latents'] or tuple(encoded.shape[-2:]) != tuple(src.shape[-2:]):
                    raise ValueError(f'H3 Latent exact join: phase-1 re-encode gave {tuple(encoded.shape)}, '
                                     f'expected {pin["latents"]} latents of {tuple(src.shape[-2:])}.')
                self._lc_phase1_pin = encoded
            start = pin['clip_start'] - self._lc_video_context
            selected = [(i, token_start(i) - pin['clip_start']) for i in range(pin['first_latent']) if token_start(i + 1) > start]
            resampled = getattr(self, '_lc_context_info', {}).get('phase1_resampled_from')
            stages = getattr(self, '_lc_context_info', {}).get('pinned_stages')
            self._lc_context_info = {'native_overlap_frames': int(native_overlap), 'video_context_frames': int(self._lc_video_context),
                                     'video_latent_indices': [i for i, _ in selected], 'video_frame_positions': [p for _, p in selected],
                                     'pinned': {k: int(v) for k, v in pin.items()}}
            if resampled is not None:
                self._lc_context_info['phase1_resampled_from'] = resampled
            if stages:
                self._lc_context_info['pinned_stages'] = stages
            for index, position in selected:
                latents.append(src[:, :, index:index + 1].clone())
                keyframes.append({'anchor': 'frame', 'frame_index': position, 'latent_frame_count': 1})
            if stage == 'final video reduced to phase 1':
                print(f'[H3 Latent] Exact join, phase 1 follows the final video: {pin["frames"]} pinned frames re-encoded at the phase-1 '
                      f'size, {len(selected)} context blocks from the final latents reduced to phase 1.')
            else:
                print(f'[H3 Latent] Exact join ({stage} latents): {len(selected)} context blocks before the pinned frames; no VAE re-encode.')
            return
        if getattr(self, '_lc_join_mode', 'baseline') != 'baseline':
            overlap = min(self._lc_video_context, self._lc_manifest['target_frames'])
        else:
            overlap = reference_context_frames(self._lc_manifest['target_frames'], overlap)
        selected = tail_indices(src.shape[2],self._lc_manifest['target_frames'],overlap)
        resampled = getattr(self, '_lc_context_info', {}).get('phase1_resampled_from')
        self._lc_context_info = {'native_overlap_frames':int(native_overlap),
                                 'video_context_frames':int(overlap),
                                 'video_latent_indices':[i for i, _ in selected],
                                 'video_frame_positions':[p for _, p in selected]}
        if resampled is not None:
            self._lc_context_info['phase1_resampled_from'] = resampled
        for index, position in selected:
            latents.append(src[:,:,index:index+1].clone())
            keyframes.append({'anchor':'frame','frame_index':position,'latent_frame_count':1})
        print(f'[H3 Latent] Video context loaded directly ({stage} latents); no VAE re-encode. '
              f'{len(selected)} time-positioned latent blocks; context={overlap} frames, assembly overlap={native_overlap}.')

    def _lc_audio_conditions(self, latents, keyframes, overlap, fps):
        src = self._lc_source['audio']
        if not hasattr(self, '_lc_context_info'):
            self._lc_context_info = {}
        if getattr(self, '_lc_join_mode', 'baseline') in ('audio_prefix', 'pinned'):
            return  # Past sound occupies frozen target rows, not separate guide rows.
        if getattr(self, '_lc_join_mode', 'baseline') == 'context':
            start, stop, origin = audio_window(self._lc_manifest, src.shape[-1], self._lc_audio_seconds)
            latents.append(src[...,start:stop].clone())
            keyframes.append({'anchor':origin * 40, 'latent_frame_count':stop-start})
            return
        frames = self._lc_manifest['target_frames']
        overlap = reference_context_frames(frames, overlap)
        # Target frame zero represents the start of the last delivered source frame.
        source_origin = self._lc_manifest.get('audio_origin_seconds', 0.0) * 40
        boundary = (frames-1)*40.0/fps - source_origin
        start = max(0, math.floor((frames-overlap)*40.0/fps - source_origin))
        stop = min(src.shape[-1], math.ceil(frames*40.0/fps - source_origin))
        if stop > start:
            latents.append(src[...,start:stop].clone())
            keyframes.append({'anchor':float(start-boundary),'latent_frame_count':stop-start})
            self._lc_context_info['audio_latent_range'] = [int(start), int(stop)]

    def _lc_refinement(self, payload, main_payload, audio, condition_latents):
        """Audio Refinement Extra Phase of a latent continuation: keep the join anchored.

        Native refinement re-noises the whole soundtrack with no condition. For a latent
        join it keeps the saved-latent audio conditions, the frozen audio prefix (restored
        after the re-noise) and the prefix time offset. Returns the frozen target tokens."""
        check_finite('before the audio refinement', None, audio)
        if getattr(self, '_lc_source', None) is None or not isinstance(main_payload, dict):
            return condition_latents
        payload['lc_audio_offset'] = main_payload.get('lc_audio_offset', 0.0)
        keyframes = list(main_payload.get('audio_keyframes') or ())
        rows = main_payload.get('cond_audio_rows')
        count = sum(int(k['latent_frame_count']) for k in keyframes) * 2  # two packed audio channels per token
        if count:
            if rows is None or rows.shape[0] < count:
                raise ValueError('H3 Latent: audio condition rows missing for the refinement phase.')
            # Packed layout order: audio keyframe rows first, then reference rows (if kept by another plugin).
            kept, other = rows[:count], payload.get('cond_audio_rows')
            payload['audio_keyframes'] = [dict(k) for k in keyframes]
            payload['cond_audio_rows'] = kept if other is None else torch.cat((kept, other.to(kept)))
        prefix = getattr(self, '_lc_audio_prefix', None)
        if prefix is not None:
            condition_latents = int(prefix.shape[-1])
            audio[..., :condition_latents].copy_(prefix.to(audio))
            payload['target_audio_condition_latents'] = condition_latents
        print(f'[H3 Latent] Audio refinement keeps the join: {count // 2} conditioned audio tokens'
              + (f', frozen prefix {condition_latents} tokens.' if prefix is not None else '.'))
        return condition_latents

    def _lc_capture_phase1(self, video, audio=None):
        """Two-phase render: keep the low-resolution phase-1 result before the latent upscaler."""
        check_finite('at the end of phase 1', video, audio)
        job = JOB.get()
        if not job or not job['options'].get('save'): return
        job['phase1'] = video.detach().cpu().clone().contiguous()

    def _lc_capture(self, video, audio, target_frames, fps, width, height, history_count=0):
        check_finite('at the end of denoising', video, audio)
        job = JOB.get()
        if job and job.get('wgp_overlap', 0) > 0:
            # The window output is history_count copied frames + decoded target frames.
            job['part2'] = {'overlap':int(job['wgp_overlap']), 'history':int(history_count)}
        if not job or not job['options'].get('save'): return
        tensors = {'video':video.detach().cpu().clone().contiguous(),
                   'audio':audio.detach().cpu().clone().contiguous()}
        phase1, job['phase1'] = job.get('phase1'), None
        if phase1 is not None:
            if phase1.shape[:3] != tensors['video'].shape[:3]:
                raise ValueError('H3 Latent: phase-1 and final latents have different lengths; refusing to save a mismatched checkpoint.')
            tensors['video_phase1'] = phase1
        job['pending'] = {
            'tensors': tensors,
            'info': {'identity':job['identity'], 'target_frames':int(target_frames),'fps':float(fps),
                     'audio_origin_seconds':getattr(self, '_lc_audio_origin', 0.0),
                     'join_mode':getattr(self, '_lc_join_mode', 'baseline'),
                     'effective_context':getattr(self, '_lc_context_info', {}),
                     'join_settings':{k:job['options'].get(k) for k in ('video_context_frames','audio_context_seconds')},
                     'width':int(width),'height':int(height),'settings':job['settings'],
                     'guide_phases':2 if phase1 is not None else 1,
                     **({'phase1':{'width':int(phase1.shape[-1])*16,'height':int(phase1.shape[-2])*16}} if phase1 is not None else {}),
                     'prototype_version':'0.6.5', 'continuation_mode':'latent' if self._lc_source is not None else 'pixels'},
        }

    def _lc_decoded(self, decoded):
        job = JOB.get()
        seam, pin = getattr(self, '_lc_seam', None), getattr(self, '_lc_pinned', None)
        if job is not None and seam is not None and pin is not None:
            # Decoded target frames: [pinned frames | new frames]. Keep the last `frames` pinned frames, plus the one
            # before them (the switch frame, expected identical to the source decode) for the export check.
            first = pin['frames'] - seam['frames'] - 1
            part = decoded[:, first:pin['frames']]
            if part.dtype != torch.uint8:
                part = part.to(torch.float16).add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
            job['seam'] = {'frames': part.detach().cpu().clone(), 'count': int(seam['frames'])}
        if not job or (job.get('pending') is None and job.get('part2') is None): return
        # Match WanGP's fp16 buffer and truncation to uint8. Only fingerprints retained.
        first = max(0, decoded.shape[1]-71)
        hashes = [frame_hash(decoded[:,i:i+1]) for i in range(first,decoded.shape[1])]
        signatures = {}
        for end in range(max(first+8,decoded.shape[1]-63),decoded.shape[1]+1):
            signature = tuple(hashes[end-first-8:end-first])
            signatures.setdefault(signature, []).append(end)
        for holder in (job.get('pending'), job.get('part2')):
            if holder is not None:
                holder['tail_signatures'] = signatures
