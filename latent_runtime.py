"""Per-job capture/replay. No inference tensors or user choices in process-global slots."""
from contextvars import ContextVar
import hashlib
import math
import torch
from .checkpoint import load_checkpoint, tail_indices
from .seams import seam_settings, audio_window

JOB = ContextVar('h3_latent_job', default=None)


def frame_bytes(frame):
    if frame.dtype != torch.uint8:
        frame = frame.to(torch.float16).add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
    return frame.cpu().contiguous().numpy().tobytes()


def frame_hash(frame): return hashlib.sha256(frame_bytes(frame)).hexdigest()


class LatentMixin:
    def _lc_begin(self, args):
        self._lc_source = None
        self._lc_manifest = None
        self._lc_audio_prefix = None
        self._lc_audio_origin = 0.0
        self._lc_join_mode = 'baseline'
        job = JOB.get()
        if int(args.get('guide_phases', 1)) != 1:
            raise ValueError('H3 Latent Continue: use one generation phase.')
        if not job or not job['enabled']: return
        if (args.get('custom_settings') or {}).get('audio_refinement', 'none') != 'none':
            raise ValueError('H3 Latent Continue: disable the extra audio refinement phase.')
        if args.get('refinement_mode') or self.audio_only or self.fixed_prompt is not None or self.transformer.pdd_num_steps is not None:
            raise ValueError('H3 Latent Continue: refinement, PDD, audio-only and fixed-prompt models are unsupported.')
        if int(args.get('kwargs', {}).get('window_no', 1)) > 1:
            raise ValueError('H3 Latent Continue: one generated window per job; continue the saved result in a new job.')
        if args.get('image_start') is not None and job['options']['continue']:
            raise ValueError('H3 Latent Continue: select Continue Video, without a Start Image.')
        job['pending'] = None
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
            if self._lc_join_mode == 'audio_prefix':
                if self.transformer.cache is not None:
                    raise ValueError('H3 Latent audio-prefix mode: disable all step caches (No Skipping).')
                if args.get('audio_guide') is not None or args.get('audio_guide2') is not None:
                    raise ValueError('H3 Latent audio-prefix mode does not support additional audio guides.')
                start, stop, self._lc_audio_origin = audio_window(info, tensors['audio'].shape[-1], self._lc_audio_seconds)
                self._lc_audio_prefix = tensors['audio'][...,start:stop].clone()
                print(f'[H3 Join] frozen audio={stop-start} tokens, origin={self._lc_audio_origin:.6f}s, trim={round(-self._lc_audio_origin*32000)} samples')
            if self._lc_join_mode != 'baseline':
                print(f'[H3 Join] mode={self._lc_join_mode}, video context={self._lc_video_context} frames, audio context={self._lc_audio_seconds}s')
        job['identity'] = self._lc_identity
        job['settings'] = {k: args.get(k) for k in ('input_prompt','seed','sampling_steps','shift','sample_solver','guide_phases')}

    def _lc_video_conditions(self, latents, keyframes, overlap):
        src = self._lc_source['video']
        if getattr(self, '_lc_join_mode', 'baseline') != 'baseline':
            overlap = min(self._lc_video_context, self._lc_manifest['target_frames'])
        for index, position in tail_indices(src.shape[2],self._lc_manifest['target_frames'],overlap):
            latents.append(src[:,:,index:index+1].clone())
            keyframes.append({'anchor':'frame','frame_index':position,'latent_frame_count':1})
        print('[H3 Latent] Video context loaded directly; no VAE re-encode. '
              f'{len(keyframes)} time-positioned latent blocks.')

    def _lc_audio_conditions(self, latents, keyframes, overlap, fps):
        src = self._lc_source['audio']
        if getattr(self, '_lc_join_mode', 'baseline') == 'audio_prefix':
            return  # Past sound occupies frozen target rows, not separate guide rows.
        if getattr(self, '_lc_join_mode', 'baseline') == 'context':
            start, stop, origin = audio_window(self._lc_manifest, src.shape[-1], self._lc_audio_seconds)
            latents.append(src[...,start:stop].clone())
            keyframes.append({'anchor':origin * 40, 'latent_frame_count':stop-start})
            return
        frames = self._lc_manifest['target_frames']
        # Target frame zero represents the start of the last delivered source frame.
        source_origin = self._lc_manifest.get('audio_origin_seconds', 0.0) * 40
        boundary = (frames-1)*40.0/fps - source_origin
        start = max(0, math.floor((frames-overlap)*40.0/fps - source_origin))
        stop = min(src.shape[-1], math.ceil(frames*40.0/fps - source_origin))
        if stop > start:
            latents.append(src[...,start:stop].clone())
            keyframes.append({'anchor':float(start-boundary),'latent_frame_count':stop-start})

    def _lc_capture(self, video, audio, target_frames, fps, width, height):
        job = JOB.get()
        if not job or not job['options'].get('save'): return
        job['pending'] = {
            'tensors': {'video':video.detach().cpu().clone().contiguous(),
                        'audio':audio.detach().cpu().clone().contiguous()},
            'info': {'identity':job['identity'], 'target_frames':int(target_frames),'fps':float(fps),
                     'audio_origin_seconds':getattr(self, '_lc_audio_origin', 0.0),
                     'join_mode':getattr(self, '_lc_join_mode', 'baseline'),
                     'join_settings':{k:job['options'].get(k) for k in ('video_context_frames','audio_context_seconds')},
                     'width':int(width),'height':int(height),'settings':job['settings'],
                     'prototype_version':'0.2.2', 'continuation_mode':'latent' if self._lc_source is not None else 'pixels'},
        }

    def _lc_decoded(self, decoded):
        job = JOB.get()
        if not job or job.get('pending') is None: return
        # Match WanGP's fp16 buffer and truncation to uint8. Only fingerprints retained.
        pending = job['pending']
        first = max(0, decoded.shape[1]-71)
        hashes = [frame_hash(decoded[:,i:i+1]) for i in range(first,decoded.shape[1])]
        signatures = {}
        for end in range(max(first+8,decoded.shape[1]-63),decoded.shape[1]+1):
            signature = tuple(hashes[end-first-8:end-first])
            signatures.setdefault(signature, []).append(end)
        pending['tail_signatures'] = signatures
