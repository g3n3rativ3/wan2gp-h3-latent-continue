# Derived from Wan2GP MiniMax H3 transformer; see THIRD_PARTY.md.
import torch
from models.minimax_h3.transformer import _prepared_references
from .packing import build_packed_sequence, build_ref2va_packed_sequence
from .seams import shift_target_audio

def _layout(self, text_tags, latent_t, latent_h, latent_w, audio_t, payload):
    target_spatial_context = payload.get("target_spatial_context")
    signature = (text_tags.numel(), latent_t, latent_h, latent_w, audio_t,
                 payload["fps"],
                 payload.get('lc_audio_offset', 0.0),
                 target_spatial_context,
                 payload.get("target_audio_condition_latents", 0),
                 payload.get("target_video_condition_frames", 0),
                 tuple((k["anchor"], k["latent_frame_count"], k.get("frame_index")) for k in payload.get("keyframes") or ()),
                 tuple((k["anchor"], k["latent_frame_count"]) for k in payload.get("audio_keyframes") or ()),
                 tuple((r["kind"], r.get("latent_t"), r.get("latent_h"), r.get("latent_w"), r.get("ref_audio_t"))
                       for r in payload.get("refs") or ()))
    if payload.get("layout_signature") == signature:
        return payload["layout"]
    with torch.device("cpu"):
        video_time_scale = 24.0 / payload["fps"]
        anchors = tuple((keyframe["anchor"], keyframe["latent_frame_count"], keyframe.get("frame_index"))
                        for keyframe in payload.get("keyframes") or ())
        audio_anchors = tuple((keyframe["anchor"], keyframe["latent_frame_count"])
                              for keyframe in payload.get("audio_keyframes") or ())
        target_audio_condition_latents = payload.get("target_audio_condition_latents", 0)
        target_video_condition_frames = payload.get("target_video_condition_frames", 0)
        if payload.get("refs"):
            layout = build_ref2va_packed_sequence(text_tags, _prepared_references(payload["refs"]), latent_t,
                                                  latent_h, latent_w, audio_t, self.patch_size, video_time_scale,
                                                  keyframe_anchors=anchors, audio_condition_anchors=audio_anchors,
                                                  target_condition_audio_latents=target_audio_condition_latents,
                                                  target_condition_video_frames=target_video_condition_frames,
                                                  target_spatial_context=target_spatial_context)
        else:
            layout = build_packed_sequence(text_tags, latent_t, latent_h, latent_w, audio_t, self.patch_size,
                                           anchors, video_time_scale, audio_condition_anchors=audio_anchors,
                                           target_condition_audio_latents=target_audio_condition_latents,
                                           target_condition_video_frames=target_video_condition_frames,
                                           target_spatial_context=target_spatial_context)
    shift_target_audio(layout, payload.get('lc_audio_offset', 0.0))
    payload["layout_signature"], payload["layout"] = signature, layout
    return layout
