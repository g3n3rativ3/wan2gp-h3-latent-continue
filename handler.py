"""Distinct model architectures; forwarding keeps WanGP's existing form/features."""
import functools
import inspect
import json
from pathlib import Path
import types

from models.minimax_h3.minimax_h3_handler import family_handler as Native

ROOT = Path(__file__).resolve().parent
IDS = json.loads((ROOT / 'model_ids.json').read_text())


class Handler:
    @staticmethod
    def query_supported_types(): return list(IDS)
    @staticmethod
    def query_model_family(): return 'h3_latent_continue'
    @staticmethod
    def query_family_infos(): return {'h3_latent_continue': (71, 'MiniMax H3 Latent Continue')}
    @staticmethod
    def query_family_maps():
        return {k: 'h3_latent_fl2va' for k in IDS if k != 'h3_latent_fl2va'}, {}
    @staticmethod
    def register_lora_cli_args(parser, lora_root): pass  # Native handler already registers shared H3 option.

    def __getattr__(self, name):
        native = getattr(Native, name)
        if not callable(native): return native
        sig = inspect.signature(native)
        @functools.wraps(native)
        def forward(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            base = bound.arguments.get('base_model_type')
            if base in IDS: bound.arguments['base_model_type'] = IDS[base]
            result = native(*bound.args, **bound.kwargs)
            if name == 'query_model_def':
                result.update({'guidance_max_phases':1,'lock_guidance_phases':True,
                               'lora_multiplier_phases':1,'phase_2_spatial_tiling':False})
            if name == 'load_model':
                from .pipeline import MiniMaxH3Pipeline
                from .layout import _layout
                pipeline, offload = result
                from .lora_adapter import install_lora_adapter
                install_lora_adapter(pipeline.transformer, IDS)
                pipeline.__class__ = MiniMaxH3Pipeline
                # Only the new prototype's transformer instance; no upstream module monkey-patch.
                pipeline.transformer._layout = types.MethodType(_layout, pipeline.transformer)
                filename = bound.arguments.get('model_filename', '')
                if isinstance(filename, (tuple,list)): filename = filename[0]
                from models.minimax_h3.minimax_h3_main import VIDEO_VAE_FILE, AUDIO_VAE_FILE
                pipeline._lc_identity = {
                    'architecture': IDS.get(base, base),
                    'checkpoint_name': Path(str(filename)).name,
                    'video_vae': str(bound.arguments.get('model_def', {}).get('video_vae_file', VIDEO_VAE_FILE)),
                    'audio_vae': str(bound.arguments.get('model_def', {}).get('audio_vae_file', AUDIO_VAE_FILE)),
                    'normalization': 'WanGP-H3-LATENTS_MEAN_STD-v1',
                }
            return result
        return forward

family_handler = Handler()
