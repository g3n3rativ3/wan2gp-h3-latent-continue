"""Translate plugin architecture IDs at the native LoRA preprocessing boundary."""
import functools


def install_lora_adapter(transformer, model_ids):
    original = transformer.preprocess_loras
    if getattr(original, '_h3_latent_adapter', False):
        return
    mapping = dict(model_ids)

    @functools.wraps(original)
    def preprocess_loras(model_type, state_dict):
        return original(mapping.get(model_type, model_type), state_dict)

    preprocess_loras._h3_latent_adapter = True
    # Instance-local: native H3 models and other plugins keep their own methods.
    transformer.preprocess_loras = preprocess_loras
