import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lora_adapter', ROOT / 'lora_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
IDS = json.loads((ROOT / 'model_ids.json').read_text())


class Transformer:
    def preprocess_loras(self, model_type, state_dict):
        if model_type not in IDS.values():
            raise ValueError('Unsupported architecture: ' + model_type)
        self.received = model_type
        return state_dict


class LoraAdapterTests(unittest.TestCase):
    def test_four_variants_and_native_ids(self):
        transformer = Transformer()
        adapter.install_lora_adapter(transformer, IDS)
        payload = {'tensor': object()}
        for prototype, native in IDS.items():
            for name in (prototype, native):
                self.assertIs(transformer.preprocess_loras(name, payload), payload)
                self.assertEqual(transformer.received, native)

    def test_isolation_idempotence_and_errors(self):
        target, other = Transformer(), Transformer()
        adapter.install_lora_adapter(target, IDS)
        installed = target.preprocess_loras
        adapter.install_lora_adapter(target, IDS)
        self.assertIs(target.preprocess_loras, installed)
        with self.assertRaises(ValueError):
            other.preprocess_loras('h3_latent_fl2va_pruned', {})
        with self.assertRaises(ValueError):
            target.preprocess_loras('unknown', {})
