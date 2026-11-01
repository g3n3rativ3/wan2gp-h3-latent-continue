"""CPU tests for audio refinement support; run: python tests/test_refinement.py"""
import contextlib
import importlib.util
import io
import pathlib
import sys
import tempfile
import types
import unittest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h3refine', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec); sys.modules['h3refine'] = pkg; spec.loader.exec_module(pkg)
from h3refine.latent_runtime import LatentMixin
from h3refine.native_bridge import compile_edit
from h3refine.integration import preflight


class RefinementTests(unittest.TestCase):
    def test_join_rows_precede_kept_reference_rows(self):
        pipe = types.SimpleNamespace(_lc_source={'video': None}, _lc_audio_prefix=None)
        keyframe_rows, reference_rows = torch.ones(6, 32), torch.full((4, 32), 2.0)
        main = {'audio_keyframes': [{'anchor': -2.0, 'latent_frame_count': 3}],
                'cond_audio_rows': torch.cat((keyframe_rows, reference_rows)), 'lc_audio_offset': 0.0}
        # H3 Audio Refinement Controls "Keep Audio References" already put the reference rows here.
        payload = {'audio_keyframes': None, 'cond_audio_rows': reference_rows.clone(), 'refs': [{'kind': 'audio', 'ref_audio_t': 2}]}
        with contextlib.redirect_stdout(io.StringIO()):
            frozen = LatentMixin._lc_refinement(pipe, payload, main, torch.zeros(1, 32, 2, 10), 0)
        self.assertEqual(frozen, 0)
        self.assertEqual(payload['audio_keyframes'], main['audio_keyframes'])
        self.assertTrue(torch.equal(payload['cond_audio_rows'], main['cond_audio_rows']))

    def test_frozen_prefix_restored_after_renoise(self):
        prefix = torch.randn(1, 32, 2, 4)
        pipe = types.SimpleNamespace(_lc_source={'video': None}, _lc_audio_prefix=prefix)
        audio = torch.randn(1, 32, 2, 12)  # whole soundtrack re-noised by the refinement
        payload = {'cond_audio_rows': None}
        with contextlib.redirect_stdout(io.StringIO()):
            frozen = LatentMixin._lc_refinement(pipe, payload, {'lc_audio_offset': -39.0}, audio, 0)
        self.assertEqual(frozen, 4)
        self.assertEqual(payload['target_audio_condition_latents'], 4)
        self.assertEqual(payload['lc_audio_offset'], -39.0)
        self.assertTrue(torch.equal(audio[..., :4], prefix))

    def test_no_latent_join_keeps_native_refinement(self):
        payload = {'audio_keyframes': None, 'cond_audio_rows': None}
        before = dict(payload)
        self.assertEqual(LatentMixin._lc_refinement(types.SimpleNamespace(_lc_source=None), payload,
                                                    {'audio_keyframes': [{'anchor': 'first', 'latent_frame_count': 1}]}, None, 0), 0)
        self.assertEqual(payload, before)

    def test_unknown_in_place_rewrite_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'native_module.py'
            path.write_text('def generate(x):\n    y = x + 1\n    return y\n')
            spec = importlib.util.spec_from_file_location('native_module', path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            for rewritten, warned in ((False, False), (True, True)):
                if rewritten:  # another plugin swapped the body in place
                    env = {}
                    exec(compile('def generate(x):\n    y = x + 2\n    return y\n', str(path), 'exec'), env)
                    module.generate.__code__ = env['generate'].__code__
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    compile_edit(module.generate, [('    y = x + 1\n', '    y = x + 1\n    y = y * 10\n')], rewrites_for=lambda raw: [])
                self.assertEqual('WARNING: another plugin rewrote' in log.getvalue(), warned, log.getvalue())

    def test_edit_variants_first_unique_match(self):
        from h3refine.native_bridge import apply_edits, load_generate_edits
        source = 'def f():\n    a = 1\n    b = 2\n'
        # 18.19-style variant absent, 18.17-style variant present: the second one applies.
        self.assertEqual(apply_edits(source, [[('    a = 1 + t\n', 'x'), ('    a = 1\n', '    a = 3\n')]]), 'def f():\n    a = 3\n    b = 2\n')
        self.assertIsNone(apply_edits(source, [[('    c = 1\n', 'x')]]))
        self.assertIsNone(apply_edits(source + '    a = 1\n', [('    a = 1\n', 'x')]))  # ambiguous anchor refused
        self.assertTrue(any(isinstance(e, list) for e in load_generate_edits()))

    def test_refinement_settings_accepted(self):
        for value in ('none', 'enabled', 'disabled:6:0.5', 'enabled:1:0.1:audio_refs'):
            with self.subTest(value=value):
                preflight({'custom_settings': {'audio_refinement': value}}, {'save': True, 'continue': False, 'latent_path': ''})

    def test_audio_noise_option_alone_reaches_plain_renders(self):
        from h3refine import integration
        from h3refine.latent_runtime import JOB
        seen = []
        def original(task=None, model_type=None, plugin_data=None, **kwargs):
            seen.append(JOB.get())
            return True
        installed = []
        previous = integration.ensure_hooks
        integration.ensure_hooks = lambda: installed.append(1)
        try:
            wrapper = integration.make_generation_wrapper(original, lambda model: 'minimax_h3_ref2va_pruned')
            for flag, expected in ((True, True), (False, False)):
                seen.clear()
                data = {integration.KEY: {'save': False, 'continue': False, 'audio_noise_1818': flag}}
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(wrapper(task={}, model_type='minimax_h3_ref2va_pruned', plugin_data=data,
                                            image_start='start.png', sliding_window_discard_last_frames=5))
                job = seen[0]
                if expected:  # no preflight restriction (discard frames), no checkpoint required
                    self.assertTrue(job['noise_only'] and job['enabled'] and job['pending'] is None and job['part2'] is None)
                else:
                    self.assertIsNone(job)
            self.assertEqual(len(installed), 1)
        finally:
            integration.ensure_hooks = previous


if __name__ == '__main__':
    unittest.main()
