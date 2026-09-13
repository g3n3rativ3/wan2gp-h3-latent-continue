import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h3seamtest',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
pkg=importlib.util.module_from_spec(spec);sys.modules['h3seamtest']=pkg;spec.loader.exec_module(pkg)
from h3seamtest.seams import audio_window,shift_target_audio,seam_settings
from h3seamtest.packing import build_packed_sequence,build_ref2va_packed_sequence,MiniMaxH3PreparedReference,build_row_timesteps
from h3seamtest.integration import options,preflight,KEY


class SeamTests(unittest.TestCase):
    def test_audio_grid_and_origin_across_checkpoints(self):
        for fps in (24,25,30):
            for count in (107,124):
                for source_origin in (0.0,-0.975,-1.0166666666666666):
                    info={'target_frames':count,'fps':fps,'audio_origin_seconds':source_origin}
                    length=400
                    start,stop,origin=audio_window(info,length,1.0)
                    boundary=(count-1)/fps
                    self.assertAlmostEqual(origin,source_origin+start/40-boundary)
                    self.assertLessEqual(abs((stop-start)/40-1.0),0.05)
                    trim=round(-origin*32000)
                    self.assertLessEqual(abs(trim/32000+origin),0.5/32000+1e-10)
                    self.assertGreaterEqual(origin+(stop-start)/40,1/fps-1e-8)

    def test_shift_only_target_audio_for_both_layouts(self):
        for references in (False,True):
            args=[torch.zeros(3,dtype=torch.long)]
            if references:
                args.append([MiniMaxH3PreparedReference('audio',num_audio_latents=8)])
            args.extend([37,2,2,240,(1,2,2)])
            builder=build_ref2va_packed_sequence if references else build_packed_sequence
            layout=builder(*args,audio_condition_anchors=((-20.,10),),target_condition_audio_latents=40)
            before=layout.position_ids.clone()
            rows=layout.audio_indices[layout.num_condition_audio_rows:]
            shift_target_audio(layout,-39.0)
            expected=before.clone();expected[rows,0]-=39
            self.assertTrue(torch.equal(layout.position_ids,expected))
            self.assertEqual(layout.num_target_condition_audio_latents,40)
            levels,indices=build_row_timesteps(layout,0.5,0.4,0.999,1.0)
            times=levels[indices]
            self.assertTrue(torch.all(times[rows[:40]]==1))
            self.assertTrue(torch.all(times[rows[240:280]]==1))
            self.assertTrue(torch.all(times[rows[40:240]]==torch.tensor(0.4)))

    def test_modes_and_options_transport(self):
        payload={KEY:{'save':True,'continue':True,'join_mode':'audio_prefix',
                      'video_context_frames':52,'audio_context_seconds':2.0}}
        opts=options(payload)
        self.assertEqual(seam_settings(opts),('audio_prefix',52,2.0))
        with self.assertRaisesRegex(ValueError,'No Skipping'):
            preflight({'skip_steps_cache_type':'spectrum'},opts)
        with self.assertRaises(ValueError): seam_settings({'join_mode':'invalid'})
        with self.assertRaises(ValueError): seam_settings({'join_mode':'context','video_context_frames':999})


if __name__=='__main__': unittest.main()
