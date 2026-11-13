"""CPU tests for the SS64 completion helper, without importing ComfyUI."""
import importlib.util
import unittest
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / "trellis2/pipelines/ss64_t5_capture.py"
spec = importlib.util.spec_from_file_location("ss64_t5_capture_test", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestCompletion(unittest.TestCase):
    def test_capture_and_order(self):
        latent = torch.zeros((1, 8, 16, 16, 16))
        logits = torch.full((1, 1, 64, 64, 64), -120.0)
        # Three isolated missing cells with two opposite pairs.
        for x, logit in [(8, -40.0), (20, -41.0), (32, -90.0)]:
            logits[0, 0, x, 10, 10] = logit
            for y, z in [(9, 10), (11, 10), (10, 9), (10, 11)]:
                logits[0, 0, x, y, z] = 20
        # A T5-only addition and exact threshold exclusion.
        logits[0, 0, 50, 50, 50] = -1
        logits[0, 0, 51, 50, 50] = -5
        with mod.request_capture(-5, surface_all=True) as r:
            base = mod.capture_logits(r, latent, logits)
            self.assertEqual(len(base), 12)
            self.assertEqual(len(r["t5_coords"]), 13)
            self.assertEqual(r["surface_added_coords"].tolist(),
                             [[0,8,10,10], [0,20,10,10], [0,32,10,10]])
            self.assertTrue(torch.equal(r["surface_all_coords"][:13], r["t5_coords"]))
            self.assertEqual(len(r["surface_all_coords"]), 16)
        self.assertIsNone(mod.active_request())

    def test_single_pass(self):
        logits = torch.full((1, 1, 64, 64, 64), -80.0)
        # A gets filled. B would qualify only after A is added.
        for p in [(9,10,10), (11,10,10), (10,9,10), (10,11,10),
                  (12,9,10), (12,11,10), (13,10,10)]:
            logits[(0,0,*p)] = 20
        # B=(12,10,10) already has two pairs; instead verify no cascade
        # with adjacent X holes and Y pairs: A support along Z.
        logits.fill_(-80)
        for p in [(10,9,10),(10,11,10),(10,10,9),(10,10,11),
                  (11,9,10),(11,11,10),(12,10,10)]:
            logits[(0,0,*p)] = 20
        with mod.request_capture(-5, surface_all=True) as r:
            mod.capture_logits(r, torch.zeros((1,8,16,16,16)), logits)
            extra = set(map(tuple, r["surface_added_coords"].tolist()))
            self.assertIn((0,10,10,10), extra)
            self.assertNotIn((0,11,10,10), extra)

    def test_t5_only_and_context_cleanup(self):
        logits = torch.full((1,1,64,64,64), -10.0)
        logits[0,0,1,1,1] = 1
        with mod.request_capture(-5) as r:
            mod.capture_logits(r, torch.zeros((1,8,16,16,16)), logits)
            self.assertNotIn("surface_all_coords", r)
        self.assertIsNone(mod.active_request())


if __name__ == "__main__":
    unittest.main()
