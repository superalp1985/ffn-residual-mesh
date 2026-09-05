from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "tile cache requires PyTorch")
class ResidentTileCacheTests(unittest.TestCase):
    def test_tile_cache_uploads_only_requested_row_tile(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_tile_cache import ResidentTileCache
        from resident_tile_plan import TilePlan

        arrays = {
            name: {
                "residual": np.full((130, 16), fill, dtype=np.uint8),
                "alpha": np.full((130, 1), 0.5, dtype=np.float32),
            }
            for name, fill in (("gate", 3), ("up", 7))
        }
        plan = TilePlan(rows=130, tile_rows=64, projections=("gate", "up"))
        with ResidentTileCache(arrays, plan=plan) as cache:
            first = cache.acquire(0)
            cache.release(0)
            second = cache.acquire(0)
            cache.release(0)
            self.assertFalse(first["resident_hit"])
            self.assertEqual(first["weight_h2d_bytes"], 2 * (64 * 16 + 64 * 4))
            self.assertTrue(second["resident_hit"])
            self.assertEqual(second["weight_h2d_bytes"], 0)
            self.assertEqual(cache.tile_range(2), (128, 130))
            self.assertEqual(cache.package(0)["gate.residual"].shape, (64, 16))

    def test_tile_prefetch_preserves_projection_slices(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_tile_cache import ResidentTileCache
        from resident_tile_plan import TilePlan

        arrays = {
            name: {
                "residual": np.arange(130 * 16, dtype=np.uint8).reshape(130, 16),
                "alpha": np.ones((130, 1), dtype=np.float32),
            }
            for name in ("gate", "up")
        }
        plan = TilePlan(rows=130, tile_rows=64, projections=("gate", "up"))
        with ResidentTileCache(arrays, plan=plan) as cache:
            cache.prefetch_async([2])
            with self.assertRaises(RuntimeError):
                cache.package(2)
            cache.wait_prefetch(2)
            np.testing.assert_array_equal(
                cache.package(2)["gate.residual"].cpu().numpy(),
                arrays["gate"]["residual"][128:130],
            )


if __name__ == "__main__":
    unittest.main()
