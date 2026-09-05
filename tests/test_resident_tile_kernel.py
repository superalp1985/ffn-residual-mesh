from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "tile kernel test requires the cu130 venv")
class ResidentTileKernelTests(unittest.TestCase):
    def test_tile_kernel_matches_numpy_residual_dot(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_residual_cuda import launch_residual_tile

        rng = np.random.default_rng(550)
        rows, cols = 37, 256
        packed = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
        alpha = rng.standard_normal((rows, cols // 32)).astype(np.float32)
        x = rng.standard_normal(cols).astype(np.float32)
        expected_codes = np.empty((rows, cols), dtype=np.float32)
        for shift, offset in ((0, 0), (4, 1)):
            nibble = ((packed >> shift) & 15).astype(np.int16)
            expected_codes[:, offset::2] = np.where(nibble >= 8, nibble - 16, nibble)
        expected = (expected_codes.reshape(rows, -1, 32) *
                    alpha[:, :, None] * x.reshape(-1, 32)[None, :, :]).sum(axis=(1, 2))
        output = torch.empty(rows, device="cuda", dtype=torch.float32)
        launch_residual_tile(
            torch.from_numpy(packed).cuda(),
            torch.from_numpy(alpha).cuda(),
            torch.from_numpy(x).cuda(),
            output,
            rows=rows,
            cols=cols,
        )
        torch.cuda.synchronize()
        np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
