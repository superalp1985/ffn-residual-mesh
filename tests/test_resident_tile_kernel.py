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

    def test_fused_gate_up_tile_adds_base_and_produces_swiglu(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_residual_cuda import launch_fused_gate_up_tile

        rng = np.random.default_rng(551)
        rows, cols = 31, 256
        packed = {
            name: rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
            for name in ("gate", "up")
        }
        alpha = {
            name: rng.standard_normal((rows, cols // 32)).astype(np.float32)
            for name in ("gate", "up")
        }
        base = {
            name: rng.standard_normal(rows).astype(np.float32)
            for name in ("gate", "up")
        }
        x = rng.standard_normal(cols).astype(np.float32)

        def residual(name: str) -> np.ndarray:
            codes = np.empty((rows, cols), dtype=np.float32)
            for shift, offset in ((0, 0), (4, 1)):
                nibble = ((packed[name] >> shift) & 15).astype(np.int16)
                codes[:, offset::2] = np.where(nibble >= 8, nibble - 16, nibble)
            return (
                codes.reshape(rows, -1, 32)
                * alpha[name][:, :, None]
                * x.reshape(-1, 32)[None, :, :]
            ).sum(axis=(1, 2))

        expected_gate = residual("gate") + base["gate"]
        expected_up = residual("up") + base["up"]
        sigmoid = np.empty_like(expected_gate)
        positive = expected_gate >= 0
        sigmoid[positive] = 1 / (1 + np.exp(-expected_gate[positive]))
        exp_gate = np.exp(expected_gate[~positive])
        sigmoid[~positive] = exp_gate / (1 + exp_gate)
        expected_swiglu = expected_gate * sigmoid * expected_up
        outputs = {
            name: torch.empty(rows, device="cuda", dtype=torch.float32)
            for name in ("gate", "up", "swiglu")
        }
        launch_fused_gate_up_tile(
            torch.from_numpy(packed["gate"]).cuda(),
            torch.from_numpy(alpha["gate"]).cuda(),
            torch.from_numpy(packed["up"]).cuda(),
            torch.from_numpy(alpha["up"]).cuda(),
            torch.from_numpy(x).cuda(),
            torch.from_numpy(base["gate"]).cuda(),
            torch.from_numpy(base["up"]).cuda(),
            outputs["gate"],
            outputs["up"],
            outputs["swiglu"],
            rows=rows,
            cols=cols,
        )
        torch.cuda.synchronize()
        np.testing.assert_allclose(outputs["gate"].cpu(), expected_gate, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(outputs["up"].cpu(), expected_up, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(outputs["swiglu"].cpu(), expected_swiglu, rtol=3e-5, atol=1e-5)

    def test_fused_gate_up_residual_tile_matches_two_residual_dots(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_residual_cuda import launch_fused_gate_up_residual_tile

        rng = np.random.default_rng(552)
        rows, cols = 33, 256
        packed = {
            name: rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
            for name in ("gate", "up")
        }
        alpha = {
            name: rng.standard_normal((rows, cols // 32)).astype(np.float32)
            for name in ("gate", "up")
        }
        x = rng.standard_normal(cols).astype(np.float32)
        outputs = {
            name: torch.empty(rows, device="cuda", dtype=torch.float32)
            for name in ("gate", "up")
        }
        launch_fused_gate_up_residual_tile(
            torch.from_numpy(packed["gate"]).cuda(),
            torch.from_numpy(alpha["gate"]).cuda(),
            torch.from_numpy(packed["up"]).cuda(),
            torch.from_numpy(alpha["up"]).cuda(),
            torch.from_numpy(x).cuda(),
            outputs["gate"],
            outputs["up"],
            rows=rows,
            cols=cols,
        )
        torch.cuda.synchronize()
        for name in ("gate", "up"):
            codes = np.empty((rows, cols), dtype=np.float32)
            for shift, offset in ((0, 0), (4, 1)):
                nibble = ((packed[name] >> shift) & 15).astype(np.int16)
                codes[:, offset::2] = np.where(nibble >= 8, nibble - 16, nibble)
            expected = (
                codes.reshape(rows, -1, 32)
                * alpha[name][:, :, None]
                * x.reshape(-1, 32)[None, :, :]
            ).sum(axis=(1, 2))
            np.testing.assert_allclose(outputs[name].cpu(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
