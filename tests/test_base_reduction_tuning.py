from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "base reduction test requires the cu130 venv")
class BaseReductionTuningTests(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

    def test_prior_full_group_default_is_retained(self):
        from resident_residual_cuda import launch_fused_gate_up_base_residual
        from resident_tiled_ffn import TiledResidentGateUp

        self.assertEqual(
            inspect.signature(launch_fused_gate_up_base_residual).parameters["block_groups"].default,
            256,
        )
        self.assertEqual(
            inspect.signature(TiledResidentGateUp).parameters["base_block_groups"].default,
            256,
        )

    def test_base_group_tiles_preserve_all_rows_and_groups(self):
        import torch
        from resident_residual_cuda import launch_fused_gate_up_base_residual

        rng = np.random.default_rng(6801)
        for cols in (1280, 5120):
            rows, groups = 13, cols // 32
            x = rng.standard_normal(cols).astype(np.float32)
            host_sums = x.astype(np.float64).reshape(groups, 32).sum(axis=1)
            device_x = torch.from_numpy(x).cuda()
            group_sums = device_x.view(-1, 32).sum(dim=1)
            operands, expected = {}, {}
            for name in ("gate", "up"):
                signed = rng.integers(-8, 8, (rows, cols), dtype=np.int16)
                packed = ((signed[:, 0::2] & 15) | ((signed[:, 1::2] & 15) << 4)).astype(np.uint8)
                alpha = (rng.standard_normal((rows, groups)) * 0.01).astype(np.float32)
                coefficient = (rng.standard_normal((rows, groups)) * 0.02).astype(np.float32)
                operands[name] = [
                    torch.from_numpy(value).cuda() for value in (packed, alpha, coefficient)
                ]
                weights = signed.astype(np.float64) * np.repeat(alpha, 32, axis=1)
                expected[name] = weights @ x + coefficient.astype(np.float64) @ host_sums
            expected["swiglu"] = (
                expected["gate"] * np.exp(-np.logaddexp(0, -expected["gate"])) * expected["up"]
            )
            outputs = [torch.empty(rows, device="cuda") for _ in range(3)]
            for block_groups in (8, 16, 32, 64, 128, 256):
                with self.subTest(cols=cols, block_groups=block_groups):
                    compiled = launch_fused_gate_up_base_residual(
                        operands["gate"][0], operands["gate"][1],
                        operands["up"][0], operands["up"][1],
                        operands["gate"][2], operands["up"][2],
                        group_sums, device_x, *outputs,
                        rows=rows, cols=cols, block_rows=4, num_warps=4,
                        block_groups=block_groups,
                    )
                    for name, output in zip(("gate", "up", "swiglu"), outputs):
                        np.testing.assert_allclose(
                            output.cpu().numpy(), expected[name], rtol=3e-5, atol=3e-5,
                        )
                    self.assertIsNotNone(compiled, "resource diagnostics require the compiled kernel")
                    self.assertGreater(compiled.n_regs, 0)
                    self.assertGreaterEqual(compiled.n_spills, 0)

    def test_sweep_measures_complete_ffn_and_keeps_samples(self):
        import sweep_base_reduction as sweep
        from compile_resident_residual_artifact import compile_layer
        from tests.gguf_fixture import write_fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "model.gguf", q4k_down=True)
            compile_layer(root / "model.gguf", 0, 4, root / "artifact")
            report = sweep.benchmark(
                root / "artifact", candidates=(8, 32, 256), repeats=3,
                inner=2, warmup_seconds=0.01,
            )
            self.assertEqual(report["baseline_block_groups"], 256)
            self.assertEqual(report["dynamic_h2d_bytes_per_graph_ffn"], 0)
            self.assertIsNone(report["tokens_per_second"])
            self.assertEqual(len(report["execution_order"]), 3)
            for order in report["execution_order"]:
                self.assertEqual(sorted(order), [8, 32, 256])
            self.assertEqual(len(report["variants"]), 3)
            self.assertGreaterEqual(report["warmup_elapsed_seconds"], 0.01)
            self.assertGreater(report["warmup_cycles"], 0)
            self.assertEqual(report["down_projection"]["quant_type"], "Q4_K")
            self.assertGreater(report["resident_payload_bytes"], 0)
            for variant in report["variants"]:
                self.assertLess(variant["output_rel_l2"], 1e-4)
                self.assertEqual(len(variant["samples"]), 9)
                self.assertEqual(len(variant["passes"]), 3)
                self.assertGreater(variant["registers_per_thread"], 0)
                self.assertGreater(variant["ffn_graph_ms_median"], 0)
                self.assertGreater(variant["kernel_graph_ms_median"], 0)
                self.assertTrue(all(
                    sample["ffn_graph_ms"] > 0 and sample["kernel_graph_ms"] > 0
                    for sample in variant["samples"]
                ))
                for sample in variant["samples"]:
                    self.assertIn(sample["pass_index"], (0, 1, 2))
                    self.assertIn(sample["round_index"], (0, 1, 2))
                    self.assertEqual(len(sample["ffn_graph_raw_ms"]), 2)
                    self.assertEqual(len(sample["kernel_graph_raw_ms"]), 2)
                self.assertEqual(
                    variant["ffn_graph_ms_median"],
                    float(np.median([s["ffn_graph_ms"] for s in variant["samples"]])),
                )

    def test_sweep_rejects_invalid_measurement_protocol(self):
        import sweep_base_reduction as sweep

        for kwargs in (
            {"repeats": 0}, {"inner": 0}, {"warmup_seconds": -1},
            {"warmup_seconds": float("nan")}, {"candidates": ()},
            {"candidates": (8, 8, 256)}, {"candidates": (32, 63, 256)},
            {"candidates": (8, 32)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sweep.benchmark(Path("does-not-exist"), **kwargs)


if __name__ == "__main__":
    unittest.main()
