import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
    "requires cu130 environment",
)
class ResidentLatencyDiagnosticTests(unittest.TestCase):
    def test_rejects_empty_measurement_before_loading_weights(self):
        from diagnose_resident_latency import benchmark

        for options in (
            {"repeats": 0}, {"inner": 0}, {"warmup_seconds": -1},
            {"warmup_seconds": float("nan")}, {"configs": ()},
            {"configs": ((3, 2, 16),)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                benchmark(Path("missing"), **options)

    def test_mixed_q5_graph_matches_reference_and_counts_only_runtime_weights(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from compile_resident_residual_artifact import compile_layer
        from diagnose_resident_latency import benchmark
        from tests.gguf_fixture import write_fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "model.gguf", gate_up_types=("Q5_K", "Q4_K"), q5k_down=True)
            compile_layer(
                root / "model.gguf",
                0,
                None,
                root / "artifact",
                format_version=2,
            )
            report = benchmark(
                root / "artifact", repeats=3, inner=2, warmup_seconds=0.01,
                configs=((4, 2, 16), (8, 4, 8)),
                profile_stages=True,
            )
            self.assertEqual(report["reference_kind"], "gguf_dequantized_fp64")
            self.assertTrue(report["validation_passed"])
            self.assertIsNone(report["tokens_per_second"])
            self.assertEqual(report["runtime_h2d_bytes"], 0)
            # 256x256: split gate=57,344, up=49,152, original down=45,056.
            self.assertEqual(report["split_weight_bytes"], 151552)
            self.assertEqual(report["native_weight_bytes"], 126976)
            self.assertEqual(report["inner_invocations"], 2)
            self.assertEqual(len(report["execution_orders"]), 3)
            self.assertEqual(len(report["variants"]), 3)
            for result in report["variants"].values():
                self.assertEqual(len(result["samples_ms"]), 3)
                self.assertGreater(result["median_ms"], 0)
                self.assertLess(result["max_relative_l2"], 1e-4)
                self.assertLess(result["max_reference_relative_l2"], 1e-4)
                self.assertEqual(result["validation_inputs"], 2)
            profile = report["stage_profile"]
            self.assertTrue(profile["includes_event_nodes"])
            for result in profile["variants"].values():
                self.assertEqual(len(result["samples"]), 3)
                for sample in result["samples"]:
                    self.assertGreater(sample["gate_up_swiglu_ms"], 0)
                    self.assertGreater(sample["down_ms"], 0)
                    self.assertAlmostEqual(
                        sample["gate_up_swiglu_ms"] + sample["down_ms"],
                        sample["span_ms"],
                        delta=1e-4,
                    )

    def test_reference_ffn_matches_full_dequantization(self):
        import numpy as np
        from gguf.quants import dequantize
        from diagnose_resident_latency import _load_projection_inputs, _reference_ffn
        from tests.gguf_fixture import write_fixture

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            write_fixture(model, gate_up_types=("Q5_K", "Q4_K"), q5k_down=True)
            projections, down = _load_projection_inputs(model, 0)
            x = np.random.default_rng(46).standard_normal(256)
            g, u = [
                dequantize(projections[name].data, projections[name].tensor_type)
                .astype(np.float64) @ x
                for name in ("gate", "up")
            ]
            expected = dequantize(down.data, down.tensor_type).astype(np.float64) @ (
                g * np.exp(-np.logaddexp(0, -g)) * u
            )
            np.testing.assert_allclose(
                _reference_ffn(projections, down, x, chunk_rows=64),
                expected,
                rtol=1e-12,
                atol=1e-9,
            )

    def test_scale_reduction_comparison_measures_complete_ffn(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from compile_resident_residual_artifact import compile_layer
        from diagnose_resident_latency import benchmark
        from tests.gguf_fixture import write_fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "model.gguf", gate_up_types=("Q5_K", "Q5_K"), q5k_down=True)
            compile_layer(root / "model.gguf", 0, None, root / "artifact", format_version=2)
            report = benchmark(
                root / "artifact", repeats=3, inner=2, warmup_seconds=0.01,
                compare_scale_reduction=True, profile_stages=True,
            )
            self.assertTrue(report["validation_passed"])
            base = report["variants"]["split_r4_w2_g16"]
            candidate = report["variants"]["split_r4_w2_g16_scale_after"]
            self.assertEqual(base["weight_bytes"], candidate["weight_bytes"])
            self.assertGreater(candidate["kernel_resources"]["registers_per_thread"], 0)
            self.assertLess(candidate["max_reference_relative_l2"], 1e-4)
            self.assertTrue(candidate["config"]["scale_after_reduce"])
            self.assertEqual(len(candidate["samples_ms"]), 3)
            self.assertIn("split_r4_w2_g16_scale_after", report["stage_profile"]["variants"])

    def test_scale_reduction_comparison_rejects_pure_q4(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from compile_resident_residual_artifact import compile_layer
        from diagnose_resident_latency import benchmark
        from tests.gguf_fixture import write_fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "model.gguf", gate_up_types=("Q4_K", "Q4_K"), q4k_down=True)
            compile_layer(root / "model.gguf", 0, None, root / "artifact", format_version=2)
            with self.assertRaisesRegex(ValueError, "requires at least one Q5"):
                benchmark(
                    root / "artifact", repeats=1, inner=1, warmup_seconds=0.01,
                    compare_scale_reduction=True,
                )


if __name__ == "__main__":
    unittest.main()
