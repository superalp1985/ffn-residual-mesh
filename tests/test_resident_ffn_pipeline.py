import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "requires cu130 venv")
class ResidentFullFfnTests(unittest.TestCase):
    def test_full_ffn_report_validates_final_output_and_counts_workspace(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from benchmark_resident_ffn_pipeline import run_resident_ffn
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf", quantized_down=True)
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            report = run_resident_ffn(root / "artifact", repeats=3, cpu_threads=1)
            self.assertLess(report["output_rel_l2"], 1e-5)
            self.assertEqual(report["quality_scope"], "synthetic_inputs_not_model_quality")
            self.assertIsNone(report["tokens_per_second"])
            self.assertGreaterEqual(report["cuda_peak_allocated_bytes"],
                                    report["resident_payload_bytes"])
            self.assertEqual(report["dynamic_h2d_bytes"], 3072)
            self.assertEqual(report["residual_weight_h2d_bytes_per_token"], 0)
            self.assertEqual(len(report["samples"]), 3)

    def test_down_receives_merged_swiglu_and_reports_no_weight_uploads(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_residual_cuda import DirectIQ4NLProjection, ResidentGateUp
        rng = np.random.default_rng(5301)
        raw_down = rng.integers(0, 256, (256, 8, 18), dtype=np.uint8)
        raw_down[:, :, :2] = np.array([0.001], dtype="<f2").view(np.uint8)
        raw_down = raw_down.reshape(256, -1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                runner = ResidentGateUp(artifact)
                down = DirectIQ4NLProjection(raw_down, 256)
                x = rng.standard_normal(256).astype(np.float32)
                result = runner.run(x, down=down)
                gate = artifact.reconstruct_weights("gate").astype(np.float64) @ x
                up = artifact.reconstruct_weights("up").astype(np.float64) @ x
                h = gate / (1 + np.exp(-gate)) * up
                expected = dequantize(raw_down, GGMLQuantizationType.IQ4_NL).astype(np.float64) @ h
                np.testing.assert_allclose(result["down"], expected, atol=1e-4, rtol=2e-5)
                self.assertGreater(result["timing"]["down_stream_span_ms"], 0)
                self.assertEqual(result["dynamic_h2d_bytes"], 3072)


if __name__ == "__main__":
    unittest.main()
