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
                     "CUDA test requires the cu130 venv")
class ResidentCudaTests(unittest.TestCase):
    def test_iq4nl_down_uses_original_nonlinear_quantizer(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from resident_residual_cuda import DirectIQ4NLProjection
        rng = np.random.default_rng(530)
        raw = rng.integers(0, 256, (37, 24, 18), dtype=np.uint8)
        raw[:, :, :2] = np.array([0.003], dtype="<f2").view(np.uint8)
        raw = raw.reshape(37, -1)
        baseline = DirectIQ4NLProjection(raw, 768, chunk_cols=384)
        x = rng.standard_normal(768).astype(np.float32)
        baseline.launch(torch.from_numpy(x).cuda())
        expected = dequantize(raw, GGMLQuantizationType.IQ4_NL).astype(np.float64) @ x
        np.testing.assert_allclose(baseline.output.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_raw_q4k_baseline_decodes_all_header_bits(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        from tests.gguf_fixture import write_fixture
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from resident_residual_cuda import DirectQ4Projection
        with tempfile.TemporaryDirectory() as directory:
            raw = write_fixture(Path(directory) / "fixture.gguf")["gate"]
            for kernel, block_qblocks in (("legacy", 1), ("grouped", 1), ("grouped", 2)):
                baseline = DirectQ4Projection(raw, 256, kernel=kernel, block_qblocks=block_qblocks)
                for seed in (51, 52):
                    x = np.random.default_rng(seed).standard_normal(256).astype(np.float32)
                    device_x = torch.from_numpy(x).cuda()
                    baseline.launch(device_x)
                    actual = baseline.output.cpu().numpy()
                    expected = dequantize(raw, GGMLQuantizationType.Q4_K).astype(np.float64) @ x
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_q4k_k_tiling_matches_reference_for_multiple_chunk_sizes(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from resident_residual_cuda import DirectQ4Projection

        rng = np.random.default_rng(553)
        raw = rng.integers(0, 256, (23, 288), dtype=np.uint8)
        raw[:, 0:2] = np.array([0.002], dtype="<f2").view(np.uint8)
        raw[:, 2:4] = np.array([0.001], dtype="<f2").view(np.uint8)
        raw[:, 144:146] = np.array([0.003], dtype="<f2").view(np.uint8)
        raw[:, 146:148] = np.array([0.0015], dtype="<f2").view(np.uint8)
        x = rng.standard_normal(512).astype(np.float32)
        expected = dequantize(raw, GGMLQuantizationType.Q4_K).astype(np.float64) @ x
        device_x = torch.from_numpy(x).cuda()
        for kernel, block_qblocks in (("legacy", 1), ("grouped", 1), ("grouped", 2)):
            for chunk_cols in (256, 512):
                projection = DirectQ4Projection(
                    raw,
                    512,
                    chunk_cols=chunk_cols,
                    kernel=kernel,
                    block_qblocks=block_qblocks,
                )
                projection.launch(device_x)
                np.testing.assert_allclose(
                    projection.output.cpu().numpy(),
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_q5k_down_matches_reference_including_partial_tiles(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from resident_residual_cuda import DirectQ5Projection

        rng = np.random.default_rng(554)
        for cols in (256, 512, 1280):
            raw = rng.integers(0, 256, (23, cols // 256, 176), dtype=np.uint8)
            raw[:, :, :2] = np.array([0.002], dtype="<f2").view(np.uint8)
            raw[:, :, 2:4] = np.array([0.001], dtype="<f2").view(np.uint8)
            raw = raw.reshape(23, -1)
            x = rng.standard_normal(cols).astype(np.float32)
            expected = dequantize(raw, GGMLQuantizationType.Q5_K).astype(np.float64) @ x
            for qblocks in (1, 2, 4):
                with self.subTest(cols=cols, qblocks=qblocks):
                    projection = DirectQ5Projection(
                        raw, cols, block_rows=2, num_warps=2, block_qblocks=qblocks,
                    )
                    projection.launch(torch.from_numpy(x).cuda())
                    np.testing.assert_allclose(
                        projection.output.cpu().numpy(), expected, rtol=2e-5, atol=2e-5,
                    )

    def test_iq4xs_down_matches_reference_including_partial_tiles(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from resident_residual_cuda import DirectIQ4XSProjection

        rng = np.random.default_rng(555)
        for cols in (256, 512, 1280):
            raw = rng.integers(0, 256, (23, cols // 256, 136), dtype=np.uint8)
            raw[:, :, :2] = np.array([0.002], dtype="<f2").view(np.uint8)
            raw = raw.reshape(23, -1)
            x = rng.standard_normal(cols).astype(np.float32)
            expected = dequantize(raw, GGMLQuantizationType.IQ4_XS).astype(np.float64) @ x
            for qblocks in (1, 2, 4):
                with self.subTest(cols=cols, qblocks=qblocks):
                    projection = DirectIQ4XSProjection(
                        raw, cols, block_rows=2, num_warps=2, block_qblocks=qblocks,
                    )
                    projection.launch(torch.from_numpy(x).cuda())
                    np.testing.assert_allclose(
                        projection.output.cpu().numpy(), expected, rtol=2e-5, atol=2e-5,
                    )

    def test_q4k_rejects_invalid_chunk_before_gpu_upload(self):
        from resident_residual_cuda import DirectQ4Projection

        raw = np.zeros((1, 144), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "positive multiple of 256"):
            DirectQ4Projection(raw, 256, chunk_cols=128)

    def test_residuals_upload_once_and_new_inputs_recompute_base_and_swiglu(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_residual_cuda import ResidentGateUp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                for residual_kernel, block_rows in (
                    ("legacy", 1),
                    ("grouped", 1),
                    ("grouped", 8),
                    ("grouped_fused", 8),
                ):
                    runner = ResidentGateUp(
                        artifact,
                        block_rows=block_rows,
                        num_warps=4,
                        residual_kernel=residual_kernel,
                    )
                    initial = dict(runner.traffic)
                    for seed in (7, 17):
                        x = np.random.default_rng(seed).standard_normal(256).astype(np.float32)
                        result = runner.run(x)
                        expected = {p: artifact.reconstruct_weights(p).astype(np.float64) @ x
                                    for p in ("gate", "up")}
                        for p in ("gate", "up"):
                            np.testing.assert_allclose(result[p], expected[p], atol=1e-5, rtol=1e-5)
                        swiglu = expected["gate"] / (1 + np.exp(-expected["gate"])) * expected["up"]
                        np.testing.assert_allclose(result["swiglu"], swiglu, atol=1e-4, rtol=1e-5)
                    self.assertEqual(runner.traffic["weight_upload_bytes"], initial["weight_upload_bytes"])
                    self.assertEqual(runner.traffic["dynamic_h2d_bytes"], 2 * (256 + 512) * 4)
                    self.assertEqual(runner.resident_bytes, artifact.gate_up_bytes())
                    self.assertEqual(runner.traffic["weight_h2d_bytes_per_run"], 0)
                    expected_kernel_count = 1 if residual_kernel == "grouped_fused" else 2
                    self.assertEqual(len(runner.kernel_resources), expected_kernel_count)
                    self.assertIn("registers_per_thread", runner.kernel_resources[0])
                    self.assertIn("spills", runner.kernel_resources[0])


if __name__ == "__main__":
    unittest.main()
