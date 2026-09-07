from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from compile_resident_residual_artifact import compile_layer
from resident_residual_format import ResidentArtifact
from tests.gguf_fixture import write_fixture


class AffineV2ArtifactTests(unittest.TestCase):
    def test_q5_and_mixed_affine_roundtrip_without_source(self):
        for types in (("Q5_K", "Q5_K"), ("Q4_K", "Q5_K"), ("Q5_K", "Q4_K")):
            with self.subTest(types=types), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "fixture.gguf"
                raw = write_fixture(source, gate_up_types=types)
                result = compile_layer(source, 0, None, root / "artifact", format_version=2,
                                       chunk_rows=17)
                source.unlink()
                with ResidentArtifact.open(Path(result["path"]), verify_hashes=True) as artifact:
                    self.assertFalse(artifact.manifest["runtime_requires_table_lookup"])
                    expected_resident = 0
                    for name, type_name in zip(("gate", "up"), types):
                        bits = 5 if type_name == "Q5_K" else 4
                        self.assertEqual(artifact.residual_bits(name), bits)
                        expected = dequantize(raw[name], GGMLQuantizationType[type_name])
                        np.testing.assert_array_equal(artifact.reconstruct_weights(name), expected)
                        self.assertEqual(artifact.arrays[name]["residual"].nbytes, 256 * 256 * bits // 8)
                        residual = artifact.unpack_residual(name)
                        self.assertGreaterEqual(int(residual.min()), -(1 << (bits - 1)))
                        self.assertLessEqual(int(residual.max()), (1 << (bits - 1)) - 1)
                        expected_resident += 256 * 256 * bits // 8 + 256 * 8 * 4
                        for seed in (17, 77):
                            x = np.random.default_rng(seed).standard_normal(256).astype(np.float32)
                            base, tail = artifact.project_parts(name, x)
                            np.testing.assert_allclose(base + tail, expected.astype(np.float64) @ x,
                                                       atol=2e-5, rtol=2e-5)
                    self.assertEqual(artifact.gate_up_bytes(), expected_resident)
                    self.assertEqual(result["byte_ledger"]["resident_gate_up_bytes"], expected_resident)

    def test_q5_decode_preserves_high_planes_and_multiple_blocks(self):
        import compile_resident_residual_artifact as compiler

        self.assertTrue(hasattr(compiler, "decode_q5k"), "missing exact Q5_K decoder")
        raw = np.zeros((3, 2, 176), dtype=np.uint8)
        raw[:, :, :2] = np.array([1], dtype="<f2").view(np.uint8)
        raw[:, :, 4:16] = 1
        # All q values 0..31 in each group, across every high-bit plane.
        raw[:, :, 16:48] = np.where(np.arange(32) >= 16, 255, 0).astype(np.uint8)
        raw[:, :, 48:] = np.tile((np.arange(32, dtype=np.uint8) % 16) * 17, 4)
        codes, alpha, beta = compiler.decode_q5k(raw.reshape(3, -1))
        np.testing.assert_array_equal(codes, np.tile(np.arange(32), (3, 16)))
        actual = (alpha[:, :, None] * codes.reshape(3, 16, 32).astype(np.float32)
                  + beta[:, :, None]).reshape(3, -1)
        np.testing.assert_array_equal(
            actual, dequantize(raw.reshape(3, -1), GGMLQuantizationType.Q5_K),
        )

    def test_v2_rejects_wrong_bitwidth_and_packing_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.gguf"
            write_fixture(source, gate_up_types=("Q5_K", "Q5_K"))
            compile_layer(source, 0, None, root / "artifact", format_version=2)
            path = root / "artifact" / "manifest.json"
            manifest = json.loads(path.read_text())
            for field, value in (("residual_bits", 4), ("packing", "signed_nibble_pairs_v1")):
                changed = json.loads(json.dumps(manifest))
                changed["projections"]["gate"][field] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    ResidentArtifact.open(path)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "CUDA test requires the cu130 venv")
class AffineV2CudaTests(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")

    def test_q5_residual_all_signed_values_and_partial_row_group_tiles(self):
        import torch
        from resident_residual_cuda import _launch_residual_dot

        rows, cols = 23, 1280
        signed = np.tile(np.arange(-16, 16, dtype=np.int16), (rows, cols // 32))
        low = ((signed[:, 0::2] & 15) | ((signed[:, 1::2] & 15) << 4)).astype(np.uint8)
        high = np.packbits(((signed & 31) >> 4).astype(np.uint8).reshape(rows, -1, 32),
                           axis=-1, bitorder="little")
        packed = np.concatenate((low.reshape(rows, -1, 16), high), axis=-1).reshape(rows, -1)
        rng = np.random.default_rng(671)
        alpha = rng.standard_normal((rows, cols // 32)).astype(np.float32)
        x = rng.standard_normal(cols).astype(np.float32)
        output = torch.empty(rows, device="cuda")
        _launch_residual_dot(
            torch.from_numpy(packed).cuda(), torch.from_numpy(alpha).cuda(),
            torch.from_numpy(x).cuda(), output, rows=rows, cols=cols,
            block_rows=8, num_warps=4, kernel="grouped", block_groups=16, bits=5,
        )
        expected = (signed.astype(np.float64) * np.repeat(alpha, 32, axis=1)) @ x
        np.testing.assert_allclose(output.cpu().numpy(), expected, atol=2e-4, rtol=2e-5)

    def test_q5_mixed_resident_merge_matches_original_weights_and_uploads_once(self):
        from resident_residual_cuda import ResidentGateUp

        for types in (("Q5_K", "Q5_K"), ("Q4_K", "Q5_K"), ("Q5_K", "Q4_K")):
            with self.subTest(types=types), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw = write_fixture(root / "fixture.gguf", gate_up_types=types)
                compile_layer(root / "fixture.gguf", 0, None, root / "artifact", format_version=2)
                with ResidentArtifact.open(root / "artifact") as artifact:
                    for kernel in ("grouped", "grouped_fused"):
                        runner = ResidentGateUp(artifact, residual_kernel=kernel,
                                                block_rows=8, residual_block_groups=16)
                        initial = runner.traffic["weight_upload_bytes"]
                        for seed in (67, 68):
                            x = np.random.default_rng(seed).standard_normal(256).astype(np.float32)
                            actual = runner.run(x)
                            reference = {
                                name: dequantize(raw[name], GGMLQuantizationType[type_name]).astype(np.float64) @ x
                                for name, type_name in zip(("gate", "up"), types)
                            }
                            for name in ("gate", "up"):
                                np.testing.assert_allclose(actual[name], reference[name], atol=3e-5, rtol=2e-5)
                            g, u = reference["gate"], reference["up"]
                            expected = g * np.exp(-np.logaddexp(0, -g)) * u
                            np.testing.assert_allclose(actual["swiglu"], expected, atol=2e-4, rtol=3e-5)
                        self.assertEqual(runner.resident_bytes, artifact.gate_up_bytes())
                        self.assertEqual(runner.traffic["weight_upload_bytes"], initial)
                        self.assertEqual(runner.traffic["dynamic_h2d_bytes"], 2 * (256 + 512) * 4)
                        self.assertEqual(runner.traffic["weight_h2d_bytes_per_run"], 0)

    def test_q5_is_rejected_by_q4_only_execution_paths(self):
        from resident_residual_cuda import ResidentGateUp
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf", gate_up_types=("Q5_K", "Q4_K"))
            compile_layer(root / "fixture.gguf", 0, None, root / "artifact", format_version=2)
            with ResidentArtifact.open(root / "artifact") as artifact:
                with self.assertRaisesRegex(ValueError, "grouped"):
                    ResidentGateUp(artifact, residual_kernel="legacy")
                with self.assertRaisesRegex(ValueError, "Q4"):
                    TiledResidentGateUp(artifact)

    def test_full_ffn_with_q5_and_mixed_gate_up(self):
        from benchmark_resident_ffn_pipeline import run_resident_ffn

        for types in (("Q5_K", "Q5_K"), ("Q5_K", "Q4_K")):
            with self.subTest(types=types), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_fixture(root / "fixture.gguf", gate_up_types=types, q5k_down=True)
                compile_layer(root / "fixture.gguf", 0, None, root / "artifact", format_version=2)
                report = run_resident_ffn(
                    root / "artifact", repeats=3, cpu_threads=2, residual_kernel="grouped_fused",
                    block_rows=8, residual_block_groups=16, warmup_seconds=0.05,
                )
                self.assertLess(report["output_rel_l2"], 1e-5)
                self.assertEqual(report["residual_weight_h2d_bytes_measured_delta"], 0)
                self.assertEqual(report["down_quant_type"], "Q5_K")
                self.assertEqual(report["gate_up_quant_types"], dict(zip(("gate", "up"), types)))
                self.assertIsNone(report["tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
