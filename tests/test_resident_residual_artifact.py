from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from compile_resident_residual_artifact import compile_layer  # noqa: E402
from resident_residual_format import ResidentArtifact  # noqa: E402
from tests.gguf_fixture import write_fixture  # noqa: E402


MODEL = Path(os.environ.get("QWEN38_27B_GGUF", r"E:\Qwen3.8-27B\Qwen3.8-27B-UD-Q4_K_M.gguf"))


class ResidentResidualArtifactTests(unittest.TestCase):
    def test_missing_projection_does_not_leave_source_file_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.gguf"
            write_fixture(source, missing_up=True)
            with self.assertRaisesRegex(ValueError, "missing"):
                compile_layer(source, 0, 4, root / "artifact")
            source.rename(root / "still_accessible.gguf")

    def test_roundtrip_is_independent_of_source_and_matches_upstream_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.gguf"
            raw = write_fixture(source)
            result = compile_layer(source, layer=0, bits=4, out_dir=root / "artifact", chunk_rows=17)
            with ResidentArtifact.open(Path(result["path"]), verify_hashes=True) as artifact:
                self.assertFalse(artifact.manifest["runtime_requires_table_lookup"])
                self.assertEqual(artifact.fallbacks["down"]["type_name"], "F32")
                rng = np.random.default_rng(42)
                for projection in ("gate", "up"):
                    expected = dequantize(raw[projection], GGMLQuantizationType.Q4_K)
                    actual = artifact.reconstruct_weights(projection)
                    np.testing.assert_array_equal(actual, expected)
                    expected_codes = np.empty((256, 256), dtype=np.uint8)
                    for group in range(8):
                        payload = raw[projection][:, 16 + (group // 2) * 32:
                                                    16 + (group // 2 + 1) * 32]
                        expected_codes[:, group * 32:(group + 1) * 32] = (
                            payload >> (4 * (group % 2))
                        ) & 15
                    np.testing.assert_array_equal(artifact.reconstruct_codes(projection), expected_codes)
                    x = rng.standard_normal(256).astype(np.float32)
                    base, residual = artifact.project_parts(projection, x)
                    np.testing.assert_allclose(base + residual, expected.astype(np.float64) @ x,
                                               atol=1e-5, rtol=1e-5)
                ledger = artifact.manifest["byte_ledger"]
                self.assertEqual(ledger["resident_gate_up_bytes"], 2 * (256 * 128 + 256 * 8 * 4))
                self.assertEqual(ledger["residual_code_bytes"], 256 * 256)
                self.assertGreater(ledger["host_base_bytes"], 0)

    def test_corrupt_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.gguf"
            write_fixture(source)
            result = compile_layer(source, layer=0, bits=4, out_dir=root / "artifact")
            manifest = json.loads(Path(result["path"]).read_text())
            payload = root / "artifact" / manifest["projections"]["gate"]["files"]["residual"]["file"]
            with payload.open("r+b") as stream:
                stream.write(b"\xff" * 8)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                ResidentArtifact.open(Path(result["path"]), verify_hashes=True)

    def test_exact_mode_rejects_unsupported_bitwidth_and_reused_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.gguf"
            write_fixture(source)
            with self.assertRaisesRegex(ValueError, "4"):
                compile_layer(source, layer=0, bits=2, out_dir=root / "bad")
            compile_layer(source, layer=0, bits=4, out_dir=root / "artifact")
            with self.assertRaises(FileExistsError):
                compile_layer(source, layer=0, bits=4, out_dir=root / "artifact")

    @unittest.skipUnless(os.environ.get("RUN_QWEN38_INTEGRATION") == "1" and MODEL.exists(),
                         "set RUN_QWEN38_INTEGRATION=1 for the real model")
    def test_q4k_projection_roundtrips_codes_without_runtime_table(self) -> None:
        with tempfile.TemporaryDirectory(prefix="resident-artifact-") as directory:
            result = compile_layer(MODEL, layer=3, bits=4, out_dir=Path(directory) / "artifact")
            reader = GGUFReader(MODEL)
            by_name = {item.name: item for item in reader.tensors}
            with ResidentArtifact.open(Path(result["path"]), verify_hashes=True) as loaded:
                self.assertIn("down", loaded.fallbacks)
                for projection in ("gate", "up"):
                    tensor = by_name[f"blk.3.ffn_{projection}.weight"]
                    for start in range(0, int(tensor.shape[1]), 128):
                        stop = min(start + 128, int(tensor.shape[1]))
                        expected = dequantize(tensor.data[start:stop], tensor.tensor_type)
                        np.testing.assert_array_equal(
                            loaded.reconstruct_weights(projection, start, stop), expected,
                        )
            reader.data._mmap.close()


if __name__ == "__main__":
    unittest.main()
