from __future__ import annotations

import json
import os
import sys
import hashlib
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_qwen38_manifest import build_manifest  # noqa: E402
from tests.gguf_fixture import write_fixture  # noqa: E402


MODEL = Path(os.environ.get("QWEN38_27B_GGUF", r"E:\Qwen3.8-27B\Qwen3.8-27B-UD-Q4_K_M.gguf"))
CONFIG = MODEL.with_name("config.json")


class ManifestFixtureTests(unittest.TestCase):
    def test_mtp_metadata_is_subtracted_only_from_main_layer_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.gguf"
            write_fixture(path, mtp_layers=1)
            manifest = build_manifest(path, expected_dimensions={"hidden": 256, "ffn": 256, "layers": 1})
            self.assertEqual(manifest["metadata"]["gguf_total_blocks"], 2)
            self.assertEqual(manifest["metadata"]["mtp_layers"], 1)
            self.assertEqual(manifest["dimensions"]["layers"], 1)

    def test_gguf_metadata_works_without_external_config_and_records_real_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.gguf"
            write_fixture(path)
            manifest = build_manifest(path, expected_dimensions={"hidden": 256, "ffn": 256, "layers": 1})
            self.assertEqual(manifest["metadata"]["architecture"], "qwen35")
            self.assertEqual(manifest["tensor_type_counts"], {"F32": 1, "Q4_K": 2})
            self.assertEqual(manifest["file"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(manifest["integrity"]["status"], "local_digest_only")
            self.assertEqual(manifest["compilable_gate_up_layers"], [0])

    def test_config_cannot_hide_wrong_gguf_dimensions_or_missing_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.gguf"
            write_fixture(path, missing_up=True)
            with self.assertRaisesRegex(ValueError, "missing.*ffn_up"):
                build_manifest(path, expected_dimensions={"hidden": 256, "ffn": 256, "layers": 1})
            config = root / "config.json"
            config.write_text(json.dumps({"hidden_size": 5120, "intermediate_size": 17408,
                                         "num_hidden_layers": 64}))
            with self.assertRaisesRegex(ValueError, "config.*GGUF"):
                build_manifest(path, config_path=config)


@unittest.skipUnless(os.environ.get("RUN_QWEN38_INTEGRATION") == "1" and MODEL.exists(),
                     "set RUN_QWEN38_INTEGRATION=1 for the real model")
class Qwen38AssetManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_manifest(MODEL, config_path=CONFIG if CONFIG.exists() else None)

    def test_manifest_requires_qwen38_27b_gguf_and_ffn_tensors(self) -> None:
        manifest = self.manifest

        self.assertEqual(manifest["model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(manifest["quantization"], "UD-Q4_K_M (mixed)")
        self.assertEqual(
            manifest["dimensions"],
            {"hidden": 5120, "ffn": 17408, "layers": 64},
        )
        self.assertEqual(manifest["file"]["bytes"], MODEL.stat().st_size)
        self.assertRegex(manifest["file"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(manifest["tensors"]), 64 * 3)
        self.assertTrue(all(item["name"].startswith("blk.") for item in manifest["tensors"]))
        self.assertEqual(sum(manifest["tensor_type_counts"].values()), 192)
        self.assertEqual(manifest["tensors"][0]["type_name"], "IQ4_XS")

    def test_manifest_is_json_serializable(self) -> None:
        json.dumps(self.manifest, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
