from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_qwen38_manifest import ALLOWED_TYPES, build_manifest  # noqa: E402


MODEL = Path(os.environ.get("QWEN38_27B_GGUF", r"E:\Qwen3.8-27B\Qwen3.8-27B-UD-Q4_K_M.gguf"))
CONFIG = MODEL.with_name("config.json")


@unittest.skipUnless(MODEL.exists(), "Qwen3.8-27B GGUF is not available")
class Qwen38AssetManifestTests(unittest.TestCase):
    def test_manifest_requires_qwen38_27b_gguf_and_ffn_tensors(self) -> None:
        manifest = build_manifest(MODEL, config_path=CONFIG if CONFIG.exists() else None)

        self.assertEqual(manifest["model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(manifest["quantization"], "Q4_K")
        self.assertEqual(
            manifest["dimensions"],
            {"hidden": 5120, "ffn": 17408, "layers": 64},
        )
        self.assertEqual(manifest["file"]["bytes"], MODEL.stat().st_size)
        self.assertRegex(manifest["file"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(manifest["tensors"]), 64 * 3)
        self.assertTrue(all(item["name"].startswith("blk.") for item in manifest["tensors"]))
        self.assertTrue(all(item["type_name"] in ALLOWED_TYPES for item in manifest["tensors"]))
        self.assertFalse(manifest["runtime_requires_table_lookup"])

    def test_manifest_is_json_serializable(self) -> None:
        manifest = build_manifest(MODEL, config_path=CONFIG if CONFIG.exists() else None)
        json.dumps(manifest, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
