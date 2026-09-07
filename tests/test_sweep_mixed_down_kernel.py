from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "mixed down sweep test requires the cu130 venv")
class SweepMixedDownKernelTests(unittest.TestCase):
    def test_projection_factory_dispatches_all_supported_fixture_quantizers(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from gguf import GGUFReader
        from sweep_mixed_down_kernel import build_projection
        from tests.gguf_fixture import write_fixture

        expected = {
            "quantized_down": "IQ4_NL",
            "q4k_down": "Q4_K",
            "q5k_down": "Q5_K",
            "iq4xs_down": "IQ4_XS",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for option, quant_name in expected.items():
                path = root / f"{option}.gguf"
                write_fixture(path, **{option: True})
                reader = GGUFReader(path)
                try:
                    tensor = next(item for item in reader.tensors if item.name == "blk.0.ffn_down.weight")
                    projection, metadata = build_projection(
                        tensor,
                        block_rows=2,
                        num_warps=2,
                        block_qblocks=2,
                    )
                    self.assertEqual(metadata["quant_type"], quant_name)
                    self.assertEqual(projection.cols, 256)
                    self.assertEqual(projection.rows, 256)
                finally:
                    reader.data._mmap.close()


if __name__ == "__main__":
    unittest.main()
