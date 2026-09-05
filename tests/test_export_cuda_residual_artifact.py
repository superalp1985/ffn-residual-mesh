import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_cuda_residual_artifact_v2 import artifact_manifest, pack_row_codes


class ExportCudaResidualArtifactTests(unittest.TestCase):
    def test_row_packing_keeps_four_two_bit_codes_per_byte(self) -> None:
        codes = np.array([[0, 1, 2, 3, 3, 2, 1, 0]], dtype=np.uint8)

        packed = pack_row_codes(codes, bits=2)

        np.testing.assert_array_equal(packed, np.array([[228, 27]], dtype=np.uint8))

    def test_manifest_declares_exact_lossless_runtime_contract(self) -> None:
        manifest = artifact_manifest(
            projection="gate",
            layer=23,
            rows=256,
            hidden=2048,
            residual_bits=2,
            code_bytes=131072,
            alpha_bytes=524288,
        )

        self.assertTrue(manifest["lossless"])
        self.assertEqual(manifest["runtime_weight_reads"], 0)
        self.assertEqual(manifest["packed_code_bytes"], 131072)
        self.assertEqual(manifest["alpha_bytes"], 524288)
