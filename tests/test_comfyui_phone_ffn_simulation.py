import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simulate_comfyui_phone_ffn import latent_shape, projection_weight_bytes


class ComfyUIPhoneSimulationTests(unittest.TestCase):
    def test_h3_default_latent_shape(self) -> None:
        self.assertEqual(latent_shape(832, 480, 124), (124, 37, 14430, 207, 414))

    def test_h3_residual_weight_ledger(self) -> None:
        self.assertEqual(projection_weight_bytes(28672, 5376, 2, 32, 4), 57_802_752)
        self.assertEqual(projection_weight_bytes(5376, 14336, 2, 32, 4), 28_901_376)


if __name__ == "__main__":
    unittest.main()
