import sys
import unittest
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "src"))

from full_ffn_loopback import direct_ffn, random_layer, run_full_ffn_layer  # noqa: E402
from simulate_full_ffn_phone_offload import simulate  # noqa: E402


class FullFfnPhoneOffloadTests(unittest.TestCase):
    def test_loopback_full_ffn_matches_direct_cpu(self) -> None:
        layer = random_layer(3, hidden=16, ffn=32, seed=7)
        x = np.random.default_rng(9).standard_normal((2, 16)).astype(np.float32)
        expected = direct_ffn(x, layer.gate, layer.up, layer.down)
        actual = run_full_ffn_layer(layer, x)
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)

    def test_simulation_shows_small_text_activation_boundary(self) -> None:
        args = type(
            "Args",
            (),
            {
                "rows": 1,
                "hidden": 2048,
                "ffn": 6144,
                "layers": 24,
                "batch": 1,
                "steps": 1,
                "cpu_full_model_ms": 23.3,
                "ffn_share": 0.55,
                "phone_ffn_ms": 2.0,
                "network_gbps": 10.0,
                "network_latency_ms": 0.35,
                "weight_bits": 4,
                "activation_bytes": 2,
                "phone_counts": "1,8",
            },
        )()
        result = simulate(args)
        self.assertEqual(result["ledger"]["all_boundaries_mib_per_batch"], 0.1875)
        self.assertEqual(result["scenarios"][0]["single_stream_parallelism"], 1)

    def test_simulation_exposes_video_boundary_cost(self) -> None:
        args = type(
            "Args",
            (),
            {
                "rows": 15356,
                "hidden": 5376,
                "ffn": 14336,
                "layers": 50,
                "batch": 1,
                "steps": 1,
                "cpu_full_model_ms": 12330.0,
                "ffn_share": 0.55,
                "phone_ffn_ms": 100.0,
                "network_gbps": 10.0,
                "network_latency_ms": 0.35,
                "weight_bits": 8,
                "activation_bytes": 2,
                "phone_counts": "1",
            },
        )()
        result = simulate(args)
        self.assertGreater(result["ledger"]["all_boundaries_mib_per_batch"], 15000.0)
        self.assertGreater(result["scenarios"][0]["phone_ffn_serial_ms"], 1000.0)


if __name__ == "__main__":
    unittest.main()
