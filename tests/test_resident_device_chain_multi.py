from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
    "multi-layer device-chain test requires the cu130 venv",
)
class ResidentDeviceChainMultiTests(unittest.TestCase):
    def test_parse_config(self) -> None:
        from benchmark_resident_device_chain_multi import _parse_config

        self.assertEqual(
            _parse_config("block_rows=4,num_warps=2,block_qblocks=4"),
            {"block_rows": 4, "num_warps": 2, "block_qblocks": 4},
        )

    def test_parse_config_rejects_unknown_keys(self) -> None:
        from benchmark_resident_device_chain_multi import _parse_config

        with self.assertRaises(ValueError):
            _parse_config("block_rows=4,unknown=2")


if __name__ == "__main__":
    unittest.main()
