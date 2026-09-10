from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
    "device-chain CLI test requires the cu130 venv",
)
class ResidentDeviceChainCliTests(unittest.TestCase):
    def test_main_forwards_per_layer_down_tuning(self) -> None:
        import benchmark_resident_device_chain as module

        argv = [
            "benchmark_resident_device_chain.py",
            "--first-artifact", "first",
            "--first-layer", "40",
            "--second-artifact", "second",
            "--second-layer", "42",
            "--model", "model.gguf",
            "--first-down-block-rows", "2",
            "--first-down-num-warps", "2",
            "--first-down-block-qblocks", "4",
            "--second-down-block-rows", "4",
            "--second-down-num-warps", "2",
            "--second-down-block-qblocks", "4",
        ]
        with mock.patch.object(module, "benchmark", return_value={"ok": True}) as run:
            with mock.patch.object(sys, "argv", argv):
                module.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["first_down_block_rows"], 2)
        self.assertEqual(kwargs["first_down_num_warps"], 2)
        self.assertEqual(kwargs["first_down_block_qblocks"], 4)
        self.assertEqual(kwargs["second_down_block_rows"], 4)
        self.assertEqual(kwargs["second_down_num_warps"], 2)
        self.assertEqual(kwargs["second_down_block_qblocks"], 4)


if __name__ == "__main__":
    unittest.main()
