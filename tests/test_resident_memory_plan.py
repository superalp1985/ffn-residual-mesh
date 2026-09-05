import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plan_resident_memory import plan_residency


class ResidentMemoryPlanTests(unittest.TestCase):
    def test_reserve_and_double_buffer_reduce_window_without_negative_budget(self):
        config = {
            "num_hidden_layers": 4, "layer_types": ["linear_attention", "full_attention"] * 2,
            "hidden_size": 256, "intermediate_size": 256, "num_key_value_heads": 2, "head_dim": 32,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
        }
        tensors = [{"bytes": 300000, "projection": f"ffn_{p}", "layer": layer,
                    "type_name": "Q4_K"} for layer in range(4) for p in ("gate", "up", "down")]
        report = plan_residency({"tensors": tensors, "compilable_gate_up_layers": [0, 1, 2, 3]},
                                config, vram_bytes=1000000, context_tokens=128,
                                fixed_reserve_bytes=100000)
        self.assertEqual(report["kv_bytes"], 2 * 128 * 2 * 32 * 2 * 2)
        self.assertFalse(report["all_layers_resident"])
        for window in report["windows"]:
            self.assertLessEqual(window["working_set_bytes"], 1000000)
        self.assertIsNone(report["tokens_per_second"])
        self.assertEqual(report["status"], "capacity_estimate_not_runtime_measurement")

    def test_insufficient_memory_produces_no_plan_not_negative_window(self):
        config = {"num_hidden_layers": 1, "layer_types": ["full_attention"],
                  "hidden_size": 256, "intermediate_size": 256,
                  "num_key_value_heads": 1, "head_dim": 32}
        report = plan_residency({"tensors": [{"layer": 0, "bytes": 1000, "type_name": "Q4_K",
                                             "projection": "ffn_down"}],
                                "compilable_gate_up_layers": []}, config, vram_bytes=10,
                               context_tokens=128, fixed_reserve_bytes=100)
        self.assertEqual(report["windows"], [])
        self.assertEqual(report["budget_for_ffn_bytes"], 0)
        self.assertFalse(report["all_layers_resident"])


if __name__ == "__main__":
    unittest.main()
