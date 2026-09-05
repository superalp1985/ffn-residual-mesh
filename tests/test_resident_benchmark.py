import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "GPU benchmark test needs cu130 venv")
class ResidentBenchmarkTests(unittest.TestCase):
    def test_report_separates_static_traffic_runtime_traffic_and_unmeasured_model(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from benchmark_resident_residual_cuda import benchmark
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            report = benchmark(root / "artifact", repeats=3, cpu_threads=(1,), launch_shapes=((1, 4),))
            self.assertEqual(report["status"], "measured_gate_up_swiglu_only")
            self.assertIsNone(report["tokens_per_second"])
            self.assertEqual(report["dynamic_h2d_bytes_per_run"], 3072)
            self.assertEqual(report["residual_weight_h2d_bytes_per_run"], 0)
            self.assertGreater(report["resident_weight_bytes"], 0)
            self.assertLess(report["correctness"]["gate"]["rel_l2"], 1e-5)
            self.assertLess(report["correctness"]["swiglu"]["rel_l2"], 1e-5)
            self.assertEqual(report["unmeasured"], ["down", "attention", "KV", "window_paging", "generation"])
            self.assertEqual(report["streamed_raw_q4_baseline"]["weight_h2d_bytes_per_run"], 73728)
            self.assertGreater(report["streamed_raw_q4_baseline"]["median"]["weight_h2d_ms"], 0)


if __name__ == "__main__":
    unittest.main()
