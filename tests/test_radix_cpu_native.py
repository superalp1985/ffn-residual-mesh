import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_exact_radix_split_pipeline import compile_radix_table


@unittest.skipUnless(shutil.which("g++"), "native AVX2 test requires g++ on an AVX2 host")
class RadixCpuNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="radix-native-tests-")
        cls.directory = Path(cls.temp.name)
        cls.exe = cls.directory / "bench.exe"
        source = ROOT / "src" / "radix_cpu_bench.cpp"
        if not source.exists():
            return
        subprocess.run(
            ["g++", "-O3", "-mavx2", "-ffp-contract=off", "-std=c++17", "-pthread", str(source), "-o", str(cls.exe)],
            check=True, capture_output=True, text=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_exactness_and_prefetch_with_signed_extremes_and_row_tails(self):
        self.assertTrue(self.exe.exists(), "native optimized benchmark has not been implemented")
        rng = np.random.default_rng(49)
        for block in (2, 4):
            rows, hidden, tokens = 37, 64, 5
            q = rng.integers(0, 16, (rows, hidden), dtype=np.uint8)
            q[0] = 15
            q[1] = 0
            q[2] = np.arange(hidden, dtype=np.uint8) & 15
            alpha = rng.uniform(0.005, 0.2, (rows, hidden // 32)).astype("<f4")
            beta = rng.uniform(-0.5, 0, alpha.shape).astype("<f4")
            z = rng.integers(-128, 128, (tokens, hidden), dtype=np.int16).astype("i1")
            z[0] = -128
            z[1] = 127
            z[2] = 0
            scale = rng.uniform(0.01, 0.1, (tokens, hidden // 32)).astype("<f4")
            table, high_sum = compile_radix_table(q.reshape(rows, -1, 32) >> 2, block)
            for name, value in (
                ("table", table), ("q", q), ("alpha", alpha), ("beta", beta),
                ("high_sum", high_sum.astype("<i2")), ("z", z), ("scale", scale),
            ):
                value.tofile(self.directory / f"{name}.bin")
            completed = subprocess.run(
                [str(self.exe), str(self.directory), str(rows), str(hidden),
                 str(block), str(tokens), "5", "3", "0,1,2,4,8"],
                check=True, capture_output=True, text=True,
            )
            report = json.loads(completed.stdout)
            self.assertEqual(report["integer_mismatches"], 0)
            self.assertLess(report["max_scaled_abs_error"], 0.001)
            self.assertLess(report["max_merge_abs_error"], 0.001)
            self.assertEqual(len(report["methods"]), 7)
            for method in report["methods"]:
                self.assertEqual(len(method["samples_ms"]), 5)
                self.assertGreater(method["median_ms"], 0)
                self.assertTrue(np.isfinite(method["checksum"]))

    def test_native_bridge_recomputes_each_input_and_matches_python_oracle(self):
        wrapper = ROOT / "scripts" / "native_radix_cpu.py"
        self.assertTrue(wrapper.exists(), "native bridge has not been implemented")
        from native_radix_cpu import NativeRadixBase
        from benchmark_exact_radix_split_pipeline import (
            direct_group_dots, projection_from_group_dots,
        )
        rng = np.random.default_rng(490)
        rows, groups, block = 37, 2, 4
        high = rng.integers(0, 4, (rows, groups, 32), dtype=np.uint8)
        table, high_sum = compile_radix_table(high, block)
        alpha = rng.normal(size=(rows, groups)).astype("<f4")
        beta = rng.normal(size=alpha.shape).astype("<f4")
        for suffix, value in (
            ("table.u8", table), ("high_sum.i16", high_sum.astype("<i2")),
            ("alpha.f32", alpha), ("beta.f32", beta),
        ):
            value.tofile(self.directory / f"gate.{suffix}.bin")
        for mode in ("legacy", "fused"):
            with NativeRadixBase(self.directory, "gate", rows, groups * 32, block,
                                 threads=3, prefetch=4, mode=mode) as native:
                for _ in range(3):
                    z = rng.integers(-128, 128, (1, groups * 32), dtype=np.int16).astype("i1")
                    scales = rng.uniform(0.001, 0.01, (1, groups)).astype("<f4")
                    expected = projection_from_group_dots(
                        direct_group_dots(high, z), z, scales, alpha, beta, code_multiplier=4,
                    )
                    actual = np.full(rows, np.nan, dtype="<f4")
                    native.evaluate(z, scales, actual)
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                with self.assertRaises(ValueError):
                    native.evaluate(z[:, :-1], scales, actual)


if __name__ == "__main__":
    unittest.main()
