from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "cache test requires PyTorch")
class ResidentPackageCacheTests(unittest.TestCase):
    def test_first_miss_uploads_only_package_and_hit_is_zero_transfer(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidentWindowScheduler

        packages = {
            3: {
                "residual": np.arange(64, dtype=np.int8).reshape(8, 8),
                "alpha": np.ones((8, 1), dtype=np.float32),
            },
        }
        scheduler = ResidentWindowScheduler(
            window_layers=1,
            vram_budget_bytes=4096,
            layer_bytes={3: sum(value.nbytes for value in packages[3].values())},
        )
        cache = ResidentPackageCache(packages, scheduler=scheduler)
        first = cache.acquire(3)
        second = cache.acquire(3)

        self.assertFalse(first["resident_hit"])
        self.assertEqual(first["weight_h2d_bytes"], 96)
        self.assertTrue(second["resident_hit"])
        self.assertEqual(second["weight_h2d_bytes"], 0)
        self.assertEqual(cache.traffic["weight_h2d_bytes"], 96)
        self.assertTrue(cache.package(3)["residual"].is_cuda)
        np.testing.assert_array_equal(cache.package(3)["residual"].cpu().numpy(),
                                      packages[3]["residual"])
        cache.release(3)
        cache.close()

    def test_copy_stream_event_is_recorded_for_a_miss(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidentWindowScheduler

        payload = np.zeros((32, 32), dtype=np.float16)
        scheduler = ResidentWindowScheduler(
            window_layers=1,
            vram_budget_bytes=4096,
            layer_bytes={0: payload.nbytes},
        )
        cache = ResidentPackageCache({0: {"residual": payload}}, scheduler=scheduler)
        result = cache.acquire(0)
        self.assertGreaterEqual(result["copy_ms"], 0.0)
        self.assertEqual(result["weight_h2d_bytes"], payload.nbytes)
        self.assertIsNotNone(result["copy_event"])
        result["copy_event"].synchronize()
        cache.close()

    def test_automatic_eviction_frees_device_package_and_reload_is_a_miss(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidentWindowScheduler
        payloads = {i: {"residual": np.full(256, i, dtype=np.uint8)} for i in range(3)}
        scheduler = ResidentWindowScheduler(1, 512, layer_bytes={i: 256 for i in payloads})
        with ResidentPackageCache(payloads, scheduler=scheduler) as cache:
            cache.acquire(0)
            cache.release(0)
            cache.acquire(1)
            cache.release(1)
            cache.acquire(2)
            cache.release(2)
            self.assertEqual(cache.device_layers(), scheduler.resident_layers())
            self.assertEqual(cache.device_bytes(), 512)
            victim = ({0, 1, 2} - set(cache.device_layers())).pop()
            self.assertFalse(cache.acquire(victim)["resident_hit"])
            np.testing.assert_array_equal(cache.package(victim)["residual"].cpu(),
                                          payloads[victim]["residual"])
            self.assertEqual(cache.traffic["weight_h2d_bytes"], 1024)

    def test_prefetch_and_lease_protect_copy_and_kernel_lifetimes(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidencyError, ResidentWindowScheduler
        payload = np.arange(1024, dtype=np.float32)
        scheduler = ResidentWindowScheduler(1, payload.nbytes, layer_bytes={0: payload.nbytes})
        stream = torch.cuda.Stream()
        with ResidentPackageCache({0: {"value": payload}}, scheduler=scheduler) as cache:
            cache.prefetch([0])
            with cache.lease(0, stream=stream):
                with torch.cuda.stream(stream):
                    result = cache.package(0)["value"] * 3
                with self.assertRaises(ResidencyError):
                    cache.evict(0)
            stream.synchronize()
            np.testing.assert_array_equal(result.cpu(), payload * 3)
            cache.evict(0)
            self.assertEqual(cache.device_bytes(), 0)

    def test_mismatched_byte_ledger_is_rejected_before_copy(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidentWindowScheduler
        scheduler = ResidentWindowScheduler(1, 4096, layer_bytes={0: 1})
        with self.assertRaises(ValueError):
            ResidentPackageCache({0: {"value": np.ones(512, dtype=np.float32)}},
                                 scheduler=scheduler)

    def test_persistent_initialization_uploads_once_then_survives_swaps(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidentWindowScheduler
        packages = {i: {"value": np.full(256, i, dtype=np.uint8)} for i in range(3)}
        scheduler = ResidentWindowScheduler(1, 512, layer_bytes={i: 256 for i in packages})
        with ResidentPackageCache(packages, scheduler=scheduler) as cache:
            cache.initialize_persistent([0])
            for i in (1, 2, 1, 2):
                cache.acquire(i)
                cache.release(i)
                self.assertEqual(cache.acquire(0)["weight_h2d_bytes"], 0)
                cache.release(0)
            self.assertIn(0, cache.device_layers())
            self.assertEqual(cache.traffic["weight_h2d_bytes"], 5 * 256)

    def test_async_prefetch_can_overlap_before_acquire_and_hit_has_no_copy(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidentWindowScheduler

        packages = {
            0: {"value": np.arange(4096, dtype=np.float32)},
            1: {"value": np.arange(4096, dtype=np.float32) + 1},
        }
        size = packages[0]["value"].nbytes
        scheduler = ResidentWindowScheduler(1, size * 2, layer_bytes={0: size, 1: size})
        with ResidentPackageCache(packages, scheduler=scheduler) as cache:
            tickets = cache.prefetch_async([0, 1])
            self.assertEqual(len(tickets), 2)
            self.assertEqual(cache.pending_layers(), [0, 1])
            cache.wait_prefetch(0)
            first = cache.acquire(0)
            self.assertEqual(first["weight_h2d_bytes"], 0)
            cache.release(0)
            cache.wait_prefetch(1)
            second = cache.acquire(1)
            self.assertEqual(second["weight_h2d_bytes"], 0)
            cache.release(1)
            self.assertEqual(cache.traffic["weight_h2d_bytes"], size * 2)

    def test_async_prefetch_requires_wait_before_package_access(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidentWindowScheduler

        payload = np.ones(1024, dtype=np.float32)
        scheduler = ResidentWindowScheduler(1, payload.nbytes, layer_bytes={0: payload.nbytes})
        with ResidentPackageCache({0: {"value": payload}}, scheduler=scheduler) as cache:
            cache.prefetch_async([0])
            with self.assertRaises(RuntimeError):
                cache.package(0)
            cache.wait_prefetch(0)
            np.testing.assert_array_equal(cache.package(0)["value"].cpu(), payload)

    def test_compute_stream_wait_event_is_recorded_for_async_ticket(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from resident_package_cache import ResidentPackageCache
        from resident_window_scheduler import ResidentWindowScheduler

        payload = np.arange(512, dtype=np.float32)
        scheduler = ResidentWindowScheduler(1, payload.nbytes, layer_bytes={0: payload.nbytes})
        compute = torch.cuda.Stream()
        with ResidentPackageCache({0: {"value": payload}}, scheduler=scheduler) as cache:
            cache.prefetch_async([0])
            event = cache.wait_prefetch(0, stream=compute)
            self.assertIsNotNone(event)
            with torch.cuda.stream(compute):
                result = cache.package(0)["value"] + 2
            compute.synchronize()
            np.testing.assert_array_equal(result.cpu(), payload + 2)


if __name__ == "__main__":
    unittest.main()
