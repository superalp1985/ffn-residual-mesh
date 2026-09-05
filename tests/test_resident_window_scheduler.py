from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resident_window_scheduler import ResidencyError, ResidentWindowScheduler  # noqa: E402


class ResidentWindowSchedulerTests(unittest.TestCase):
    def make_scheduler(self) -> ResidentWindowScheduler:
        return ResidentWindowScheduler(
            window_layers=2,
            vram_budget_bytes=1_000,
            reserve_bytes=100,
            layer_bytes={0: 300, 1: 300, 2: 300, 3: 300},
            persistent_layers=(0,),
        )

    def test_persistent_residual_hit_has_zero_weight_h2d(self) -> None:
        scheduler = self.make_scheduler()

        first = scheduler.acquire_layer(0)
        second = scheduler.acquire_layer(0)

        self.assertTrue(first["resident_hit"])
        self.assertTrue(second["resident_hit"])
        self.assertEqual(scheduler.traffic["weight_h2d_bytes"], 0)
        self.assertEqual(scheduler.traffic["resident_hits"], 2)
        self.assertEqual(scheduler.resident_layers(), [0])
        scheduler.release_window(0)
        self.assertEqual(scheduler.active_layers(), [])
        self.assertEqual(scheduler.resident_layers(), [0])

    def test_prefetch_and_acquire_move_only_the_missing_package(self) -> None:
        scheduler = self.make_scheduler()

        report = scheduler.prefetch_window(1)
        self.assertEqual(report["pending_layers"], [1, 2])
        self.assertEqual(report["planned_h2d_bytes"], 600)
        self.assertEqual(scheduler.traffic["weight_h2d_bytes"], 0)

        result = scheduler.acquire_layer(1)
        self.assertFalse(result["resident_hit"])
        self.assertEqual(result["weight_h2d_bytes"], 300)
        self.assertEqual(scheduler.traffic["weight_h2d_bytes"], 300)
        self.assertEqual(scheduler.pending_layers(), [2])

    def test_inflight_layer_cannot_be_evicted_or_overcommitted(self) -> None:
        scheduler = self.make_scheduler()
        scheduler.acquire_layer(1)
        scheduler.begin_kernel(1)

        with self.assertRaises(ResidencyError):
            scheduler.evict_layer(1)
        with self.assertRaises(ResidencyError):
            scheduler.prefetch_window(2)

        scheduler.end_kernel(1)
        scheduler.release_layer(1)
        report = scheduler.prefetch_window(2)
        self.assertEqual(report["pending_layers"], [2, 3])
        scheduler.complete_prefetch(2)
        self.assertEqual(scheduler.resident_layers(), [0, 2])

    def test_inflight_prefetch_cannot_be_reused_or_evicted(self) -> None:
        scheduler = self.make_scheduler()
        scheduler.prefetch_window(1)
        scheduler.begin_prefetch(1)

        with self.assertRaises(ResidencyError):
            scheduler.acquire_layer(1)
        with self.assertRaises(ResidencyError):
            scheduler.evict_layer(1)

        scheduler.complete_prefetch(1)
        self.assertEqual(scheduler.acquire_layer(1)["resident_hit"], True)

    def test_failed_prefetch_does_not_partially_evict_transient_layers(self) -> None:
        scheduler = ResidentWindowScheduler(
            window_layers=1,
            vram_budget_bytes=500,
            layer_bytes={0: 200, 1: 200, 2: 400},
        )
        scheduler.acquire_layer(0)
        scheduler.acquire_layer(1)
        scheduler.release_layer(1)

        with self.assertRaises(ResidencyError):
            scheduler.prefetch_window(2)
        self.assertEqual(scheduler.resident_layers(), [0, 1])

    def test_release_window_keeps_persistent_package(self) -> None:
        scheduler = self.make_scheduler()
        scheduler.acquire_layer(1)
        scheduler.release_window(1)

        self.assertEqual(scheduler.active_layers(), [])
        self.assertEqual(scheduler.resident_layers(), [0, 1])
        scheduler.evict_layer(1)
        self.assertEqual(scheduler.resident_layers(), [0])
        with self.assertRaises(ResidencyError):
            scheduler.evict_layer(0)

    def test_trim_does_not_charge_reserve_twice(self) -> None:
        scheduler = self.make_scheduler()
        scheduler.acquire_layer(1)
        scheduler.complete_prefetch(2)
        scheduler.release_window(1)
        self.assertEqual(scheduler.trim_to_budget(), [])
        self.assertEqual(scheduler.resident_layers(), [0, 1, 2])

    def test_persistent_addition_checks_union_and_existing_residency(self) -> None:
        scheduler = ResidentWindowScheduler(
            1, 500, layer_bytes={0: 300, 1: 300}, persistent_layers=(0,),
        )
        with self.assertRaises(ResidencyError):
            scheduler.set_persistent_layers([1])
        self.assertEqual(scheduler.resident_layers(), [0])
        self.assertEqual(scheduler.budget_report()["persistent_layers"], [0])

    def test_promotion_of_pending_copy_is_rejected(self) -> None:
        scheduler = self.make_scheduler()
        scheduler.prefetch_window(1)
        scheduler.begin_prefetch(1)
        with self.assertRaises(ResidencyError):
            scheduler.set_persistent_layers([1])
        self.assertEqual(scheduler.pending_layers(), [1, 2])

    def test_tail_window_and_sparse_registered_ids(self) -> None:
        scheduler = ResidentWindowScheduler(4, 1000, layer_bytes={3: 100, 21: 200, 51: 300})
        report = scheduler.prefetch_window(21)
        self.assertEqual(report["target_layers"], [21, 51])
        scheduler.acquire_layer(51)
        self.assertEqual(scheduler.traffic["weight_h2d_bytes"], 300)

    def test_release_window_does_not_release_another_window(self) -> None:
        scheduler = ResidentWindowScheduler(1, 1000, layer_bytes={0: 100, 1: 100})
        scheduler.acquire_layer(0)
        scheduler.acquire_layer(1)
        scheduler.release_window(0)
        self.assertEqual(scheduler.active_layers(), [1])


if __name__ == "__main__":
    unittest.main()
