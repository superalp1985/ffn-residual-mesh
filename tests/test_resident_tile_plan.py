from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resident_tile_plan import TilePlan  # noqa: E402
from resident_tiled_ffn import resolve_q5_base_schedule  # noqa: E402


class ResidentTilePlanTests(unittest.TestCase):
    def test_tiles_cover_rows_without_overlap(self) -> None:
        plan = TilePlan(rows=17408, tile_rows=1024, projections=("gate", "up"))
        self.assertEqual(plan.ranges()[0], (0, 1024))
        self.assertEqual(plan.ranges()[-1], (16384, 17408))
        flattened = [row for start, stop in plan.ranges() for row in range(start, stop)]
        self.assertEqual(flattened, list(range(17408)))

    def test_package_bytes_are_per_projection_and_tile(self) -> None:
        plan = TilePlan(rows=256, tile_rows=64, projections=("gate", "up"))
        self.assertEqual(plan.tile_bytes(cols=512, alpha_cols=16), 2 * (64 * 256 + 64 * 16 * 4))
        self.assertEqual(plan.total_bytes(cols=512, alpha_cols=16), 8 * (64 * 256 + 64 * 16 * 4))
        self.assertEqual(
            plan.tile_bytes(
                cols=512,
                alpha_cols=16,
                residual_bits={"gate": 4, "up": 5},
            ),
            64 * (512 * 4 // 8 + 16 * 4)
            + 64 * (512 * 5 // 8 + 16 * 4),
        )

    def test_invalid_tile_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TilePlan(rows=0, tile_rows=64, projections=("gate", "up"))
        with self.assertRaises(ValueError):
            TilePlan(rows=256, tile_rows=63, projections=("gate",))
        with self.assertRaises(ValueError):
            TilePlan(rows=256, tile_rows=64, projections=())

    def test_tile_slices_preserve_row_bounds(self) -> None:
        plan = TilePlan(rows=130, tile_rows=64, projections=("gate", "up"))
        self.assertEqual(plan.tile_slices(), [(0, 64), (64, 128), (128, 130)])

    def test_q4_keeps_legacy_base_schedule(self) -> None:
        self.assertEqual(
            resolve_q5_base_schedule(
                {"gate": 4, "up": 4},
                block_rows=2,
                num_warps=2,
                base_block_groups=256,
                auto_tune_q5_base=True,
            ),
            (2, 2, 256, False),
        )

    def test_q5_default_uses_low_register_schedule(self) -> None:
        self.assertEqual(
            resolve_q5_base_schedule(
                {"gate": 5, "up": 5},
                block_rows=2,
                num_warps=2,
                base_block_groups=256,
                auto_tune_q5_base=True,
            ),
            (4, 2, 16, True),
        )

    def test_mixed_default_uses_low_register_schedule(self) -> None:
        self.assertEqual(
            resolve_q5_base_schedule(
                {"gate": 4, "up": 5},
                block_rows=2,
                num_warps=2,
                base_block_groups=256,
                auto_tune_q5_base=True,
            ),
            (4, 2, 16, True),
        )

    def test_explicit_sweep_shape_is_not_overridden(self) -> None:
        self.assertEqual(
            resolve_q5_base_schedule(
                {"gate": 5, "up": 4},
                block_rows=4,
                num_warps=8,
                base_block_groups=64,
                auto_tune_q5_base=False,
            ),
            (4, 8, 64, False),
        )


if __name__ == "__main__":
    unittest.main()
