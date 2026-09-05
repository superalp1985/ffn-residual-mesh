from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resident_tile_plan import TilePlan  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
