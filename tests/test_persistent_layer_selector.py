from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persistent_layer_selector import select_persistent_layers  # noqa: E402


class PersistentLayerSelectorTests(unittest.TestCase):
    def test_selector_maximizes_weighted_hits_under_capacity(self) -> None:
        result = select_persistent_layers(
            {0: 6, 1: 4, 2: 4},
            capacity_bytes=8,
            access_weights={0: 1, 1: 5, 2: 5},
        )
        self.assertEqual(result["selected_layers"], [1, 2])
        self.assertEqual(result["selected_bytes"], 8)
        self.assertEqual(result["weighted_hit_score"], 10)

    def test_ties_are_deterministic_and_budget_is_strict(self) -> None:
        result = select_persistent_layers(
            {2: 4, 0: 4, 1: 4},
            capacity_bytes=7,
            access_weights={0: 1, 1: 1, 2: 1},
        )
        self.assertEqual(result["selected_layers"], [0])
        self.assertEqual(result["selected_bytes"], 4)
        self.assertEqual(result["unselected_layers"], [1, 2])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_persistent_layers({0: 0}, capacity_bytes=1)
        with self.assertRaises(ValueError):
            select_persistent_layers({0: 1}, capacity_bytes=-1)


if __name__ == "__main__":
    unittest.main()
