import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_exact_radix_activation_error import ffn_forward, relative_l2_error


class ActivationQuantizationErrorTests(unittest.TestCase):
    def test_ffn_forward_preserves_batch_and_hidden_shape(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(3, 4)).astype(np.float32)
        wg = rng.normal(size=(6, 4)).astype(np.float32)
        wu = rng.normal(size=(6, 4)).astype(np.float32)
        wd = rng.normal(size=(4, 6)).astype(np.float32)

        output = ffn_forward(x, wg, wu, wd)

        self.assertEqual(output.shape, (3, 4))

    def test_relative_l2_error_is_zero_for_identical_arrays(self) -> None:
        value = np.arange(8, dtype=np.float32).reshape(2, 4)

        self.assertEqual(relative_l2_error(value, value), 0.0)
