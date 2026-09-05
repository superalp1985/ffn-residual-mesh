import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_exact_radix_split_pipeline import (
    compile_radix_table,
    encode_signed_base4_states,
    evaluate_radix_table,
    pack_2bit_rows,
    quantize_groupwise_q8,
    unpack_2bit_rows,
)


class ExactRadixMainTableTests(unittest.TestCase):
    def test_table_reconstructs_signed_int8_high_dot_exactly(self) -> None:
        q_hi = (np.arange(2 * 2 * 8, dtype=np.uint8).reshape(2, 2, 8) * 3 + 1) & 3
        z = np.array(
            [
                [
                    [-128, -65, -1, 0, 1, 63, 64, 127],
                    [127, 64, 63, 1, 0, -1, -65, -128],
                ],
                [
                    [-17, 5, 12, 31, -44, 88, -99, 106],
                    [13, -26, 39, -52, 65, -78, 91, -104],
                ],
            ],
            dtype=np.int8,
        )
        table, high_sum = compile_radix_table(q_hi, block_size=4)

        for token in z:
            states = encode_signed_base4_states(token.reshape(-1), block_size=4)
            reconstructed = evaluate_radix_table(table, high_sum, states, blocks_per_group=2)
            direct = np.einsum("rgi,gi->rg", q_hi.astype(np.int32), token.astype(np.int32))
            np.testing.assert_array_equal(reconstructed, direct)

    def test_block_two_table_is_also_exact(self) -> None:
        rng = np.random.default_rng(8)
        q_hi = rng.integers(0, 4, size=(3, 2, 8), dtype=np.uint8)
        z = rng.integers(-128, 128, size=(16,), dtype=np.int16).astype(np.int8)
        table, high_sum = compile_radix_table(q_hi, block_size=2)
        states = encode_signed_base4_states(z, block_size=2)

        reconstructed = evaluate_radix_table(table, high_sum, states, blocks_per_group=4)
        direct = np.einsum("rgi,gi->rg", q_hi.astype(np.int32), z.reshape(2, 8).astype(np.int32))

        np.testing.assert_array_equal(reconstructed, direct)

    def test_full_high_low_merge_matches_original_codes(self) -> None:
        rng = np.random.default_rng(7)
        q = rng.integers(0, 16, size=(5, 3, 8), dtype=np.uint8)
        q_hi = q >> 2
        q_lo = q & 3
        z = rng.integers(-128, 128, size=(3, 8), dtype=np.int16).astype(np.int8)
        alpha = rng.normal(size=(5, 3)).astype(np.float32)
        beta = rng.normal(size=(5, 3)).astype(np.float32)
        scale = rng.uniform(0.001, 0.02, size=3).astype(np.float32)
        table, high_sum = compile_radix_table(q_hi, block_size=4)
        states = encode_signed_base4_states(z.reshape(-1), block_size=4)
        high_dot = evaluate_radix_table(table, high_sum, states, blocks_per_group=2)
        low_dot = np.einsum("rgi,gi->rg", q_lo.astype(np.int32), z.astype(np.int32))
        z_sum = z.astype(np.int32).sum(axis=1)

        split = np.sum(scale[None, :] * (alpha * (4 * high_dot + low_dot) + beta * z_sum[None, :]), axis=1)
        direct_dot = np.einsum("rgi,gi->rg", q.astype(np.int32), z.astype(np.int32))
        direct = np.sum(scale[None, :] * (alpha * direct_dot + beta * z_sum[None, :]), axis=1)
        np.testing.assert_allclose(split, direct, rtol=1e-6, atol=1e-6)

    def test_two_bit_packing_is_lossless(self) -> None:
        values = np.array([[0, 1, 2, 3, 3, 2, 1, 0]], dtype=np.uint8)
        packed = pack_2bit_rows(values)

        self.assertEqual(packed.tolist(), [[228, 27]])
        np.testing.assert_array_equal(unpack_2bit_rows(packed, values.shape[1]), values)

    def test_groupwise_q8_quantization_reports_scale_and_codes(self) -> None:
        x = np.array([[-1.0, -0.5, 0.0, 1.0, -2.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        codes, scales = quantize_groupwise_q8(x, group_size=4)

        np.testing.assert_allclose(scales, [[1.0 / 127.0, 2.0 / 127.0]], rtol=1e-6)
        np.testing.assert_array_equal(codes[0, :4], [-127, -64, 0, 127])
        np.testing.assert_array_equal(codes[0, 4:], [-127, 0, 64, 127])


if __name__ == "__main__":
    unittest.main()
