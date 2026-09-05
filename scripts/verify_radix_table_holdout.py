from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark_exact_radix_split_pipeline import (
    compile_radix_table,
    direct_group_dots,
    encode_signed_base4_states,
    evaluate_radix_table,
    load_q4_projection,
    quantize_groupwise_q8,
)
from evaluate_polynomial_base_residual import load_layer


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact radix table reconstruction over holdout tokens")
    parser.add_argument("model", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    x = load_layer(args.holdout_root, args.layer)[0][: args.tokens].astype(np.float32, copy=False)
    z, scales = quantize_groupwise_q8(x, group_size=32)
    rows = []
    for projection in ("gate", "up"):
        codes, _, _, _ = load_q4_projection(args.model, args.layer, projection)
        table_start = time.perf_counter()
        table, high_sum = compile_radix_table(codes >> 2, block_size=args.block_size)
        compile_seconds = time.perf_counter() - table_start
        exact = True
        max_error = 0
        for token in range(len(x)):
            states = encode_signed_base4_states(z[token], block_size=args.block_size)
            predicted = evaluate_radix_table(table, high_sum, states, 32 // args.block_size)
            direct = np.einsum(
                "rgi,gi->rg",
                (codes >> 2).astype(np.int32),
                z[token].reshape(codes.shape[1], codes.shape[2]).astype(np.int32),
                optimize=True,
            )
            error = np.abs(predicted - direct)
            max_error = max(max_error, int(error.max()))
            exact = exact and bool(np.array_equal(predicted, direct))
        rows.append(
            {
                "projection": projection,
                "tokens": len(x),
                "block_size": args.block_size,
                "table_bytes": int(table.nbytes),
                "table_mib": table.nbytes / 2**20,
                "cold_compile_seconds": compile_seconds,
                "integer_high_dot_exact_for_all_tokens": exact,
                "max_integer_error": max_error,
                "activation_scale_bytes_per_token": int(scales.shape[1] * 4),
            }
        )
        del table

    result = {
        "experiment": "exact_radix_table_holdout_verification",
        "layer": args.layer,
        "rows": rows,
        "scope": "exactness of the compiled high-term table for observed int8 activation states; activation quantization error is measured separately",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
