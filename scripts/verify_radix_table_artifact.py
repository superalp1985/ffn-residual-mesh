from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_exact_radix_split_pipeline import (
    direct_group_dots,
    evaluate_radix_table,
    load_q4_projection,
    projection_from_group_dots,
    quantize_groupwise_q8,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a persisted radix table artifact against Q4_K codes")
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    args = parser.parse_args()

    manifest = json.loads((args.artifact / "manifest.json").read_text(encoding="utf-8"))
    block_size = int(manifest["block_size"])
    rows = int(manifest["projections"]["gate"]["rows"])
    groups = int(manifest["projections"]["gate"]["groups"])
    blocks = int(manifest["projections"]["gate"]["blocks"])
    state_count = int(manifest["projections"]["gate"]["state_count"])
    x = np.fromfile(args.input, dtype="<f4").reshape(1, -1)
    z, scales = quantize_groupwise_q8(x, group_size=32)
    states = np.fromfile(args.artifact / "states.u16.bin", dtype="<u2")
    rows_out = []
    for projection in ("gate", "up"):
        codes, alpha, beta, _ = load_q4_projection(args.model, args.layer, projection)
        table = np.fromfile(args.artifact / f"{projection}.table.u8.bin", dtype=np.uint8)
        table = table.reshape(blocks, state_count, rows)
        high_sum = np.fromfile(args.artifact / f"{projection}.high_sum.i16.bin", dtype="<i2").reshape(rows, groups)
        alpha_artifact = np.fromfile(args.artifact / f"{projection}.alpha.f32.bin", dtype="<f4").reshape(rows, groups)
        beta_artifact = np.fromfile(args.artifact / f"{projection}.beta.f32.bin", dtype="<f4").reshape(rows, groups)
        if not np.array_equal(alpha, alpha_artifact) or not np.array_equal(beta, beta_artifact):
            raise AssertionError(f"{projection}: alpha/beta artifact mismatch")
        high_direct = direct_group_dots(codes >> 2, z)
        high_table = evaluate_radix_table(table, high_sum, states.reshape(4, blocks), 32 // block_size)
        low_direct = direct_group_dots(codes & 3, z)
        base = projection_from_group_dots(high_table, z, scales, alpha, beta, code_multiplier=4)
        direct_base = projection_from_group_dots(high_direct, z, scales, alpha, beta, code_multiplier=4)
        residual = projection_from_group_dots(low_direct, z, scales, alpha, np.zeros_like(beta), code_multiplier=1)
        direct_full = projection_from_group_dots(4 * high_direct + low_direct, z, scales, alpha, beta, code_multiplier=1)
        merged = base + residual
        rows_out.append(
            {
                "projection": projection,
                "integer_high_exact": bool(np.array_equal(high_table, high_direct)),
                "base_max_abs": float(np.max(np.abs(base - direct_base))),
                "merged_max_abs": float(np.max(np.abs(merged - direct_full))),
                "merged_rel_l2": float(np.linalg.norm(merged - direct_full) / max(np.linalg.norm(direct_full), 1e-12)),
            }
        )
    result = {
        "experiment": "verify_persisted_radix_table_artifact",
        "layer": args.layer,
        "block_size": block_size,
        "rows": rows_out,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
