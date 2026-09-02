from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--table-k", type=int, default=8)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--approx-fraction", type=float, default=0.75)
    parser.add_argument("--fallback-fraction", type=float, default=0.25)
    args = parser.parse_args()
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    hidden = 2048
    n_ff = 6144
    blocks = (n_ff + args.block_dim - 1) // args.block_dim
    output_bytes = hidden * 2
    coeff_bytes = args.rank * 2
    basis_bytes = hidden * args.rank * 2
    table_bytes = blocks * args.table_k * hidden * 2
    center_bytes = blocks * args.table_k * args.block_dim * 2
    rows = []
    for layer in layout["layers"]:
        full = int(layer["bytes"])
        # Approx path: CPU table lookup, then either full output or rank coefficients.
        cpu_output = output_bytes
        gpu_coeff = coeff_bytes
        gpu_resident_basis = basis_bytes
        expected_full_output = args.approx_fraction * gpu_coeff + args.fallback_fraction * full
        expected_cpu_output = args.approx_fraction * cpu_output + args.fallback_fraction * full
        rows.append({
            "layer": int(layer["layer"]),
            "full_ffn_weight_bytes": full,
            "table_bytes_fp16": table_bytes + center_bytes,
            "resident_basis_bytes_fp16": basis_bytes,
            "approx_output_bytes": cpu_output,
            "approx_coeff_bytes": gpu_coeff,
            "expected_h2d_bytes_output_path": expected_cpu_output,
            "expected_h2d_bytes_coeff_path": expected_full_output,
            "h2d_reduction_output_path": 1.0 - expected_cpu_output / full,
            "h2d_reduction_coeff_path": 1.0 - expected_full_output / full,
        })
    result = {
        "model": layout["model"],
        "rank": args.rank,
        "table_k": args.table_k,
        "block_dim": args.block_dim,
        "approx_fraction": args.approx_fraction,
        "fallback_fraction": args.fallback_fraction,
        "global_table_bytes_fp16": table_bytes * len(rows),
        "global_center_bytes_fp16": center_bytes * len(rows),
        "global_basis_bytes_fp16": basis_bytes * len(rows),
        "rows": rows,
        "interpretation": [
            "The table and basis are cold-start artifacts stored in host RAM; resident_basis_bytes is a one-time GPU upload per layer if the coefficient path is used.",
            "Expected H2D excludes activations and synchronization overhead; it is a lower-bound bandwidth model.",
            "Fallback must transfer the exact quantized FFN weights or use an already resident exact path.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"global_table_mib": (table_bytes + center_bytes) * len(rows) / 2**20, "global_basis_mib": basis_bytes * len(rows) / 2**20, "layer0": rows[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
