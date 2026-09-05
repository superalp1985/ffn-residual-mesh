from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_exact_radix_split_pipeline import (
    compile_radix_table,
    encode_signed_base4_states,
    load_q4_projection,
    quantize_groupwise_q8,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cold-start radix partial-sum tables")
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--block-size", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    x = np.fromfile(args.input, dtype="<f4").reshape(1, -1)
    z, scales = quantize_groupwise_q8(x, group_size=32)
    z = z.reshape(-1)
    states = encode_signed_base4_states(z, block_size=args.block_size)
    states.astype("<u2", copy=False).tofile(args.out / "states.u16.bin")
    scales.reshape(-1).astype("<f4", copy=False).tofile(args.out / "activation.scale.f32.bin")
    z.astype("i1", copy=False).tofile(args.out / "activation.z.i8.bin")

    projections = {}
    for projection in ("gate", "up"):
        codes, alpha, beta, source_bytes = load_q4_projection(args.model, args.layer, projection)
        table, high_sum = compile_radix_table(codes >> 2, block_size=args.block_size)
        table_path = args.out / f"{projection}.table.u8.bin"
        high_path = args.out / f"{projection}.high_sum.i16.bin"
        alpha_path = args.out / f"{projection}.alpha.f32.bin"
        beta_path = args.out / f"{projection}.beta.f32.bin"
        np.ascontiguousarray(table).tofile(table_path)
        np.ascontiguousarray(high_sum.astype("<i2", copy=False)).tofile(high_path)
        np.ascontiguousarray(alpha.astype("<f4", copy=False)).tofile(alpha_path)
        np.ascontiguousarray(beta.astype("<f4", copy=False)).tofile(beta_path)
        projections[projection] = {
            "rows": int(codes.shape[0]),
            "hidden": int(codes.shape[1] * codes.shape[2]),
            "groups": int(codes.shape[1]),
            "block_size": args.block_size,
            "blocks": int(table.shape[0]),
            "state_count": int(table.shape[1]),
            "table_bytes": int(table.nbytes),
            "high_sum_bytes": int(high_sum.nbytes),
            "alpha_bytes": int(alpha.nbytes),
            "beta_bytes": int(beta.nbytes),
            "q4_source_bytes": int(source_bytes),
            "files": {
                "table": table_path.name,
                "high_sum": high_path.name,
                "alpha": alpha_path.name,
                "beta": beta_path.name,
            },
        }
    result = {
        "format": "FFN_RADIX_TABLE_ARTIFACT_V1",
        "layer": args.layer,
        "block_size": args.block_size,
        "activation": {
            "hidden": int(z.size),
            "groups": int(scales.size),
            "group_size": 32,
            "files": {"z": "activation.z.i8.bin", "scale": "activation.scale.f32.bin", "states": "states.u16.bin"},
        },
        "projections": projections,
        "runtime_contract": "CPU reads finite radix table entries; GPU reads packed exact residual tiles; raw GGUF is cold-start-only",
    }
    (args.out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
