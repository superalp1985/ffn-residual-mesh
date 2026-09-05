from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def state_table_ledger(
    *,
    input_dim: int,
    output_rows: int,
    block_size: int,
    states_per_value: int,
    table_entry_bytes: int,
) -> dict[str, int | float]:
    """Account for a table that returns one output partial sum per input block state."""
    if input_dim <= 0 or output_rows <= 0 or block_size <= 0:
        raise ValueError("dimensions and block_size must be positive")
    if input_dim % block_size:
        raise ValueError("input_dim must be divisible by block_size")
    if states_per_value < 2 or table_entry_bytes <= 0:
        raise ValueError("states_per_value must be at least two and entries must have positive size")

    input_blocks = input_dim // block_size
    states_per_block = states_per_value**block_size
    table_bytes = input_blocks * states_per_block * output_rows * table_entry_bytes
    runtime_read_bytes = input_blocks * output_rows * table_entry_bytes
    fp16_dense_bytes = input_dim * output_rows * 2
    return {
        "input_blocks": input_blocks,
        "states_per_value": states_per_value,
        "states_per_block": states_per_block,
        "table_entry_bytes": table_entry_bytes,
        "table_bytes": table_bytes,
        "runtime_table_read_bytes_per_token": runtime_read_bytes,
        "fp16_dense_bytes": fp16_dense_bytes,
        "runtime_read_reduction_vs_fp16_dense": 1.0 - runtime_read_bytes / fp16_dense_bytes,
    }


def exact_residual_bytes(*, rows: int, input_dim: int, residual_bits: int, metadata_bytes: int) -> dict[str, int | bool]:
    """Return the GPU payload for a lossless packed low-digit residual."""
    if rows <= 0 or input_dim <= 0 or residual_bits <= 0 or metadata_bytes < 0:
        raise ValueError("invalid residual dimensions")
    values = rows * input_dim
    packed = math.ceil(values * residual_bits / 8)
    return {
        "lossless": True,
        "residual_bits": residual_bits,
        "residual_values": values,
        "packed_residual_bytes": packed,
        "metadata_bytes": metadata_bytes,
        "total_gpu_payload_bytes": packed + metadata_bytes,
    }


def parse_ints(value: str) -> list[int]:
    result = [int(part) for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Account for exact residual and compiled main-term state-table layouts")
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--output-rows", type=int, default=6144)
    parser.add_argument("--block-sizes", default="1,2,4,8")
    parser.add_argument("--state-bits", default="2,3,4,8,16")
    parser.add_argument("--table-entry-bytes", type=int, default=2)
    parser.add_argument("--residual-bits", type=int, default=2)
    parser.add_argument(
        "--residual-metadata-bytes",
        type=int,
        default=1_572_864,
        help="exact fp32 alpha bytes for one 6144x2048 Q4_K projection by default",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for state_bits in parse_ints(args.state_bits):
        states_per_value = 1 << state_bits
        for block_size in parse_ints(args.block_sizes):
            if args.input_dim % block_size:
                continue
            row = state_table_ledger(
                input_dim=args.input_dim,
                output_rows=args.output_rows,
                block_size=block_size,
                states_per_value=states_per_value,
                table_entry_bytes=args.table_entry_bytes,
            )
            row["state_bits_per_value"] = state_bits
            row["table_gib"] = row["table_bytes"] / 2**30
            row["runtime_table_read_mib_per_token"] = row["runtime_table_read_bytes_per_token"] / 2**20
            rows.append(row)

    residual = exact_residual_bytes(
        rows=args.output_rows,
        input_dim=args.input_dim,
        residual_bits=args.residual_bits,
        metadata_bytes=args.residual_metadata_bytes,
    )
    result = {
        "experiment": "exact_main_state_table_and_full_residual_ledger",
        "contract": {
            "cold_start": "scan dense main weights once and materialize every state-table partial sum",
            "runtime_main": "one selected vector-table entry per input block; no dense main-weight stream scan",
            "runtime_residual": "lossless packed residual sent to GPU; no residual clipping",
            "exactness": "a finite state table is exact only for its represented activation states; an omitted x-state remainder is a separately declared error",
        },
        "dimensions": {"input_dim": args.input_dim, "output_rows": args.output_rows},
        "full_residual": residual,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "full_residual_bytes": residual["total_gpu_payload_bytes"]}))


if __name__ == "__main__":
    main()
