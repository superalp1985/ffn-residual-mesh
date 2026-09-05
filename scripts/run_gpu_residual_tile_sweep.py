from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_exact_radix_split_pipeline import (
    gpu_residual_benchmark,
    load_q4_projection,
    pack_2bit_rows,
    quantize_groupwise_q8,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep exact 2-bit residual tile sizes")
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--tile-rows", default="1024,6144")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    x = np.fromfile(args.input, dtype="<f4").reshape(1, -1)
    z, scales = quantize_groupwise_q8(x, group_size=32)
    rows = []
    for projection in ("gate", "up"):
        codes, alpha, _, _ = load_q4_projection(args.model, args.layer, projection)
        packed = pack_2bit_rows((codes & 3).reshape(codes.shape[0], -1))
        for tile_rows in (int(value) for value in args.tile_rows.split(",") if value.strip()):
            _, profile = gpu_residual_benchmark(
                packed,
                alpha,
                z,
                scales,
                tile_rows,
                warmup=1,
                repeats=args.repeats,
            )
            rows.append({"projection": projection, **profile})

    result = {
        "experiment": "gpu_exact_residual_tile_sweep",
        "layer": args.layer,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
