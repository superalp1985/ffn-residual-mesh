from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from evaluate_preexpanded_sparse_cp import load_weights


def fwht(x: torch.Tensor) -> torch.Tensor:
    """In-place-friendly Walsh-Hadamard transform over dim 0."""
    n = x.shape[0]
    h = 1
    y = x
    while h < n:
        y = y.reshape(n // (2 * h), 2, h, *y.shape[1:])
        a = y[:, 0].clone()
        b = y[:, 1].clone()
        y[:, 0] = a + b
        y[:, 1] = a - b
        y = y.reshape(n, *y.shape[3:])
        h *= 2
    return y / (n**0.5)


def scan(weight: np.ndarray, ks: list[int], device: torch.device) -> dict:
    rows, cols = weight.shape
    n = 1
    while n < rows:
        n *= 2
    padded = torch.zeros((n, cols), dtype=torch.float32, device=device)
    padded[:rows] = torch.from_numpy(weight).to(device)
    transformed = fwht(padded)
    energy = transformed.square()
    total = energy.sum(dim=0).clamp_min(1e-12)
    flat = energy.sort(dim=0, descending=True).values
    rows_out = []
    for k in ks:
        kk = min(k, n)
        captured = flat[:kk].sum(dim=0) / total
        rows_out.append(
            {
                "top_k": kk,
                "energy_fraction_mean": float(captured.mean()),
                "energy_fraction_p50": float(captured.quantile(0.50)),
                "energy_fraction_p90": float(captured.quantile(0.90)),
                "energy_fraction_p99": float(captured.quantile(0.99)),
                "sparse_fp16_index16_bytes": int(cols * kk * 4),
                "sparse_int4_index16_lower_bound_bytes": int(np.ceil(cols * kk * 3 / 2)),
                "dense_q4k_bytes": int(rows * cols * 18 / 32),
            }
        )
    return {
        "rows": rows,
        "columns": cols,
        "padded_transform_rows": n,
        "rows": rows,
        "scan": rows_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan output-channel Walsh-Hadamard energy compaction")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--ks", default="32,64,128,256,512,1024,2048,4096")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projections = {}
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    for projection in ("gate", "up"):
        (weight, q4_bytes), *_ = [load_weights(args.model, args.layer)[0 if projection == "gate" else 1]]
        projections[projection] = {
            "q4k_bytes": q4_bytes,
            "result": scan(weight, ks, device),
        }
    result = {
        "experiment": "output_channel_hadamard_sparsity",
        "scope": "fixed orthogonal transform probe; no runtime kernel or transfer benchmark",
        "model": str(args.model),
        "layer": args.layer,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "projections": projections,
        "interpretation": {
            "basis": "Walsh-Hadamard basis is implicit; only transformed coefficients would be stored.",
            "caveat": "A transform helps traffic only if sparse coefficients plus inverse transform cost beat the original tile path.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
