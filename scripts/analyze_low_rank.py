from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rank_errors(values: np.ndarray, ranks: list[int]) -> dict[str, float]:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    total = max(float(energy.sum()), 1e-12)
    cumulative = np.concatenate(([0.0], np.cumsum(energy) / total))
    return {str(rank): float(1.0 - cumulative[min(rank, len(singular))]) for rank in ranks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.probe_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
    ranks = [4, 8, 16, 32, 64]
    rows = []
    for item in manifest["tensors"]:
        if not item["name"].startswith(("ffn_swiglu-", "ffn_out-")):
            continue
        raw = np.fromfile(args.probe_dir / item["file"], dtype=np.float32)
        shape = tuple(int(x) for x in item["shape"])
        values = raw.reshape(shape[::-1])
        delta = np.diff(values, axis=0)
        rows.append({
            "name": item["name"],
            "shape": list(values.shape),
            "full_rank_error": rank_errors(values, ranks),
            "delta_rank_error": rank_errors(delta, ranks) if len(delta) > 1 else {},
        })

    result = {"ranks": ranks, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for prefix in ("ffn_swiglu", "ffn_out"):
        subset = [r for r in rows if r["name"].startswith(prefix)]
        print(prefix, {rank: round(float(np.mean([r["full_rank_error"][str(rank)] for r in subset])), 6) for rank in ranks})
        print(prefix + ".delta", {rank: round(float(np.mean([r["delta_rank_error"][str(rank)] for r in subset])), 6) for rank in ranks})


if __name__ == "__main__":
    main()
