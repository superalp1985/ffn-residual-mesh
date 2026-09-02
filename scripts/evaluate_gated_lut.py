from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_additive_holdout import apply, collect, fit


def apply_with_distance(values: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    codes, distance = [], np.zeros(len(values), dtype=np.float32)
    block_dim = model["block_dim"]
    for start, centers in zip(range(0, values.shape[1], block_dim), model["centers"]):
        block = values[:, start:start + block_dim]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
        code = np.argmin(d2, axis=1)
        codes.append(code)
        distance += np.sqrt(np.take_along_axis(d2, code[:, None], axis=1)[:, 0])
    pred = model["bias"] + sum(table[code] for table, code in zip(model["tables"], codes))
    return pred, distance / max(len(model["centers"]), 1)


def error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=Path)
    parser.add_argument("test_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--layers", default="0,10,18,22,23")
    args = parser.parse_args()
    wanted = {int(x) for x in args.layers.split(",") if x.strip()}
    train, test = collect(args.train_root), collect(args.test_root)
    rows = []
    fractions = (0.10, 0.25, 0.50, 0.75, 1.00)
    for layer in sorted(wanted & train.keys() & test.keys()):
        _, th, ty = train[layer]
        _, vh, vy = test[layer]
        model = fit(th, ty, args.block, args.k, 5000 + layer)
        pred, distance = apply_with_distance(vh, model)
        errors = error(pred, vy)
        order = np.argsort(distance)
        gates = []
        for fraction in fractions:
            accepted = max(1, int(round(len(vh) * fraction)))
            idx = order[:accepted]
            gates.append({
                "approx_fraction": fraction,
                "fallback_fraction": 1.0 - fraction,
                "accepted_rel_l2": float(errors[idx].mean()),
                "accepted_rel_l2_p95": float(np.percentile(errors[idx], 95)),
                "expected_full_output_rel_l2": float(errors[idx].sum() / len(errors)),
                "distance_threshold": float(distance[idx[-1]]),
            })
        rows.append({"layer": layer, "test_samples": len(vh), "distance_error_corr": float(np.corrcoef(distance, errors)[0, 1]), "gates": gates})
    result = {"train_root": str(args.train_root), "test_root": str(args.test_root), "k": args.k, "block": args.block, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
