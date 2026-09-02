from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_additive_holdout import collect, fit
from evaluate_gated_lut import apply_with_distance


def errors(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def select_threshold(distance: np.ndarray, error: np.ndarray, budget: float) -> tuple[float, float, float]:
    order = np.argsort(distance)
    best = (0.0, 0.0, None)
    for n in range(1, len(order) + 1):
        accepted = error[order[:n]]
        p95 = float(np.percentile(accepted, 95))
        if p95 <= budget:
            best = (float(distance[order[n - 1]]), float(n / len(order)), p95)
    return best


def evaluate_threshold(distance: np.ndarray, error: np.ndarray, threshold: float) -> dict[str, float]:
    accepted = distance <= threshold
    if not np.any(accepted):
        return {"approx_fraction": 0.0, "fallback_fraction": 1.0, "accepted_rel_l2": 0.0, "accepted_rel_l2_p95": 0.0, "expected_rel_l2": 0.0}
    ae = error[accepted]
    return {
        "approx_fraction": float(accepted.mean()),
        "fallback_fraction": float((~accepted).mean()),
        "accepted_rel_l2": float(ae.mean()),
        "accepted_rel_l2_p95": float(np.percentile(ae, 95)),
        "expected_rel_l2": float(ae.sum() / len(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default="0,10,18,22,23")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--budgets", default="0.25,0.50")
    args = parser.parse_args()
    wanted = {int(x) for x in args.layers.split(",") if x.strip()}
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    calibration, holdout = collect(args.calibration_root), collect(args.holdout_root)
    rows = []
    for layer in sorted(wanted & calibration.keys() & holdout.keys()):
        _, train_h, train_y = calibration[layer]
        _, test_h, test_y = holdout[layer]
        model = fit(train_h, train_y, args.block, args.k, 13000 + layer)
        train_pred, train_dist = apply_with_distance(train_h, model)
        test_pred, test_dist = apply_with_distance(test_h, model)
        train_error, test_error = errors(train_pred, train_y), errors(test_pred, test_y)
        policy = []
        for budget in budgets:
            threshold, train_coverage, train_p95 = select_threshold(train_dist, train_error, budget)
            policy.append({
                "budget_p95": budget,
                "distance_threshold": threshold,
                "train_approx_fraction": train_coverage,
                "train_accepted_p95": train_p95,
                "holdout": evaluate_threshold(test_dist, test_error, threshold),
            })
        rows.append({"layer": layer, "train_samples": len(train_h), "holdout_samples": len(test_h), "policy": policy})
    result = {"calibration_root": str(args.calibration_root), "holdout_root": str(args.holdout_root), "k": args.k, "block": args.block, "rows": rows, "runtime_rule": "accept approximate FFN only when per-block mean centroid distance <= layer threshold; otherwise run exact FFN"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
