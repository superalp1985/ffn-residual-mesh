from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evaluate_additive_holdout import collect


def rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def fit_formula(x: np.ndarray, y: np.ndarray, rank: int, degree: int, ridge: float) -> dict:
    center = x.mean(axis=0).astype(np.float32)
    dx = x - center
    # PCA basis is an offline layout artifact, not a runtime lookup table.
    _, _, vt = np.linalg.svd(dx, full_matrices=False)
    basis = vt[: min(rank, vt.shape[0])].T.astype(np.float32)
    z = dx @ basis
    features = [np.ones((len(x), 1), dtype=np.float32)]
    for power in range(1, degree + 1):
        features.append((z ** power).astype(np.float32))
    phi = np.concatenate(features, axis=1)
    gram = phi.T @ phi
    coef = np.linalg.solve(
        gram + np.eye(phi.shape[1], dtype=np.float32) * ridge,
        phi.T @ y,
    ).astype(np.float32)
    return {"center": center, "basis": basis, "coef": coef, "rank": int(basis.shape[1]), "degree": degree}


def features(x: np.ndarray, model: dict) -> np.ndarray:
    z = (x - model["center"]) @ model["basis"]
    parts = [np.ones((len(x), 1), dtype=np.float32)]
    for power in range(1, int(model["degree"]) + 1):
        parts.append((z ** power).astype(np.float32))
    return np.concatenate(parts, axis=1)


def predict(x: np.ndarray, model: dict) -> np.ndarray:
    return features(x, model) @ model["coef"]


def artifact_bytes(model: dict, dtype_bytes: int = 2) -> dict[str, int]:
    basis_elements = int(model["basis"].size)
    coef_elements = int(model["coef"].size)
    center_elements = int(model["center"].size)
    return {
        "center_fp16_bytes": center_elements * dtype_bytes,
        "basis_fp16_bytes": basis_elements * dtype_bytes,
        "merge_coef_fp16_bytes": coef_elements * dtype_bytes,
        "total_fp16_bytes": (center_elements + basis_elements + coef_elements) * dtype_bytes,
    }


def runtime_cost(model: dict, hidden: int) -> dict[str, int]:
    rank, degree = int(model["rank"]), int(model["degree"])
    feat = 1 + rank * degree
    # One projection and one output merge; powers are elementwise products.
    return {
        "projection_mac": hidden * rank,
        "merge_mac": hidden * feat,
        "feature_elementwise_mul": max(0, rank * (degree - 1)),
        "feature_width": feat,
    }


def profile(x: np.ndarray, model: dict, repeats: int) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        predict(x, model)
        values.append((time.perf_counter_ns() - t0) / len(x))
    return {
        "per_token_us_median": float(np.median(values) / 1000),
        "per_token_us_p95": float(np.percentile(values, 95) / 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-LUT nonlinear feature formula for a layered FFN")
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--degrees", default="1,2,3")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train, holdout = collect(args.calibration_root), collect(args.holdout_root)
    if args.layer not in train or args.layer not in holdout:
        raise SystemExit(f"layer {args.layer} was not captured in both datasets")
    train_x, _, train_y = train[args.layer]
    test_x, _, test_y = holdout[args.layer]
    ranks = [int(v) for v in args.ranks.split(",") if v.strip()]
    degrees = [int(v) for v in args.degrees.split(",") if v.strip()]
    rows = []
    for rank in ranks:
        for degree in degrees:
            model = fit_formula(train_x, train_y, rank, degree, args.ridge)
            train_pred, test_pred = predict(train_x, model), predict(test_x, model)
            train_err, test_err = rel_l2(train_pred, train_y), rel_l2(test_pred, test_y)
            rows.append({
                "rank": rank,
                "degree": degree,
                "train_rel_l2": float(train_err.mean()),
                "holdout_rel_l2": float(test_err.mean()),
                "holdout_rel_l2_p95": float(np.percentile(test_err, 95)),
                "artifact": artifact_bytes(model),
                "runtime_cost": runtime_cost(model, train_x.shape[1]),
                "cpu_profile": profile(test_x, model, args.profile_repeats),
            })
    result = {
        "experiment": "non_lut_nonlinear_feature_formula",
        "formula": "z = P.T @ (x-c); phi=[1,z,z^2,...,z^d]; y_hat = phi @ C",
        "interpretation": {
            "offline": "fit center, PCA-like basis P, and merge coefficients C per layer",
            "runtime": "project x, form elementwise polynomial residual features, and merge to hidden output",
            "no_lookup": True,
            "no_runtime_silu": True,
            "goal": "trade extra dense arithmetic for smaller resident/transfer artifacts",
        },
        "layer": args.layer,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "ridge": args.ridge,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
