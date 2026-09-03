from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_polynomial_base_residual import load_down, load_layer, rel_l2


def chebyshev_features(z: np.ndarray, degree: int, bound: float) -> np.ndarray:
    t = np.clip(z / np.float32(bound), -1.0, 1.0)
    terms = [np.ones_like(t, dtype=np.float32)]
    if degree >= 1:
        terms.append(t)
    for _ in range(2, degree + 1):
        terms.append((2.0 * t * terms[-1] - terms[-2]).astype(np.float32, copy=False))
    return np.stack(terms, axis=2)


def fit_chebyshev(g: np.ndarray, u: np.ndarray, h: np.ndarray, degree: int, bound: float, ridge: float) -> dict:
    mu = g.mean(axis=0).astype(np.float32)
    sigma = np.maximum(g.std(axis=0).astype(np.float32), np.float32(1e-3))
    z = (g - mu) / sigma
    basis = chebyshev_features(z, degree, bound)
    coeff = np.zeros((g.shape[1], degree + 1), dtype=np.float32)
    eye = np.eye(degree + 1, dtype=np.float32) * ridge
    for j in range(g.shape[1]):
        phi = u[:, j, None] * basis[:, j, :]
        coeff[j] = np.linalg.solve(phi.T @ phi + eye, phi.T @ h[:, j])
    return {"mu": mu, "sigma": sigma, "bound": bound, "degree": degree, "coeff": coeff}


def predict(g: np.ndarray, u: np.ndarray, model: dict) -> np.ndarray:
    z = (g - model["mu"]) / model["sigma"]
    basis = chebyshev_features(z, int(model["degree"]), float(model["bound"]))
    return np.sum(u[:, :, None] * basis * model["coeff"][None, :, :], axis=2).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chebyshev SwiGLU base approximation")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--degrees", default="2,3,4,5")
    parser.add_argument("--bounds", default="1.5,2,3,4,6")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _, train_g, train_u, train_h, train_y = load_layer(args.calibration_root, args.layer)
    _, test_g, test_u, test_h, test_y = load_layer(args.holdout_root, args.layer)
    down, _ = load_down(args.model, args.layer)
    rows = []
    for degree in [int(v) for v in args.degrees.split(",") if v.strip()]:
        for bound in [float(v) for v in args.bounds.split(",") if v.strip()]:
            model = fit_chebyshev(train_g, train_u, train_h, degree, bound, args.ridge)
            train_h_hat, test_h_hat = predict(train_g, train_u, model), predict(test_g, test_u, model)
            train_y_hat, test_y_hat = train_h_hat @ down.T, test_h_hat @ down.T
            rows.append({
                "degree": degree,
                "bound_sigma": bound,
                "activation_rel_l2_train": float(rel_l2(train_h_hat, train_h).mean()),
                "activation_rel_l2_holdout": float(rel_l2(test_h_hat, test_h).mean()),
                "output_rel_l2_train": float(rel_l2(train_y_hat, train_y).mean()),
                "output_rel_l2_holdout": float(rel_l2(test_y_hat, test_y).mean()),
                "artifact_fp16_bytes": int((train_g.shape[1] * (degree + 1) + train_g.shape[1] * 2) * 2),
                "runtime_elementwise_mul_per_neuron": int(max(degree, 0)),
            })
    result = {
        "experiment": "chebyshev_swiglu_base",
        "formula": "h_j = u_j * sum_p c[j,p] T_p(clip((g_j-mu_j)/(sigma_j*bound),-1,1))",
        "runtime_properties": {"lookup": False, "runtime_silu": False, "evaluation": "Clenshaw-equivalent recurrence; no table"},
        "layer": args.layer,
        "train_samples": len(train_g),
        "holdout_samples": len(test_g),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
