from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_polynomial_base_residual import (
    chebyshev_base,
    fit_chebyshev,
    fit_poly,
    fit_residual_map,
    load_down,
    load_layer,
    poly_base,
    residual_features,
    rel_l2,
)


def coefficients(x: np.ndarray, model: dict) -> np.ndarray:
    z = (x - model["x_mu"]) @ model["input_basis"]
    return residual_features(z, int(model["feature_degree"])) @ model["mapping"]


def topk_residual(coeff: np.ndarray, model: dict, keep: int) -> tuple[np.ndarray, np.ndarray]:
    r = coeff.shape[1]
    k = min(max(int(keep), 0), r)
    if k == 0:
        return np.broadcast_to(model["output_mean"], (len(coeff), len(model["output_mean"]))), np.linalg.norm(coeff, axis=1)
    if k == r:
        return model["output_mean"] + coeff @ model["output_basis"].T, np.zeros(len(coeff), dtype=np.float32)
    order = np.argpartition(np.abs(coeff), -k, axis=1)[:, -k:]
    selected = np.take_along_axis(coeff, order, axis=1)
    selected_basis = model["output_basis"][:, order]
    pred = model["output_mean"] + np.sum(selected[:, :, None] * selected_basis.transpose(1, 2, 0), axis=1)
    tail = np.sqrt(np.maximum(np.sum(coeff * coeff, axis=1) - np.sum(selected * selected, axis=1), 0.0))
    return pred.astype(np.float32, copy=False), tail.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual tail-energy threshold and exact fallback simulation")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--base-degree", type=int, default=4)
    parser.add_argument("--input-rank", type=int, default=128)
    parser.add_argument("--output-rank", type=int, default=64)
    parser.add_argument("--feature-degree", type=int, default=1)
    parser.add_argument("--keeps", default="4,8,16,32")
    parser.add_argument("--quantiles", default="0.50,0.75,0.90,0.95,0.99,1.00")
    parser.add_argument("--residual-target", choices=("exact", "capture"), default="capture")
    parser.add_argument("--base-basis", choices=("monomial", "chebyshev"), default="monomial")
    parser.add_argument("--chebyshev-bound", type=float, default=6.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, train_g, train_u, train_h, train_capture_y = load_layer(args.calibration_root, args.layer)
    test_x, test_g, test_u, test_h, test_capture_y = load_layer(args.holdout_root, args.layer)
    down, down_bytes = load_down(args.model, args.layer)
    train_exact_y, test_exact_y = train_h @ down.T, test_h @ down.T
    train_target = train_exact_y if args.residual_target == "exact" else train_capture_y
    test_target = test_exact_y if args.residual_target == "exact" else test_capture_y

    if args.base_basis == "chebyshev":
        poly = fit_chebyshev(train_g, train_u, train_h, args.base_degree, args.chebyshev_bound, args.ridge)
        base_train = chebyshev_base(train_g, train_u, poly) @ down.T
        base_test = chebyshev_base(test_g, test_u, poly) @ down.T
    else:
        poly = fit_poly(train_g, train_u, train_h, args.base_degree, args.ridge)
        base_train = poly_base(train_g, train_u, poly) @ down.T
        base_test = poly_base(test_g, test_u, poly) @ down.T
    residual_model = fit_residual_map(
        train_x,
        train_target - base_train,
        args.input_rank,
        args.output_rank,
        args.feature_degree,
        args.ridge,
    )
    train_coeff, test_coeff = coefficients(train_x, residual_model), coefficients(test_x, residual_model)
    rows = []
    output_rank = int(residual_model["output_basis"].shape[1])
    for keep in [int(v) for v in args.keeps.split(",") if v.strip()]:
        train_approx_residual, train_tail = topk_residual(train_coeff, residual_model, keep)
        test_approx_residual, test_tail = topk_residual(test_coeff, residual_model, keep)
        train_approx, test_approx = base_train + train_approx_residual, base_test + test_approx_residual
        for quantile in [float(v) for v in args.quantiles.split(",") if v.strip()]:
            threshold = float(np.quantile(train_tail, quantile))
            train_approx_mask, test_approx_mask = train_tail <= threshold, test_tail <= threshold
            mixed_train = np.where(train_approx_mask[:, None], train_approx, train_target)
            mixed_test = np.where(test_approx_mask[:, None], test_approx, test_target)
            approx_fraction = float(test_approx_mask.mean())
            fallback_fraction = 1.0 - approx_fraction
            mask_bytes = int((output_rank + 7) // 8)
            value_bytes = int(min(max(keep, 0), output_rank) * 2)
            approx_bytes = int(down.shape[0] * 2 + value_bytes + (mask_bytes if keep < output_rank else 0))
            expected_h2d = approx_fraction * approx_bytes + fallback_fraction * down_bytes
            rows.append({
                "keep": keep,
                "train_quantile": quantile,
                "tail_threshold": threshold,
                "approx_fraction_holdout": approx_fraction,
                "fallback_fraction_holdout": fallback_fraction,
                "mixed_rel_l2_vs_capture_holdout": float(rel_l2(mixed_test, test_capture_y).mean()),
                "mixed_rel_l2_vs_exact_holdout": float(rel_l2(mixed_test, test_exact_y).mean()),
                "approx_rel_l2_vs_capture_holdout": float(rel_l2(test_approx, test_capture_y).mean()),
                "approx_rel_l2_vs_exact_holdout": float(rel_l2(test_approx, test_exact_y).mean()),
                "transfer": {
                    "gpu_output_mean_resident_fp16_bytes": int(down.shape[0] * 2),
                    "approx_path_h2d_base_plus_coeff_plus_bitmask_bytes": approx_bytes,
                    "fallback_path_h2d_full_down_weight_bytes_q4": down_bytes,
                    "expected_h2d_bytes_per_token": expected_h2d,
                    "reduction_vs_full_down_weight": 1.0 - expected_h2d / max(down_bytes, 1),
                },
            })
    result = {
        "experiment": "residual_tail_threshold_exact_fallback",
        "formula": "route approximate if ||alpha(x)-TopK(alpha(x))||_2 <= threshold; otherwise exact FFN",
        "runtime_properties": {
            "lookup": False,
            "runtime_silu": False,
            "threshold_source": "calibration tail-energy quantile",
            "fallback": "exact target output in simulation; deployment retains exact FFN path",
        },
        "layer": args.layer,
        "base_degree": args.base_degree,
        "base_basis": args.base_basis,
        "chebyshev_bound": args.chebyshev_bound if args.base_basis == "chebyshev" else None,
        "input_rank": args.input_rank,
        "output_rank": output_rank,
        "feature_degree": args.feature_degree,
        "residual_target": args.residual_target,
        "base_rel_l2_vs_capture_holdout": float(rel_l2(base_test, test_capture_y).mean()),
        "exact_weight_replay_rel_l2_holdout": float(rel_l2(test_exact_y, test_capture_y).mean()),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
