from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_polynomial_base_residual import (
    fit_poly,
    fit_residual_map,
    load_down,
    load_layer,
    poly_base,
    residual_features,
    rel_l2,
)


def coefficient_matrix(x: np.ndarray, model: dict) -> np.ndarray:
    z = (x - model["x_mu"]) @ model["input_basis"]
    features = residual_features(z, int(model["feature_degree"]))
    return features @ model["mapping"]


def masked_predict(coeff: np.ndarray, model: dict, keep: int) -> tuple[np.ndarray, float]:
    r = coeff.shape[1]
    k = min(max(int(keep), 0), r)
    if k == r:
        return model["output_mean"] + (coeff @ model["output_basis"].T), 1.0
    if k == 0:
        return np.broadcast_to(model["output_mean"], (len(coeff), len(model["output_mean"]))), 0.0
    order = np.argpartition(np.abs(coeff), -k, axis=1)[:, -k:]
    selected = np.take_along_axis(coeff, order, axis=1)
    selected_basis = model["output_basis"][:, order]
    # Each token has k columns; the gather is equivalent to a compact basis merge.
    pred = model["output_mean"] + np.sum(selected[:, :, None] * selected_basis.transpose(1, 2, 0), axis=1)
    return pred.astype(np.float32, copy=False), float(k / max(r, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-k sparse residual coefficient transfer experiment")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--base-degree", type=int, default=4)
    parser.add_argument("--input-rank", type=int, default=128)
    parser.add_argument("--output-rank", type=int, default=64)
    parser.add_argument("--feature-degree", type=int, default=1)
    parser.add_argument("--residual-target", choices=("exact", "capture"), default="exact")
    parser.add_argument("--keeps", default="0,1,2,4,8,16,32,64")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, train_g, train_u, train_h, train_capture_y = load_layer(args.calibration_root, args.layer)
    test_x, test_g, test_u, test_h, test_capture_y = load_layer(args.holdout_root, args.layer)
    down, down_bytes = load_down(args.model, args.layer)
    train_exact_y, test_exact_y = train_h @ down.T, test_h @ down.T
    poly = fit_poly(train_g, train_u, train_h, args.base_degree, args.ridge)
    base_train = poly_base(train_g, train_u, poly) @ down.T
    base_test = poly_base(test_g, test_u, poly) @ down.T
    residual_target_train = train_exact_y if args.residual_target == "exact" else train_capture_y
    residual_target_test = test_exact_y if args.residual_target == "exact" else test_capture_y
    residual_train, residual_test = residual_target_train - base_train, residual_target_test - base_test
    residual_model = fit_residual_map(
        train_x,
        residual_train,
        args.input_rank,
        args.output_rank,
        args.feature_degree,
        args.ridge,
    )
    train_coeff = coefficient_matrix(train_x, residual_model)
    test_coeff = coefficient_matrix(test_x, residual_model)
    output_rank = int(residual_model["output_basis"].shape[1])
    rows = []
    for keep in [int(v) for v in args.keeps.split(",") if v.strip()]:
        train_residual_hat, _ = masked_predict(train_coeff, residual_model, keep)
        test_residual_hat, _ = masked_predict(test_coeff, residual_model, keep)
        pred_train = base_train + train_residual_hat
        pred_test = base_test + test_residual_hat
        mask_bytes = int((output_rank + 7) // 8)
        value_bytes = int(min(max(keep, 0), output_rank) * 2)
        index_bytes = int(min(max(keep, 0), output_rank))
        rows.append({
            "keep": keep,
            "keep_fraction": float(min(max(keep, 0), output_rank) / max(output_rank, 1)),
            "train_rel_l2_vs_capture": float(rel_l2(pred_train, train_capture_y).mean()),
            "holdout_rel_l2_vs_capture": float(rel_l2(pred_test, test_capture_y).mean()),
            "train_rel_l2_vs_exact": float(rel_l2(pred_train, train_exact_y).mean()),
            "holdout_rel_l2_vs_exact": float(rel_l2(pred_test, test_exact_y).mean()),
            "transfer": {
                "gpu_basis_resident_fp16_bytes": int(output_rank * down.shape[0] * 2),
                "per_token_h2d_values_fp16_bytes": value_bytes,
                "per_token_h2d_bitmask_bytes": mask_bytes if keep < output_rank else 0,
                "per_token_h2d_indices_uint8_bytes": index_bytes if keep < output_rank else 0,
                "per_token_h2d_total_with_bitmask_bytes": value_bytes + (mask_bytes if keep < output_rank else 0),
                "per_token_h2d_total_with_indices_bytes": value_bytes + (index_bytes if keep < output_rank else 0),
                "per_token_cpu_base_output_fp16_bytes": int(down.shape[0] * 2),
                "full_down_weight_bytes_q4": down_bytes,
            },
            "extra_arithmetic": {
                "cpu_topk_selection_comparisons": output_rank,
                "gpu_compact_residual_merge_mac": int(down.shape[0] * min(max(keep, 0), output_rank)),
                "gpu_dense_masked_residual_merge_mac": int(down.shape[0] * output_rank),
            },
        })
    result = {
        "experiment": "topk_sparse_residual_coefficients",
        "formula": "y_hat = y_base + mu_r + sum_{i in TopK(|alpha(x)|)} alpha_i(x) U_i",
        "runtime_properties": {
            "lookup": False,
            "runtime_silu": False,
            "route": "CPU computes residual coefficients and selects top-k; GPU merges selected resident basis vectors",
            "fallback": "full exact FFN remains available when residual norm or OOD score exceeds threshold",
        },
        "layer": args.layer,
        "base_degree": args.base_degree,
        "input_rank": args.input_rank,
        "output_rank": output_rank,
        "feature_degree": args.feature_degree,
        "residual_target": args.residual_target,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "base_rel_l2_vs_capture_holdout": float(rel_l2(base_test, test_capture_y).mean()),
        "residual_energy_ratio_holdout": float(np.linalg.norm(residual_test, axis=1).mean() / max(np.linalg.norm(test_exact_y, axis=1).mean(), 1e-6)),
        "exact_weight_replay_rel_l2_holdout": float(rel_l2(test_exact_y, test_capture_y).mean()),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
