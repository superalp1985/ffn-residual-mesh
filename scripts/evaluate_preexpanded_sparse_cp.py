from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

from evaluate_polynomial_base_residual import chebyshev_terms, fit_chebyshev, load_layer, rel_l2


def load_weights(model: Path, layer: int) -> tuple[tuple[np.ndarray, int], ...]:
    reader = GGUFReader(str(model))
    result = []
    for name in ("gate", "up", "down"):
        tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_{name}.weight")
        result.append((dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False), int(tensor.n_bytes)))
    return tuple(result)  # type: ignore[return-value]


def replay(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x @ weight.T


def nonlinear_scalars(g: np.ndarray, u: np.ndarray, model: dict) -> np.ndarray:
    z = (g - model["mu"]) / model["sigma"]
    terms = chebyshev_terms(z, int(model["degree"]), float(model["bound"]))
    nonconstant = np.sum(terms[:, :, 1:] * model["coeff"][None, :, 1:], axis=2)
    return (u * nonconstant).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-expanded linear base plus sparse CP FFN channels")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--chebyshev-bound", type=float, default=5.0)
    parser.add_argument("--ranks", default="64,128,256,512,1024,2048,4096,6144")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, train_g_capture, train_u_capture, train_h_capture, train_target = load_layer(
        args.calibration_root, args.layer
    )
    test_x, _, _, _, test_target = load_layer(args.holdout_root, args.layer)
    (wg, wg_bytes), (wu, wu_bytes), (wd, wd_bytes) = load_weights(args.model, args.layer)

    model = fit_chebyshev(
        train_g_capture,
        train_u_capture,
        train_h_capture,
        args.degree,
        args.chebyshev_bound,
        args.ridge,
    )
    train_g, train_u = replay(train_x, wg), replay(train_x, wu)
    test_g, test_u = replay(test_x, wg), replay(test_x, wu)
    train_nonlinear = nonlinear_scalars(train_g, train_u, model)
    test_nonlinear = nonlinear_scalars(test_g, test_u, model)

    constant = model["coeff"][:, 0]
    train_linear = (train_u * constant[None, :]) @ wd.T
    test_linear = (test_u * constant[None, :]) @ wd.T
    full_train = train_linear + train_nonlinear @ wd.T
    full_test = test_linear + test_nonlinear @ wd.T

    down_norm = np.linalg.norm(wd, axis=0)
    score = np.sqrt(np.mean(train_nonlinear * train_nonlinear, axis=0)) * down_norm
    order = np.argsort(score)[::-1]

    input_dim = train_x.shape[1]
    output_dim = wd.shape[0]
    ffn_dim = wd.shape[1]
    linear_matrix_fp16_bytes = input_dim * output_dim * 2
    original_q4_bytes = wg_bytes + wu_bytes + wd_bytes
    rows = []
    for requested_rank in [int(value) for value in args.ranks.split(",") if value.strip()]:
        rank = min(requested_rank, ffn_dim)
        selected = order[:rank]
        pred_train = train_linear + train_nonlinear[:, selected] @ wd[:, selected].T
        pred_test = test_linear + test_nonlinear[:, selected] @ wd[:, selected].T
        selected_factor_fp16_bytes = rank * (2 * input_dim + output_dim) * 2
        stats_bytes = rank * (2 + args.degree + 1) * 2
        artifact_bytes = linear_matrix_fp16_bytes + selected_factor_fp16_bytes + stats_bytes
        mac = input_dim * output_dim + 2 * input_dim * rank + output_dim * rank
        rows.append(
            {
                "rank": rank,
                "selected_fraction": rank / ffn_dim,
                "train_rel_l2_vs_capture": float(rel_l2(pred_train, train_target).mean()),
                "holdout_rel_l2_vs_capture": float(rel_l2(pred_test, test_target).mean()),
                "holdout_rel_l2_p95_vs_capture": float(np.percentile(rel_l2(pred_test, test_target), 95)),
                "artifact": {
                    "linear_matrix_fp16_bytes": linear_matrix_fp16_bytes,
                    "selected_factors_fp16_bytes": selected_factor_fp16_bytes,
                    "selected_stats_fp16_bytes": stats_bytes,
                    "total_fp16_bytes": artifact_bytes,
                    "ratio_vs_original_q4": artifact_bytes / original_q4_bytes,
                },
                "runtime": {
                    "mac_per_token": mac,
                    "ratio_vs_three_dense_projections": mac / (3 * input_dim * ffn_dim),
                },
            }
        )

    result = {
        "experiment": "preexpanded_linear_plus_sparse_cp_channels",
        "formula": "y = L0 x + Wd_S [u_S * sum_{p>=1} c_p T_p(g_S)], L0 = Wd diag(c0) Wu",
        "layer": args.layer,
        "degree": args.degree,
        "chebyshev_bound": args.chebyshev_bound,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "dimensions": {"input": input_dim, "ffn": ffn_dim, "output": output_dim},
        "original_q4_bytes": original_q4_bytes,
        "weight_replay_full_polynomial": {
            "train_rel_l2_vs_capture": float(rel_l2(full_train, train_target).mean()),
            "holdout_rel_l2_vs_capture": float(rel_l2(full_test, test_target).mean()),
            "holdout_rel_l2_p95_vs_capture": float(np.percentile(rel_l2(full_test, test_target), 95)),
        },
        "linear_only": {
            "train_rel_l2_vs_capture": float(rel_l2(train_linear, train_target).mean()),
            "holdout_rel_l2_vs_capture": float(rel_l2(test_linear, test_target).mean()),
            "holdout_rel_l2_p95_vs_capture": float(np.percentile(rel_l2(test_linear, test_target), 95)),
        },
        "selection": "RMS nonlinear scalar times down-column L2 norm on calibration data",
        "rows": rows,
        "caveat": (
            "Accuracy is evaluated in float32. The fp16 artifact size assumes L0 and selected Wg/Wu/Wd factors are "
            "stored after cold-start expansion; quantization and runtime kernels are not yet included."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
