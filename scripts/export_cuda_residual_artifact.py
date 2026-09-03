from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from evaluate_polynomial_base_residual import (
    chebyshev_base,
    fit_chebyshev,
    fit_residual_map,
    load_down,
    load_layer,
    rel_l2,
    residual_features,
)


MAGIC = b"FFNRES01"
VERSION = 1


def coefficients(x: np.ndarray, model: dict) -> np.ndarray:
    z = (x - model["x_mu"]) @ model["input_basis"]
    return residual_features(z, int(model["feature_degree"])) @ model["mapping"]


def topk(coeff: np.ndarray, keep: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(max(int(keep), 0), coeff.shape[1])
    if k == 0:
        return np.empty((len(coeff), 0), dtype=np.float32), np.empty((len(coeff), 0), dtype=np.uint16)
    indices = np.argpartition(np.abs(coeff), -k, axis=1)[:, -k:]
    values = np.take_along_axis(coeff, indices, axis=1)
    return values.astype(np.float32, copy=False), indices.astype(np.uint16, copy=False)


def quantized_reference(
    base: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    output_mean: np.ndarray,
    output_basis: np.ndarray,
) -> np.ndarray:
    base16 = base.astype(np.float16)
    values16 = values.astype(np.float16)
    mean16 = output_mean.astype(np.float16)
    basis16 = output_basis.T.astype(np.float16)
    pred = base16.astype(np.float32) + mean16.astype(np.float32)[None, :]
    for row in range(len(base16)):
        for slot in range(values16.shape[1]):
            pred[row] += np.float32(values16[row, slot]) * basis16[indices[row, slot]].astype(np.float32)
    return pred


def write_artifact(
    path: Path,
    layer: int,
    base: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    output_mean: np.ndarray,
    output_basis: np.ndarray,
    capture_target: np.ndarray,
    reference: np.ndarray,
) -> None:
    tokens, hidden = base.shape
    rank = output_basis.shape[1]
    keep = values.shape[1]
    header = struct.pack(
        "<8s7I",
        MAGIC,
        VERSION,
        layer,
        tokens,
        hidden,
        rank,
        keep,
        0,
    )
    payloads = (
        output_basis.T.astype("<f2", copy=False),
        output_mean.astype("<f2", copy=False),
        base.astype("<f2", copy=False),
        values.astype("<f2", copy=False),
        indices.astype("<u2", copy=False),
        capture_target.astype("<f4", copy=False),
        reference.astype("<f4", copy=False),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        for payload in payloads:
            handle.write(np.ascontiguousarray(payload).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a real FFN residual packet artifact for the CUDA probe")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--base-degree", type=int, default=5)
    parser.add_argument("--chebyshev-bound", type=float, default=5.0)
    parser.add_argument("--input-rank", type=int, default=128)
    parser.add_argument("--output-rank", type=int, default=64)
    parser.add_argument("--feature-degree", type=int, default=1)
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, train_g, train_u, train_h, train_capture = load_layer(args.calibration_root, args.layer)
    test_x, test_g, test_u, test_h, test_capture = load_layer(args.holdout_root, args.layer)
    down, down_bytes = load_down(args.model, args.layer)

    base_model = fit_chebyshev(
        train_g,
        train_u,
        train_h,
        args.base_degree,
        args.chebyshev_bound,
        args.ridge,
    )
    base_train = chebyshev_base(train_g, train_u, base_model) @ down.T
    base_test = chebyshev_base(test_g, test_u, base_model) @ down.T
    residual_model = fit_residual_map(
        train_x,
        train_capture - base_train,
        args.input_rank,
        args.output_rank,
        args.feature_degree,
        args.ridge,
    )
    train_coeff = coefficients(train_x, residual_model)
    test_coeff = coefficients(test_x, residual_model)
    values, indices = topk(test_coeff, args.keep)
    reference = quantized_reference(
        base_test,
        values,
        indices,
        residual_model["output_mean"],
        residual_model["output_basis"],
    )
    float_pred = base_test + residual_model["output_mean"]
    for row in range(len(test_x)):
        float_pred[row] += (
            residual_model["output_basis"][:, indices[row]] @ values[row]
        ).astype(np.float32, copy=False)

    write_artifact(
        args.artifact,
        args.layer,
        base_test,
        values,
        indices,
        residual_model["output_mean"],
        residual_model["output_basis"],
        test_capture,
        reference,
    )

    selected_train, _ = topk(train_coeff, args.keep)
    result = {
        "experiment": "real_layer_cuda_residual_artifact",
        "formula": "y_hat = fp16(base) + fp16(mu_r) + sum_i fp16(alpha_i) * fp16(U_i)",
        "layer": args.layer,
        "base_degree": args.base_degree,
        "chebyshev_bound": args.chebyshev_bound,
        "input_rank": args.input_rank,
        "output_rank": args.output_rank,
        "feature_degree": args.feature_degree,
        "keep": args.keep,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "artifact": str(args.artifact),
        "artifact_bytes": args.artifact.stat().st_size,
        "down_weight_bytes_q4": down_bytes,
        "resident_fp16_bytes": int((args.output_rank + 1) * test_capture.shape[1] * 2),
        "packet_payload_bytes_per_token": int(test_capture.shape[1] * 2 + args.keep * 4),
        "float32_formula_rel_l2_vs_capture_holdout": float(rel_l2(float_pred, test_capture).mean()),
        "fp16_artifact_rel_l2_vs_capture_holdout": float(rel_l2(reference, test_capture).mean()),
        "fp16_artifact_rel_l2_p95_vs_capture_holdout": float(np.percentile(rel_l2(reference, test_capture), 95)),
        "fp16_vs_float_formula_rel_l2_holdout": float(rel_l2(reference, float_pred).mean()),
        "train_topk_abs_coeff_mean": float(np.abs(selected_train).mean()),
        "holdout_topk_abs_coeff_mean": float(np.abs(values).mean()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
