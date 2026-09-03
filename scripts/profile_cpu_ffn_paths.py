from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evaluate_activation_sensitive_weight_split import load_weights, silu
from evaluate_polynomial_base_residual import (
    chebyshev_base,
    fit_chebyshev,
    fit_residual_map,
    load_layer,
    residual_features,
)


def time_per_token(fn, values: np.ndarray, repeats: int) -> dict[str, float]:
    for x in values[: min(2, len(values))]:
        fn(x[None, :])
    samples = []
    for _ in range(repeats):
        for x in values:
            start = time.perf_counter_ns()
            fn(x[None, :])
            samples.append((time.perf_counter_ns() - start) / 1000.0)
    return {
        "median_us": float(np.median(samples)),
        "p95_us": float(np.percentile(samples, 95)),
        "mean_us": float(np.mean(samples)),
        "tokens_timed": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode-style CPU microbenchmark for exact and expanded FFN paths")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--chebyshev-bound", type=float, default=5.0)
    parser.add_argument("--input-rank", type=int, default=128)
    parser.add_argument("--output-rank", type=int, default=64)
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, train_g, train_u, train_h, train_capture_y = load_layer(args.calibration_root, args.layer)
    test_x, test_g, test_u, test_h, test_capture_y = load_layer(args.holdout_root, args.layer)
    (wg, wg_bytes), (wu, wu_bytes), (wd, wd_bytes) = load_weights(args.model, args.layer)
    poly = fit_chebyshev(train_g, train_u, train_h, args.degree, args.chebyshev_bound, 1e-2)
    base_train = chebyshev_base(train_g, train_u, poly) @ wd.T
    residual_model = fit_residual_map(
        train_x,
        train_capture_y - base_train,
        args.input_rank,
        args.output_rank,
        1,
        1e-2,
    )

    def exact(x: np.ndarray) -> np.ndarray:
        g = x @ wg.T
        u = x @ wu.T
        return (silu(g) * u) @ wd.T

    def base(x: np.ndarray) -> np.ndarray:
        g = x @ wg.T
        u = x @ wu.T
        return chebyshev_base(g, u, poly) @ wd.T

    def residual_coeff(x: np.ndarray) -> np.ndarray:
        z = (x - residual_model["x_mu"]) @ residual_model["input_basis"]
        features = residual_features(z, int(residual_model["feature_degree"]))
        return features @ residual_model["mapping"]

    def hybrid_cpu(x: np.ndarray) -> np.ndarray:
        return base(x), residual_coeff(x)

    values = test_x[: min(args.samples, len(test_x))].astype(np.float32, copy=False)
    timings = {
        "exact_ffn_cpu": time_per_token(exact, values, args.repeats),
        "chebyshev_base_cpu": time_per_token(base, values, args.repeats),
        "residual_coeff_cpu": time_per_token(residual_coeff, values, args.repeats),
        "hybrid_cpu_stage_base_plus_coeff": time_per_token(hybrid_cpu, values, args.repeats),
    }
    result = {
        "experiment": "cpu_ffn_path_microbenchmark",
        "layer": args.layer,
        "samples": len(values),
        "repeats": args.repeats,
        "configuration": {
            "base_basis": "chebyshev",
            "degree": args.degree,
            "chebyshev_bound_sigma": args.chebyshev_bound,
            "input_rank": args.input_rank,
            "output_rank": args.output_rank,
            "keep": args.keep,
        },
        "weights": {"gate_q4_bytes": wg_bytes, "up_q4_bytes": wu_bytes, "down_q4_bytes": wd_bytes},
        "timings_us_per_token": timings,
        "arithmetic_proxy": {
            "gate_projection_mac": int(values.shape[1] * train_h.shape[1]),
            "up_projection_mac": int(values.shape[1] * train_h.shape[1]),
            "down_projection_mac": int(train_h.shape[1] * wd.shape[0]),
            "base_total_projection_mac": int(3 * values.shape[1] * train_h.shape[1]),
            "residual_input_projection_mac": int(values.shape[1] * args.input_rank),
            "residual_coeff_merge_mac_if_gpu": int(wd.shape[0] * args.keep),
        },
        "note": "NumPy single-token timing on the local CPU; GPU merge and pinned H2D are not included.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
