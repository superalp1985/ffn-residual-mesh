from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from gguf import GGUFReader
from gguf.quants import dequantize
from evaluate_polynomial_base_residual import (
    chebyshev_base,
    fit_chebyshev,
    fit_residual_map,
    load_layer,
    residual_features,
)


def load_weights(model: Path, layer: int) -> tuple[tuple[np.ndarray, int], ...]:
    reader = GGUFReader(str(model))
    result = []
    for name in ("gate", "up", "down"):
        tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_{name}.weight")
        result.append((dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False), int(tensor.n_bytes)))
    return tuple(result)  # type: ignore[return-value]


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def benchmark(fn, values: np.ndarray, batch: int, repeats: int, warmup: int) -> dict[str, float]:
    sample = values[:batch]
    for _ in range(warmup):
        fn(sample)
    elapsed_us = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn(sample)
        elapsed_us.append((time.perf_counter_ns() - start) / 1000.0)
    return {
        "median_us_per_batch": float(np.median(elapsed_us)),
        "p95_us_per_batch": float(np.percentile(elapsed_us, 95)),
        "median_us_per_token": float(np.median(elapsed_us) / batch),
        "p95_us_per_token": float(np.percentile(elapsed_us, 95) / batch),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU FFN pre-expansion batch scaling benchmark")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--chebyshev-bound", type=float, default=5.0)
    parser.add_argument("--input-rank", type=int, default=128)
    parser.add_argument("--output-rank", type=int, default=64)
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--batches", default="1,4,16,64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    batches = [int(value) for value in args.batches.split(",") if value.strip()]
    train_x, train_g, train_u, train_h, train_capture = load_layer(args.calibration_root, args.layer)
    test_x, _, _, _, _ = load_layer(args.holdout_root, args.layer)
    (wg, wg_bytes), (wu, wu_bytes), (wd, wd_bytes) = load_weights(args.model, args.layer)
    poly = fit_chebyshev(train_g, train_u, train_h, args.degree, args.chebyshev_bound, 1e-2)
    base_train = chebyshev_base(train_g, train_u, poly) @ wd.T
    residual_model = fit_residual_map(
        train_x,
        train_capture - base_train,
        args.input_rank,
        args.output_rank,
        1,
        1e-2,
    )

    def exact(x: np.ndarray) -> np.ndarray:
        gate = x @ wg.T
        up = x @ wu.T
        return (silu(gate) * up) @ wd.T

    def base(x: np.ndarray) -> np.ndarray:
        gate = x @ wg.T
        up = x @ wu.T
        return chebyshev_base(gate, up, poly) @ wd.T

    def residual_coeff(x: np.ndarray) -> np.ndarray:
        z = (x - residual_model["x_mu"]) @ residual_model["input_basis"]
        return residual_features(z, int(residual_model["feature_degree"])) @ residual_model["mapping"]

    def packet(x: np.ndarray) -> bytes:
        base16 = np.ascontiguousarray(base(x).astype("<f2", copy=False))
        coeff = residual_coeff(x)
        indices = np.argpartition(np.abs(coeff), -args.keep, axis=1)[:, -args.keep:]
        values16 = np.ascontiguousarray(np.take_along_axis(coeff, indices, axis=1).astype("<f2", copy=False))
        indices16 = np.ascontiguousarray(indices.astype("<u2", copy=False))
        return b"".join((base16.tobytes(), values16.tobytes(), indices16.tobytes()))

    values = test_x.astype(np.float32, copy=False)
    if max(batches) > len(values):
        raise ValueError(f"largest batch {max(batches)} exceeds holdout samples {len(values)}")

    rows = []
    for batch in batches:
        timings = {
            "exact_ffn": benchmark(exact, values, batch, args.repeats, args.warmup),
            "chebyshev_base": benchmark(base, values, batch, args.repeats, args.warmup),
            "residual_coeff": benchmark(residual_coeff, values, batch, args.repeats, args.warmup),
            "full_packet": benchmark(packet, values, batch, args.repeats, args.warmup),
        }
        base_us = timings["chebyshev_base"]["median_us_per_token"]
        packet_us = timings["full_packet"]["median_us_per_token"]
        rows.append(
            {
                "batch": batch,
                "scenario": "single-stream decode" if batch == 1 else "prefill or concurrent sequences",
                "timings": timings,
                "single_layer_tokens_per_second_if_serial": 1_000_000.0 / packet_us,
                "twenty_four_layer_proxy_ms_per_token": packet_us * 24 / 1000.0,
                "twenty_four_layer_proxy_tokens_per_second": 1_000_000.0 / (packet_us * 24),
                "packet_overhead_us_per_token": packet_us - base_us,
            }
        )

    result = {
        "experiment": "cpu_ffn_preexpansion_batch_scaling",
        "layer": args.layer,
        "configuration": {
            "degree": args.degree,
            "chebyshev_bound": args.chebyshev_bound,
            "input_rank": args.input_rank,
            "output_rank": args.output_rank,
            "keep": args.keep,
            "packet_bytes_per_token": int(wd.shape[0] * 2 + args.keep * 4),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "weights": {
            "gate_q4_bytes": wg_bytes,
            "up_q4_bytes": wu_bytes,
            "down_q4_bytes": wd_bytes,
        },
        "rows": rows,
        "caveat": (
            "Batch 1 is the relevant lower bound for one autoregressive decode stream. Larger batches only model "
            "prefill or independent concurrent sequences. The 24-layer value repeats layer 23 and is a capacity proxy, "
            "not an end-to-end model benchmark."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
