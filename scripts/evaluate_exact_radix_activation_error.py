from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np


def ffn_forward(x: np.ndarray, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray) -> np.ndarray:
    gate = x @ wg.T
    up = x @ wu.T
    hidden = (gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0))) * up).astype(np.float32, copy=False)
    return hidden @ wd.T


def relative_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32)
    pred = np.asarray(candidate, dtype=np.float32)
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-12))


def quantize_groupwise_int8(x: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % group_size:
        raise ValueError("x must have shape [tokens, hidden] and complete groups")
    grouped = values.reshape(values.shape[0], -1, group_size)
    scales = np.maximum(np.max(np.abs(grouped), axis=2) / np.float32(127.0), np.float32(1e-12))
    codes = np.clip(np.rint(grouped / scales[:, :, None]), -128, 127).astype(np.int8)
    reconstructed = codes.astype(np.float32) * scales[:, :, None]
    return reconstructed.reshape(values.shape), scales.astype(np.float32, copy=False)


def load_layer(root: Path, layer: int) -> np.ndarray:
    from evaluate_polynomial_base_residual import load_layer as load_capture_layer

    return load_capture_layer(root, layer)[0].astype(np.float32, copy=False)


def load_weights(model: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from evaluate_preexpanded_sparse_cp import load_weights as load_model_weights

    return tuple(value[0].astype(np.float32, copy=False) for value in load_model_weights(model, layer))  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure groupwise activation quantization error through exact Q4-dequantized FFN")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--group-sizes", default="4,8,16,32,64")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wg, wu, wd = load_weights(args.model, args.layer)
    rows = []
    for dataset_name, root in (("calibration", args.calibration_root), ("holdout", args.holdout_root)):
        x = load_layer(root, args.layer)
        exact = ffn_forward(x, wg, wu, wd)
        for group_size in [int(value) for value in args.group_sizes.split(",") if value.strip()]:
            quantized_x, scales = quantize_groupwise_int8(x, group_size)
            quantized_ffn = ffn_forward(quantized_x, wg, wu, wd)
            rows.append(
                {
                    "dataset": dataset_name,
                    "samples": len(x),
                    "group_size": group_size,
                    "input_rel_l2": relative_l2_error(x, quantized_x),
                    "ffn_rel_l2": relative_l2_error(exact, quantized_ffn),
                    "ffn_max_abs": float(np.max(np.abs(exact - quantized_ffn))),
                    "scale_bytes_per_token": int(scales.shape[1] * 4),
                }
            )

    result = {
        "experiment": "exact_radix_activation_quantization_error",
        "date": "2026-09-04",
        "platform": platform.platform(),
        "layer": args.layer,
        "formula": "q = 4*q_hi + q_lo exact for int8 activation codes; this experiment isolates x -> int8 state error",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
