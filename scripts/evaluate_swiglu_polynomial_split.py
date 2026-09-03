from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

from evaluate_additive_holdout import load_tensor


def load_layer(root: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    wanted = [f"attn_post_norm-{layer}", f"ffn_gate-{layer}", f"ffn_up-{layer}", f"ffn_swiglu-{layer}", f"ffn_out-{layer}"]
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: load_tensor(prompt_dir, item) for item in manifest["tensors"] if item["name"] in wanted}
        if all(name in tensors for name in wanted):
            rows.append(tuple(tensors[name] for name in wanted))
    if not rows:
        raise ValueError(f"layer {layer} missing from {root}")
    return tuple(np.concatenate([row[i] for row in rows], axis=0) for i in range(5))  # type: ignore[return-value]


def load_down(model: Path, layer: int) -> np.ndarray:
    reader = GGUFReader(str(model))
    tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_down.weight")
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)


def fit_polynomial(g: np.ndarray, u: np.ndarray, h: np.ndarray, degree: int, block: int, per_neuron: bool, ridge: float) -> dict:
    n, width = g.shape
    means = g.mean(axis=0).astype(np.float32)
    scales = np.std(g, axis=0).astype(np.float32)
    scales = np.maximum(scales, np.float32(1e-3))
    z = (g - means) / scales
    coeff = np.zeros((width, degree + 1), dtype=np.float32)
    if per_neuron:
        for j in range(width):
            phi = np.stack([u[:, j] * (z[:, j] ** p) for p in range(degree + 1)], axis=1).astype(np.float32)
            gram = phi.T @ phi
            coeff[j] = np.linalg.solve(gram + np.eye(degree + 1, dtype=np.float32) * ridge, phi.T @ h[:, j])
    else:
        for start in range(0, width, block):
            end = min(start + block, width)
            zg = z[:, start:end].reshape(-1)
            uu = u[:, start:end].reshape(-1)
            hh = h[:, start:end].reshape(-1)
            phi = np.stack([uu * (zg ** p) for p in range(degree + 1)], axis=1).astype(np.float32)
            gram = phi.T @ phi
            c = np.linalg.solve(gram + np.eye(degree + 1, dtype=np.float32) * ridge, phi.T @ hh)
            coeff[start:end] = c[None, :]
    return {"means": means, "scales": scales, "coeff": coeff, "degree": degree, "block": block, "per_neuron": per_neuron}


def predict_h(g: np.ndarray, u: np.ndarray, model: dict) -> np.ndarray:
    z = (g - model["means"]) / model["scales"]
    coeff = model["coeff"]
    out = np.zeros_like(g, dtype=np.float32)
    for p in range(int(model["degree"]) + 1):
        out += coeff[:, p][None, :] * (u * (z ** p))
    return out


def rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def block_merge(h: np.ndarray, down: np.ndarray, block: int) -> np.ndarray:
    out = np.zeros((len(h), down.shape[0]), dtype=np.float32)
    # down is [hidden, intermediate], so each block is an independent output
    # contribution. The merge is only hidden-dimensional addition.
    for start in range(0, h.shape[1], block):
        end = min(start + block, h.shape[1])
        out += h[:, start:end] @ down[:, start:end].T
    return out


def artifact_bytes(model: dict, dtype_bytes: int = 2) -> dict[str, int]:
    # Stored in fp16 after offline fitting; runtime can promote to fp32/fp16.
    raw = model["means"].nbytes + model["scales"].nbytes + model["coeff"].nbytes
    return {"polynomial_fp16_bytes": int(raw * dtype_bytes / 4), "polynomial_fp32_bytes": int(raw), "degree": int(model["degree"])}


def runtime_cost(hidden: int, intermediate: int, degree: int, block: int) -> dict[str, int]:
    return {
        "gate_projection_mac": hidden * intermediate,
        "up_projection_mac": hidden * intermediate,
        "polynomial_elementwise_mul": intermediate * max(degree, 0),
        "down_projection_mac": hidden * intermediate,
        "merge_additions": hidden * ((intermediate + block - 1) // block - 1),
        "down_block_width": block,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-LUT polynomial SwiGLU split and block merge")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--degrees", default="1,2,3,4,5,7")
    parser.add_argument("--blocks", default="256,512,1024")
    parser.add_argument("--modes", default="per_neuron,per_block")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _, train_g, train_u, train_h, train_y = load_layer(args.calibration_root, args.layer)
    _, test_g, test_u, test_h, test_y = load_layer(args.holdout_root, args.layer)
    down = load_down(args.model, args.layer)
    rows = []
    for degree in [int(v) for v in args.degrees.split(",") if v.strip()]:
        for block in [int(v) for v in args.blocks.split(",") if v.strip()]:
            for mode in [v.strip() for v in args.modes.split(",") if v.strip()]:
                per_neuron = mode == "per_neuron"
                model = fit_polynomial(train_g, train_u, train_h, degree, block, per_neuron, args.ridge)
                pred_h_train = predict_h(train_g, train_u, model)
                pred_h_test = predict_h(test_g, test_u, model)
                pred_y_train = block_merge(pred_h_train, down, block)
                pred_y_test = block_merge(pred_h_test, down, block)
                err_h_train = rel_l2(pred_h_train, train_h)
                err_h_test = rel_l2(pred_h_test, test_h)
                err_y_train = rel_l2(pred_y_train, train_y)
                err_y_test = rel_l2(pred_y_test, test_y)
                rows.append({
                    "mode": mode,
                    "degree": degree,
                    "block": block,
                    "activation_rel_l2_train": float(err_h_train.mean()),
                    "activation_rel_l2_holdout": float(err_h_test.mean()),
                    "output_rel_l2_train": float(err_y_train.mean()),
                    "output_rel_l2_holdout": float(err_y_test.mean()),
                    "output_rel_l2_holdout_p95": float(np.percentile(err_y_test, 95)),
                    "artifact": artifact_bytes(model),
                    "runtime_cost": runtime_cost(train_g.shape[1], train_h.shape[1], degree, block),
                })
    result = {
        "experiment": "non_lut_swiglu_polynomial_split",
        "formula": "h_j = u_j * p_j((g_j-mu_j)/sigma_j), y = sum_b Wdown_b @ h_b",
        "runtime_properties": {
            "lookup": False,
            "runtime_silu": False,
            "offline": "fit per-neuron or per-block polynomial coefficients and store them with the expanded layer artifacts",
            "runtime": "gate/up projections, elementwise powers and products, block down projections, hidden-vector addition",
            "goal": "preserve arithmetic while shifting storage/transfer pressure away from full nonlinear FFN weights",
        },
        "layer": args.layer,
        "train_samples": len(train_g),
        "holdout_samples": len(test_g),
        "down_weight_shape": list(down.shape),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
