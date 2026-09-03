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
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: load_tensor(prompt_dir, item) for item in manifest["tensors"]}
        names = [f"attn_post_norm-{layer}", f"ffn_gate-{layer}", f"ffn_up-{layer}", f"ffn_swiglu-{layer}", f"ffn_out-{layer}"]
        if all(name in tensors for name in names):
            rows.append(tuple(tensors[name] for name in names))
    if not rows:
        raise ValueError(f"layer {layer} missing from {root}")
    return tuple(np.concatenate([row[i] for row in rows], axis=0) for i in range(5))  # type: ignore[return-value]


def load_down_weight(model: Path, layer: int) -> np.ndarray:
    reader = GGUFReader(str(model))
    name = f"blk.{layer}.ffn_down.weight"
    tensor = next(item for item in reader.tensors if item.name == name)
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)


def silu_derivatives(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = 1.0 / (1.0 + np.exp(-x))
    s0 = x * s
    s1 = s + x * s * (1.0 - s)
    s2 = s * (1.0 - s) * (2.0 + x * (1.0 - 2.0 * s))
    return s0.astype(np.float32), s1.astype(np.float32), s2.astype(np.float32)


def algebraic_terms(g: np.ndarray, u: np.ndarray, g0: np.ndarray, u0: np.ndarray, order: int) -> np.ndarray:
    s0, s1, s2 = silu_derivatives(g0)
    dg, du = g - g0, u - u0
    out = np.broadcast_to(s0 * u0, np.broadcast(g, u).shape).copy()
    if order >= 1:
        out = out + s0 * du + s1 * u0 * dg
    if order >= 2:
        out = out + s1 * dg * du + 0.5 * s2 * u0 * dg * dg
    if order >= 3:
        # Third-order term uses a finite-difference derivative of SiLU'' at
        # calibration time; runtime still only evaluates multiplies/adds.
        eps = np.float32(1e-2)
        _, _, s2p = silu_derivatives(g0 + eps)
        _, _, s2m = silu_derivatives(g0 - eps)
        s3 = (s2p - s2m) / (2.0 * eps)
        out = out + 0.5 * s2 * dg * dg * du + (s3 / 6.0) * u0 * dg * dg * dg
    return out.astype(np.float32, copy=False)


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    err = np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)
    return float(err.mean())


def rank_reconstruction(train_residual: np.ndarray, test_residual: np.ndarray, rank: int) -> tuple[float, float]:
    mean = train_residual.mean(axis=0)
    centered = train_residual - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    r = min(rank, vt.shape[0])
    basis = vt[:r].T.astype(np.float32)
    train_hat = mean + (train_residual - mean) @ basis @ basis.T
    test_hat = mean + (test_residual - mean) @ basis @ basis.T
    return rel_l2(train_hat, train_residual), rel_l2(test_hat, test_residual)


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-LUT algebraic SwiGLU split and merge experiment")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--orders", default="0,1,2,3")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _, train_g, train_u, train_h, train_y = load_layer(args.calibration_root, args.layer)
    _, test_g, test_u, test_h, test_y = load_layer(args.holdout_root, args.layer)
    down = load_down_weight(args.model, args.layer)
    # GGUF down is [hidden, intermediate], while captured vectors are row-wise.
    replay_train_y = train_h @ down.T
    replay_test_y = test_h @ down.T
    g0, u0 = train_g.mean(axis=0), train_u.mean(axis=0)
    base_h = algebraic_terms(train_g[:1], train_u[:1], g0, u0, 0)
    if base_h.ndim == 2:
        base_h = base_h[0]
    base_y = base_h @ down.T
    rows = []
    for order in [int(v) for v in args.orders.split(",") if v.strip()]:
        train_formula_h = algebraic_terms(train_g, train_u, g0, u0, order)
        test_formula_h = algebraic_terms(test_g, test_u, g0, u0, order)
        train_formula_y = train_formula_h @ down.T
        test_formula_y = test_formula_h @ down.T
        rows.append({
            "order": order,
            "activation_rel_l2_train": rel_l2(train_formula_h, train_h),
            "activation_rel_l2_holdout": rel_l2(test_formula_h, test_h),
            "output_rel_l2_vs_capture_train": rel_l2(train_formula_y, train_y),
            "output_rel_l2_vs_capture_holdout": rel_l2(test_formula_y, test_y),
            "output_rel_l2_vs_weight_replay_train": rel_l2(train_formula_y, replay_train_y),
            "output_rel_l2_vs_weight_replay_holdout": rel_l2(test_formula_y, replay_test_y),
            "dynamic_output_energy_ratio_train": float(np.linalg.norm(train_formula_y - base_y, axis=1).mean() / max(np.linalg.norm(train_formula_y, axis=1).mean(), 1e-6)),
            "dynamic_output_energy_ratio_holdout": float(np.linalg.norm(test_formula_y - base_y, axis=1).mean() / max(np.linalg.norm(test_formula_y, axis=1).mean(), 1e-6)),
        })

    exact_train_residual = replay_train_y - base_y
    exact_test_residual = replay_test_y - base_y
    rank_rows = []
    for rank in [int(v) for v in args.ranks.split(",") if v.strip()]:
        train_err, test_err = rank_reconstruction(exact_train_residual, exact_test_residual, rank)
        rank_rows.append({"rank": rank, "train_residual_rel_l2": train_err, "holdout_residual_rel_l2": test_err})
    result = {
        "experiment": "non_lut_swiglu_algebraic_split",
        "formula": "silu(g)u ~= s0*u0 + s0*du + s1*u0*dg + s1*dg*du + 0.5*s2*u0*dg^2 + ...",
        "runtime_properties": {
            "lookup": False,
            "runtime_silu": False,
            "offline": "g0/u0, SiLU derivatives, base output, and residual layout",
            "runtime": "linear projections, elementwise multiply/add, and one output merge",
            "purpose": "trade extra arithmetic for lower dynamic weight movement; no claim of lower total FLOPs",
        },
        "layer": args.layer,
        "train_samples": len(train_g),
        "holdout_samples": len(test_g),
        "down_weight_shape": list(down.shape),
        "weight_replay_rel_l2_train": rel_l2(replay_train_y, train_y),
        "weight_replay_rel_l2_holdout": rel_l2(replay_test_y, test_y),
        "base_output_norm": float(np.linalg.norm(base_y)),
        "orders": rows,
        "residual_rank": rank_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
