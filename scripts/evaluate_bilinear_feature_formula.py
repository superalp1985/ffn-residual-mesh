from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

from evaluate_additive_holdout import collect


def rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def load_weights(model: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = GGUFReader(str(model))
    result = []
    for name in ("gate", "up", "down"):
        tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_{name}.weight")
        result.append(dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False))
    return tuple(result)  # type: ignore[return-value]


def load_data(root: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {}
        for item in manifest["tensors"]:
            if item["name"] in {f"attn_post_norm-{layer}", f"ffn_gate-{layer}", f"ffn_up-{layer}", f"ffn_swiglu-{layer}", f"ffn_out-{layer}"}:
                raw = np.fromfile(prompt_dir / item["file"], dtype=np.float32)
                tensors[item["name"]] = raw.reshape(tuple(int(x) for x in item["shape"])[::-1])
        names = [f"attn_post_norm-{layer}", f"ffn_gate-{layer}", f"ffn_up-{layer}", f"ffn_swiglu-{layer}", f"ffn_out-{layer}"]
        if all(name in tensors for name in names):
            rows.append(tuple(tensors[name] for name in names))
    if not rows:
        raise ValueError(f"layer {layer} missing from {root}")
    return tuple(np.concatenate([row[i] for row in rows], axis=0) for i in range(5))  # type: ignore[return-value]


def feature_maps(x: np.ndarray, model: dict) -> np.ndarray:
    zg = (x - model["x_center"]) @ model["pg"]
    zu = (x - model["x_center"]) @ model["pu"]
    parts = [np.ones((len(x), 1), dtype=np.float32), zg, zu]
    if model["bilinear"] == "full":
        parts.append((zg[:, :, None] * zu[:, None, :]).reshape(len(x), -1))
    elif model["bilinear"] == "diagonal":
        parts.append(zg * zu)
    elif model["bilinear"] == "paired":
        parts.append(zg[:, ::2] * zu[:, 1::2])
    if model["squares"]:
        parts.extend([zg * zg, zu * zu])
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def fit_model(x: np.ndarray, g: np.ndarray, u: np.ndarray, y: np.ndarray, rank: int, bilinear: str, squares: bool, ridge: float) -> dict:
    x_center = x.mean(axis=0).astype(np.float32)
    # Activation PCA yields input directions P = W.T @ Q, allowing runtime
    # features to be formed from x without a lookup table.
    _, _, vt_g = np.linalg.svd(g - g.mean(axis=0), full_matrices=False)
    _, _, vt_u = np.linalg.svd(u - u.mean(axis=0), full_matrices=False)
    qg = vt_g[:rank].T.astype(np.float32)
    qu = vt_u[:rank].T.astype(np.float32)
    wg, wu, _ = model_weights
    pg = (wg.T @ qg).astype(np.float32)
    pu = (wu.T @ qu).astype(np.float32)
    model = {"x_center": x_center, "pg": pg, "pu": pu, "bilinear": bilinear, "squares": squares}
    phi = feature_maps(x, model)
    gram = phi.T @ phi
    coef = np.linalg.solve(gram + np.eye(phi.shape[1], dtype=np.float32) * ridge, phi.T @ y).astype(np.float32)
    model["coef"] = coef
    model["rank"] = rank
    return model


def artifact_bytes(model: dict, dtype_bytes: int = 2) -> dict[str, int]:
    return {
        "input_basis_fp16_bytes": int((model["pg"].nbytes + model["pu"].nbytes) * dtype_bytes / 4),
        "merge_coef_fp16_bytes": int(model["coef"].nbytes * dtype_bytes / 4),
        "total_fp16_bytes": int((model["pg"].nbytes + model["pu"].nbytes + model["coef"].nbytes) * dtype_bytes / 4),
    }


def runtime_cost(model: dict, hidden: int) -> dict[str, int]:
    rank = int(model["rank"])
    bilinear_width = rank * rank if model["bilinear"] == "full" else rank if model["bilinear"] == "diagonal" else rank // 2 if model["bilinear"] == "paired" else 0
    width = 1 + 2 * rank + bilinear_width + (2 * rank if model["squares"] else 0)
    return {
        "input_projection_mac": 2 * hidden * rank,
        "bilinear_elementwise_mul": bilinear_width,
        "square_elementwise_mul": 2 * rank if model["squares"] else 0,
        "output_merge_mac": hidden * width,
        "feature_width": width,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-LUT bilinear gate/up feature formula")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--ranks", default="8,16,24,32,48")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global model_weights
    model_weights = load_weights(args.model, args.layer)
    train_x, train_g, train_u, _, train_y = load_data(args.calibration_root, args.layer)
    test_x, test_g, test_u, _, test_y = load_data(args.holdout_root, args.layer)
    rows = []
    for rank in [int(v) for v in args.ranks.split(",") if v.strip()]:
        for bilinear, squares, name in (("none", False, "linear"), ("diagonal", False, "diagonal_bilinear"), ("paired", False, "paired_bilinear"), ("diagonal", True, "diagonal_bilinear_plus_squares")):
            model = fit_model(train_x, train_g, train_u, train_y, rank, bilinear, squares, args.ridge)
            train_pred = feature_maps(train_x, model) @ model["coef"]
            test_pred = feature_maps(test_x, model) @ model["coef"]
            train_err, test_err = rel_l2(train_pred, train_y), rel_l2(test_pred, test_y)
            rows.append({
                "family": name,
                "rank": rank,
                "train_rel_l2": float(train_err.mean()),
                "holdout_rel_l2": float(test_err.mean()),
                "holdout_rel_l2_p95": float(np.percentile(test_err, 95)),
                "artifact": artifact_bytes(model),
                "runtime_cost": runtime_cost(model, train_x.shape[1]),
            })
    result = {
        "experiment": "non_lut_bilinear_gate_up_formula",
        "formula": "z_g=P_g.T x; z_u=P_u.T x; phi=[1,z_g,z_u,z_g tensor-product z_u,(z_g^2,z_u^2)]",
        "runtime_properties": {
            "lookup": False,
            "runtime_silu": False,
            "purpose": "represent gate/up nonlinear interaction as pre-expanded bilinear features; spend arithmetic to reduce dynamic weight movement",
        },
        "layer": args.layer,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
