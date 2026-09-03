from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

from evaluate_additive_holdout import load_tensor


def load_layer(root: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = [f"attn_post_norm-{layer}", f"ffn_gate-{layer}", f"ffn_up-{layer}", f"ffn_swiglu-{layer}", f"ffn_out-{layer}"]
    rows = []
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: load_tensor(prompt_dir, item) for item in manifest["tensors"] if item["name"] in names}
        if all(name in tensors for name in names):
            rows.append(tuple(tensors[name] for name in names))
    if not rows:
        raise ValueError(f"layer {layer} missing from {root}")
    return tuple(np.concatenate([row[i] for row in rows], axis=0) for i in range(5))  # type: ignore[return-value]


def load_down(model: Path, layer: int) -> tuple[np.ndarray, int]:
    reader = GGUFReader(str(model))
    tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_down.weight")
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False), int(tensor.n_bytes)


def fit_poly(g: np.ndarray, u: np.ndarray, h: np.ndarray, degree: int, ridge: float) -> dict:
    mu = g.mean(axis=0).astype(np.float32)
    sigma = np.maximum(g.std(axis=0).astype(np.float32), np.float32(1e-3))
    z = (g - mu) / sigma
    # h_j ~= u_j * sum_p c[j,p] z_j^p
    phi = np.stack([u * (z ** p) for p in range(degree + 1)], axis=2).astype(np.float32)
    coeff = np.zeros((g.shape[1], degree + 1), dtype=np.float32)
    eye = np.eye(degree + 1, dtype=np.float32) * ridge
    for j in range(g.shape[1]):
        a = phi[:, j, :]
        coeff[j] = np.linalg.solve(a.T @ a + eye, a.T @ h[:, j])
    return {"mu": mu, "sigma": sigma, "coeff": coeff, "degree": degree}


def chebyshev_terms(z: np.ndarray, degree: int, bound: float) -> np.ndarray:
    t = np.clip(z / np.float32(bound), -1.0, 1.0)
    terms = [np.ones_like(t, dtype=np.float32)]
    if degree >= 1:
        terms.append(t)
    for _ in range(2, degree + 1):
        terms.append((2.0 * t * terms[-1] - terms[-2]).astype(np.float32, copy=False))
    return np.stack(terms, axis=2)


def fit_chebyshev(g: np.ndarray, u: np.ndarray, h: np.ndarray, degree: int, bound: float, ridge: float) -> dict:
    mu = g.mean(axis=0).astype(np.float32)
    sigma = np.maximum(g.std(axis=0).astype(np.float32), np.float32(1e-3))
    z = (g - mu) / sigma
    terms = chebyshev_terms(z, degree, bound)
    coeff = np.zeros((g.shape[1], degree + 1), dtype=np.float32)
    eye = np.eye(degree + 1, dtype=np.float32) * ridge
    for j in range(g.shape[1]):
        phi = u[:, j, None] * terms[:, j, :]
        coeff[j] = np.linalg.solve(phi.T @ phi + eye, phi.T @ h[:, j])
    return {"mu": mu, "sigma": sigma, "coeff": coeff, "degree": degree, "bound": bound}


def poly_base(g: np.ndarray, u: np.ndarray, model: dict) -> np.ndarray:
    z = (g - model["mu"]) / model["sigma"]
    out = np.zeros_like(g, dtype=np.float32)
    for p in range(int(model["degree"]) + 1):
        out += model["coeff"][:, p][None, :] * u * (z ** p)
    return out


def chebyshev_base(g: np.ndarray, u: np.ndarray, model: dict) -> np.ndarray:
    z = (g - model["mu"]) / model["sigma"]
    terms = chebyshev_terms(z, int(model["degree"]), float(model["bound"]))
    return np.sum(u[:, :, None] * terms * model["coeff"][None, :, :], axis=2).astype(np.float32, copy=False)


def residual_features(z: np.ndarray, feature_degree: int) -> np.ndarray:
    features = [z]
    if feature_degree >= 2:
        features.append(z * z)
    if feature_degree >= 3:
        features.append(z * z * z)
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def fit_residual_map(x: np.ndarray, residual: np.ndarray, input_rank: int, output_rank: int, feature_degree: int, ridge: float) -> dict:
    x_mu = x.mean(axis=0).astype(np.float32)
    dx = x - x_mu
    _, _, vt = np.linalg.svd(dx, full_matrices=False)
    p = vt[: min(input_rank, vt.shape[0])].T.astype(np.float32)
    z = dx @ p
    features = residual_features(z, feature_degree)
    _, _, vt_r = np.linalg.svd(residual - residual.mean(axis=0), full_matrices=False)
    u = vt_r[: min(output_rank, vt_r.shape[0])].T.astype(np.float32)
    target = (residual - residual.mean(axis=0)) @ u
    gram = features.T @ features
    mapping = np.linalg.solve(gram + np.eye(features.shape[1], dtype=np.float32) * ridge, features.T @ target).astype(np.float32)
    return {
        "x_mu": x_mu,
        "input_basis": p,
        "feature_degree": feature_degree,
        "output_mean": residual.mean(axis=0).astype(np.float32),
        "output_basis": u,
        "mapping": mapping,
    }


def residual_predict(x: np.ndarray, model: dict) -> np.ndarray:
    z = (x - model["x_mu"]) @ model["input_basis"]
    features = residual_features(z, int(model["feature_degree"]))
    return model["output_mean"] + (features @ model["mapping"]) @ model["output_basis"].T


def rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-LUT polynomial FFN base plus GPU residual coefficient experiment")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--degrees", default="2,3,4")
    parser.add_argument("--input-ranks", default="32,64,128")
    parser.add_argument("--output-ranks", default="8,16,32,64")
    parser.add_argument("--feature-degrees", default="1,2")
    parser.add_argument("--residual-target", choices=("exact", "capture"), default="exact")
    parser.add_argument("--base-basis", choices=("monomial", "chebyshev"), default="monomial")
    parser.add_argument("--chebyshev-bound", type=float, default=6.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, train_g, train_u, train_h, train_capture_y = load_layer(args.calibration_root, args.layer)
    test_x, test_g, test_u, test_h, test_capture_y = load_layer(args.holdout_root, args.layer)
    down, down_bytes = load_down(args.model, args.layer)
    train_exact_y, test_exact_y = train_h @ down.T, test_h @ down.T
    rows = []
    for degree in [int(v) for v in args.degrees.split(",") if v.strip()]:
        if args.base_basis == "chebyshev":
            poly = fit_chebyshev(train_g, train_u, train_h, degree, args.chebyshev_bound, args.ridge)
            base_h_train, base_h_test = chebyshev_base(train_g, train_u, poly), chebyshev_base(test_g, test_u, poly)
        else:
            poly = fit_poly(train_g, train_u, train_h, degree, args.ridge)
            base_h_train, base_h_test = poly_base(train_g, train_u, poly), poly_base(test_g, test_u, poly)
        base_y_train, base_y_test = base_h_train @ down.T, base_h_test @ down.T
        residual_target_train = train_exact_y if args.residual_target == "exact" else train_capture_y
        residual_target_test = test_exact_y if args.residual_target == "exact" else test_capture_y
        residual_train, residual_test = residual_target_train - base_y_train, residual_target_test - base_y_test
        base_rel = {
            "train": float(rel_l2(base_y_train, train_capture_y).mean()),
            "holdout": float(rel_l2(base_y_test, test_capture_y).mean()),
        }
        residual_ratio = {
            "train": float(np.linalg.norm(residual_train, axis=1).mean() / max(np.linalg.norm(train_exact_y, axis=1).mean(), 1e-6)),
            "holdout": float(np.linalg.norm(residual_test, axis=1).mean() / max(np.linalg.norm(test_exact_y, axis=1).mean(), 1e-6)),
        }
        for input_rank in [int(v) for v in args.input_ranks.split(",") if v.strip()]:
            for feature_degree in [int(v) for v in args.feature_degrees.split(",") if v.strip()]:
                for output_rank in [int(v) for v in args.output_ranks.split(",") if v.strip()]:
                    residual_model = fit_residual_map(train_x, residual_train, input_rank, output_rank, feature_degree, args.ridge)
                    pred_train = base_y_train + residual_predict(train_x, residual_model)
                    pred_test = base_y_test + residual_predict(test_x, residual_model)
                    # CPU creates the base; GPU receives output_rank coefficients and
                    # applies a resident output basis. The basis itself is uploaded once.
                    basis_bytes = int(output_rank * down.shape[0] * 2)
                    coeff_bytes = int(output_rank * 2)
                    rows.append({
                        "degree": degree,
                        "input_rank": input_rank,
                        "feature_degree": feature_degree,
                        "output_rank": output_rank,
                        "base_rel_l2_vs_capture": base_rel,
                        "residual_energy_ratio": residual_ratio,
                        "hybrid_rel_l2_vs_capture_train": float(rel_l2(pred_train, train_capture_y).mean()),
                        "hybrid_rel_l2_vs_capture_holdout": float(rel_l2(pred_test, test_capture_y).mean()),
                        "hybrid_rel_l2_vs_exact_train": float(rel_l2(pred_train, train_exact_y).mean()),
                        "hybrid_rel_l2_vs_exact_holdout": float(rel_l2(pred_test, test_exact_y).mean()),
                        "transfer": {
                            "gpu_basis_resident_fp16_bytes": basis_bytes,
                            "per_token_h2d_residual_coeff_fp16_bytes": coeff_bytes,
                            "per_token_cpu_base_output_fp16_bytes": int(down.shape[0] * 2),
                            "full_down_weight_bytes_q4": down_bytes,
                        },
                        "extra_arithmetic": {
                            "cpu_gate_projection_mac": int(train_x.shape[1] * train_h.shape[1]),
                            "cpu_up_projection_mac": int(train_x.shape[1] * train_h.shape[1]),
                            "cpu_base_down_mac": int(train_h.shape[1] * down.shape[0]),
                            "cpu_base_total_projection_mac": int(3 * train_x.shape[1] * train_h.shape[1]),
                            "cpu_base_elementwise_mul": int(train_h.shape[1] * max(degree, 0)),
                            "cpu_residual_projection_mac": int(train_x.shape[1] * input_rank),
                            "cpu_residual_feature_mul": int(input_rank * max(feature_degree - 1, 0)),
                            "gpu_residual_merge_mac": int(down.shape[0] * output_rank),
                        },
                    })
    result = {
        "experiment": "polynomial_base_plus_residual_coefficients",
        "formula": (
            "y = W_down @ [u * p((g-mu)/sigma)] + U_r @ alpha(phi(P.T @ (x-mu)))"
            if args.base_basis == "monomial"
            else "y = W_down @ [u * T((g-mu)/(sigma*bound))] + U_r @ alpha(phi(P.T @ (x-mu)))"
        ),
        "runtime_properties": {
            "lookup": False,
            "runtime_silu": False,
            "base": "CPU/RAM-side polynomial SwiGLU output using pre-expanded coefficients",
            "residual": "GPU-resident output basis times small per-token coefficients",
            "purpose": "explicitly trade CPU/GPU arithmetic for lower per-token H2D traffic",
        },
        "layer": args.layer,
        "base_basis": args.base_basis,
        "chebyshev_bound": args.chebyshev_bound if args.base_basis == "chebyshev" else None,
        "residual_target": args.residual_target,
        "train_samples": len(train_g),
        "holdout_samples": len(test_g),
        "down_weight_shape": list(down.shape),
        "exact_weight_replay_rel_l2_train": float(rel_l2(train_exact_y, train_capture_y).mean()),
        "exact_weight_replay_rel_l2_holdout": float(rel_l2(test_exact_y, test_capture_y).mean()),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
