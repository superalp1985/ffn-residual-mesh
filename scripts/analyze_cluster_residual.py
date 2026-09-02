from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


RANKS = (4, 8, 16, 32, 64, 128)
HIDDEN = 2048


def load_tensor(root: Path, item: dict) -> np.ndarray:
    raw = np.fromfile(root / item["file"], dtype=np.float32)
    shape = tuple(int(x) for x in item["shape"])
    return raw.reshape(shape[::-1])


def collect(root: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    by_layer: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: load_tensor(prompt_dir, item) for item in manifest["tensors"]}
        for name, values in tensors.items():
            if not name.startswith("attn_post_norm-"):
                continue
            layer = int(name.rsplit("-", 1)[1])
            target_name = f"ffn_out-{layer}"
            if target_name not in tensors:
                continue
            x, y = values, tensors[target_name]
            if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
                raise ValueError(f"token alignment mismatch in {prompt_dir}: {x.shape} {y.shape}")
            by_layer.setdefault(layer, []).append((x, y))
    return {
        layer: (np.concatenate([x for x, _ in chunks]), np.concatenate([y for _, y in chunks]))
        for layer, chunks in by_layer.items()
    }


def rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def cosine(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    return np.sum(pred * target, axis=1) / np.maximum(denom, 1e-6)


def fit_shared_basis(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Dual ridge regression avoids a 2048x2048 solve when calibration samples are few.
    gram = dx @ dx.T
    ridge = max(float(np.trace(gram)) / max(len(dx), 1) * 1e-4, 1e-6)
    coeff = np.linalg.solve(gram + ridge * np.eye(len(dx), dtype=np.float32), dy)
    # B = coeff.T @ dx. Factor the 2048-dimensional map through the sample
    # dimension, then SVD only the small n_samples x n_samples core.
    up, rp = np.linalg.qr(coeff.T, mode="reduced")
    qx, rx = np.linalg.qr(dx.T, mode="reduced")
    uc, s, vtc = np.linalg.svd(rp @ rx.T, full_matrices=False)
    return up @ uc, s, vtc @ qx.T


def predict_residual(basis: np.ndarray, dx: np.ndarray) -> np.ndarray:
    return dx @ basis.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=384)
    args = parser.parse_args()

    data = collect(args.calibration_root)
    rows = []
    for layer, (x, y) in sorted(data.items()):
        if len(x) > args.max_samples:
            rng = np.random.default_rng(0xFF00 + layer)
            keep = rng.choice(len(x), size=args.max_samples, replace=False)
            x, y = x[keep], y[keep]
        max_k = min(args.max_k, len(x))
        k_values = [k for k in (1, 4, 16, 64) if k <= max_k]
        layer_row = {"layer": layer, "samples": int(len(x)), "hidden": int(x.shape[1]), "k": {}}
        for k in k_values:
            km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=min(256, len(x)), max_iter=100)
            labels = km.fit_predict(x)
            centers = km.cluster_centers_.astype(np.float32)
            base_outputs = np.zeros_like(y)
            counts = np.bincount(labels, minlength=k)
            for cluster in range(k):
                mask = labels == cluster
                if counts[cluster] > 0:
                    base_outputs[mask] = y[mask].mean(axis=0)
            dx = x - centers[labels]
            dy = y - base_outputs
            result = {
                "lookup_rel_l2": float(rel_l2(base_outputs, y).mean()),
                "lookup_rel_l2_p95": float(np.percentile(rel_l2(base_outputs, y), 95)),
                "lookup_cosine": float(cosine(base_outputs, y).mean()),
                "cluster_min": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
                "cluster_max": int(counts.max()) if len(counts) else 0,
                "ranks": {},
            }
            u, s, vt = fit_shared_basis(dx, dy)
            for rank in RANKS:
                r = min(rank, len(s))
                basis = (u[:, :r] * s[:r]) @ vt[:r]
                pred = base_outputs + predict_residual(basis, dx)
                errs = rel_l2(pred, y)
                result["ranks"][str(rank)] = {
                    "rel_l2": float(errs.mean()),
                    "rel_l2_p95": float(np.percentile(errs, 95)),
                    "cosine": float(cosine(pred, y).mean()),
                    "basis_bytes_fp16": int(2 * HIDDEN * rank * 2),
                }
            # FP16 table proxy: input centers for routing + output centers for lookup.
            result["table_bytes_fp16"] = int(k * HIDDEN * 2 * 2)
            layer_row["k"][str(k)] = result
        rows.append(layer_row)

    aggregate = {}
    for k in (1, 4, 16, 64):
        subset = [row["k"][str(k)] for row in rows if str(k) in row["k"]]
        if not subset:
            continue
        aggregate[str(k)] = {
            "lookup_rel_l2": float(np.mean([r["lookup_rel_l2"] for r in subset])),
            "lookup_cosine": float(np.mean([r["lookup_cosine"] for r in subset])),
            "ranks": {
                str(rank): {
                    "rel_l2": float(np.mean([r["ranks"][str(rank)]["rel_l2"] for r in subset])),
                    "cosine": float(np.mean([r["ranks"][str(rank)]["cosine"] for r in subset])),
                }
                for rank in RANKS
            },
        }

    result = {
        "calibration_root": str(args.calibration_root),
        "layers": rows,
        "aggregate": aggregate,
        "notes": [
            "Lookup output is the empirical mean ffn_out per input cluster.",
            "Residual uses one shared low-rank linear map across clusters; this is a deliberately conservative table-size baseline.",
            "All errors are relative per-token L2 against the captured dense ffn_out.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"layers": len(rows), "aggregate": aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
