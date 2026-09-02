from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from analyze_cluster_residual import fit_shared_basis
from evaluate_additive_holdout import collect


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    e = np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)
    c = np.sum(pred * target, axis=1) / np.maximum(np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1), 1e-6)
    return {"rel_l2": float(e.mean()), "rel_l2_p95": float(np.percentile(e, 95)), "cosine": float(c.mean())}


def fit_model(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> dict:
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=min(256, len(x)), max_iter=100)
    labels = km.fit_predict(x)
    centers = km.cluster_centers_.astype(np.float32)
    base = np.zeros_like(y)
    counts = np.bincount(labels, minlength=k)
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            base[mask] = y[mask].mean(axis=0)
    dx, dy = x - centers[labels], y - base
    u, s, vt = fit_shared_basis(dx, dy)
    return {"centers": centers, "base_outputs": np.array([y[labels == c].mean(axis=0) if counts[c] else np.zeros(y.shape[1], dtype=np.float32) for c in range(k)]), "u": u, "s": s, "vt": vt}


def apply_model(x: np.ndarray, model: dict, rank: int) -> tuple[np.ndarray, np.ndarray]:
    centers = model["centers"]
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    distance = np.sqrt(d2[np.arange(len(x)), labels])
    r = min(rank, len(model["s"]))
    basis = (model["u"][:, :r] * model["s"][:r]) @ model["vt"][:r]
    pred = model["base_outputs"][labels] + (x - centers[labels]) @ basis.T
    return pred, distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=Path)
    parser.add_argument("test_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default="0,10,18,22,23")
    parser.add_argument("--ks", default="16,64")
    args = parser.parse_args()
    wanted = {int(x) for x in args.layers.split(",") if x.strip()}
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    train, test = collect(args.train_root), collect(args.test_root)
    rows = []
    for layer in sorted(wanted & train.keys() & test.keys()):
        tx, _, ty = train[layer]
        vx, _, vy = test[layer]
        layer_row = {"layer": layer, "train_samples": len(tx), "test_samples": len(vx), "ks": {}}
        for k in ks:
            model = fit_model(tx, ty, k, 9000 + layer + k)
            rank_rows = {}
            for rank in (16, 32, 64, 128):
                pred, distance = apply_model(vx, model, rank)
                rank_rows[str(rank)] = metrics(pred, vy)
                rank_rows[str(rank)]["distance_error_corr"] = float(np.corrcoef(distance, np.linalg.norm(pred - vy, axis=1))[0, 1])
            layer_row["ks"][str(k)] = rank_rows
        rows.append(layer_row)
    result = {"train_root": str(args.train_root), "test_root": str(args.test_root), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
