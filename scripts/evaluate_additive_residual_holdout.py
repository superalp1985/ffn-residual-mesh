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


def fit_additive(values: np.ndarray, targets: np.ndarray, block_dim: int, k: int, seed: int) -> dict:
    centers, codes = [], []
    for j, start in enumerate(range(0, values.shape[1], block_dim)):
        block = values[:, start:start + block_dim]
        km = MiniBatchKMeans(n_clusters=min(k, len(values)), random_state=seed + j, n_init=2, batch_size=min(256, len(values)), max_iter=60)
        centers.append(km.fit(block).cluster_centers_.astype(np.float32))
        codes.append(km.labels_)
    tables = [np.zeros((len(c), targets.shape[1]), dtype=np.float32) for c in centers]
    bias = targets.mean(axis=0)
    pred = bias + sum(table[code] for table, code in zip(tables, codes))
    for _ in range(7):
        for j, code in enumerate(codes):
            pred -= tables[j][code]
            for c in range(len(tables[j])):
                mask = code == c
                if np.any(mask):
                    tables[j][c] = (targets[mask] - pred[mask]).mean(axis=0)
            pred += tables[j][code]
        bias = (targets - sum(table[code] for table, code in zip(tables, codes))).mean(axis=0)
        pred = bias + sum(table[code] for table, code in zip(tables, codes))
    quantized = np.concatenate([centers[j][codes[j]] for j in range(len(centers))], axis=1)
    base = pred
    u, s, vt = fit_shared_basis(values - quantized, targets - base)
    return {"centers": centers, "tables": tables, "bias": bias, "block_dim": block_dim, "u": u, "s": s, "vt": vt}


def apply(values: np.ndarray, model: dict, rank: int) -> np.ndarray:
    codes, quantized = [], []
    for start, centers in zip(range(0, values.shape[1], model["block_dim"]), model["centers"]):
        block = values[:, start:start + model["block_dim"]]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        code = np.argmin(d2, axis=1)
        codes.append(code)
        quantized.append(centers[code])
    base = model["bias"] + sum(table[code] for table, code in zip(model["tables"], codes))
    if rank <= 0:
        return base
    r = min(rank, len(model["s"]))
    basis = (model["u"][:, :r] * model["s"][:r]) @ model["vt"][:r]
    return base + (values - np.concatenate(quantized, axis=1)) @ basis.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=Path)
    parser.add_argument("test_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default="0,10,18,22,23")
    args = parser.parse_args()
    wanted = {int(x) for x in args.layers.split(",") if x.strip()}
    train, test = collect(args.train_root), collect(args.test_root)
    rows = []
    for layer in sorted(wanted & train.keys() & test.keys()):
        _, th, ty = train[layer]
        _, vh, vy = test[layer]
        model = fit_additive(th, ty, 256, 8, 11000 + layer)
        row = {"layer": layer, "train_samples": len(th), "test_samples": len(vh), "ranks": {}}
        for rank in (0, 16, 32, 64, 128):
            row["ranks"][str(rank)] = metrics(apply(vh, model, rank), vy)
        rows.append(row)
    result = {"train_root": str(args.train_root), "test_root": str(args.test_root), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
