from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def load_tensor(root: Path, item: dict) -> np.ndarray:
    raw = np.fromfile(root / item["file"], dtype=np.float32)
    return raw.reshape(tuple(int(x) for x in item["shape"])[::-1])


def collect(root: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    layers: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: load_tensor(prompt_dir, item) for item in manifest["tensors"]}
        for name, x in tensors.items():
            if not name.startswith("attn_post_norm-"):
                continue
            layer = int(name.rsplit("-", 1)[1])
            if f"ffn_swiglu-{layer}" in tensors and f"ffn_out-{layer}" in tensors:
                layers.setdefault(layer, []).append((x, tensors[f"ffn_swiglu-{layer}"], tensors[f"ffn_out-{layer}"]))
    return {layer: (np.concatenate([x for x, _, _ in v]), np.concatenate([h for _, h, _ in v]), np.concatenate([y for _, _, y in v])) for layer, v in layers.items()}


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    e = np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)
    c = np.sum(pred * target, axis=1) / np.maximum(np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1), 1e-6)
    return {"rel_l2": float(e.mean()), "rel_l2_p95": float(np.percentile(e, 95)), "cosine": float(c.mean())}


def fit(values: np.ndarray, targets: np.ndarray, block_dim: int, k: int, seed: int) -> dict:
    n, dim = values.shape
    centers, codes = [], []
    for j, start in enumerate(range(0, dim, block_dim)):
        end = min(start + block_dim, dim)
        km = MiniBatchKMeans(n_clusters=min(k, n), random_state=seed + j, n_init=2, batch_size=min(256, n), max_iter=60)
        codes.append(km.fit_predict(values[:, start:end]))
        centers.append(km.cluster_centers_.astype(np.float32))
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
    return {"centers": centers, "tables": tables, "bias": bias, "block_dim": block_dim}


def apply(values: np.ndarray, model: dict) -> np.ndarray:
    codes = []
    for start, centers in zip(range(0, values.shape[1], model["block_dim"]), model["centers"]):
        block = values[:, start:start + model["block_dim"]]
        distances = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        codes.append(np.argmin(distances, axis=1))
    return model["bias"] + sum(table[code] for table, code in zip(model["tables"], codes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=Path)
    parser.add_argument("test_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--input-block", type=int, default=128)
    parser.add_argument("--swiglu-block", type=int, default=256)
    parser.add_argument("--layers", default="0,10,18,22,23")
    args = parser.parse_args()
    wanted = {int(x) for x in args.layers.split(",") if x.strip()}
    train, test = collect(args.train_root), collect(args.test_root)
    rows = []
    for layer in sorted(wanted & train.keys() & test.keys()):
        tx, th, ty = train[layer]
        vx, vh, vy = test[layer]
        xm, hm = fit(tx, ty, args.input_block, args.k, 3000 + layer), fit(th, ty, args.swiglu_block, args.k, 4000 + layer)
        rows.append({"layer": layer, "train_samples": len(tx), "test_samples": len(vx), "input_additive": metrics(apply(vx, xm), vy), "swiglu_additive": metrics(apply(vh, hm), vy)})
    result = {"train_root": str(args.train_root), "test_root": str(args.test_root), "k": args.k, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
