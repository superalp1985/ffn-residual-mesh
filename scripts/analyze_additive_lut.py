from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def load_tensor(root: Path, item: dict) -> np.ndarray:
    raw = np.fromfile(root / item["file"], dtype=np.float32)
    shape = tuple(int(x) for x in item["shape"])
    return raw.reshape(shape[::-1])


def collect(root: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    all_layers: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: load_tensor(prompt_dir, item) for item in manifest["tensors"]}
        for name, x in tensors.items():
            if not name.startswith("attn_post_norm-"):
                continue
            layer = int(name.rsplit("-", 1)[1])
            h_name, y_name = f"ffn_swiglu-{layer}", f"ffn_out-{layer}"
            if h_name in tensors and y_name in tensors:
                all_layers.setdefault(layer, []).append((x, tensors[h_name], tensors[y_name]))
    return {
        layer: (np.concatenate([x for x, _, _ in chunks]),
                np.concatenate([h for _, h, _ in chunks]),
                np.concatenate([y for _, _, y in chunks]))
        for layer, chunks in all_layers.items()
    }


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    errs = np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)
    return float(errs.mean())


def cosine(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    return float((np.sum(pred * target, axis=1) / np.maximum(denom, 1e-6)).mean())


def fit_additive(values: np.ndarray, targets: np.ndarray, block_dim: int, k: int, seed: int) -> tuple[np.ndarray, dict]:
    n, dim = values.shape
    blocks = [(start, min(start + block_dim, dim)) for start in range(0, dim, block_dim)]
    codes = []
    centers = []
    for block_id, (start, end) in enumerate(blocks):
        km = MiniBatchKMeans(n_clusters=min(k, n), random_state=seed + block_id, n_init=2,
                             batch_size=min(256, n), max_iter=60)
        code = km.fit_predict(values[:, start:end])
        codes.append(code)
        centers.append(km.cluster_centers_.astype(np.float32))

    n_out = targets.shape[1]
    tables = [np.zeros((len(c), n_out), dtype=np.float32) for c in centers]
    bias = targets.mean(axis=0)
    pred = np.broadcast_to(bias, targets.shape).copy()
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

    meta = {
        "blocks": len(blocks),
        "block_dim": block_dim,
        "k": k,
        "table_bytes_fp16": int(sum(table.size for table in tables) * 2),
        "center_bytes_fp16": int(sum(center.size for center in centers) * 2),
        "codes": codes,
        "centers": centers,
    }
    return pred, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--input-block", type=int, default=128)
    parser.add_argument("--swiglu-block", type=int, default=256)
    parser.add_argument("--layers", default="0,10,18,22,23")
    parser.add_argument("--max-samples", type=int, default=384)
    args = parser.parse_args()

    wanted = {int(x) for x in args.layers.split(",") if x.strip()}
    rows = []
    for layer, (x, h, y) in sorted(collect(args.calibration_root).items()):
        if layer not in wanted:
            continue
        if len(x) > args.max_samples:
            rng = np.random.default_rng(0xA100 + layer)
            keep = rng.choice(len(x), size=args.max_samples, replace=False)
            x, h, y = x[keep], h[keep], y[keep]
        x_pred, x_meta = fit_additive(x, y, args.input_block, args.k, 1000 + layer)
        h_pred, h_meta = fit_additive(h, y, args.swiglu_block, args.k, 2000 + layer)
        rows.append({
            "layer": layer,
            "samples": len(x),
            "input_additive": {
                "rel_l2": rel_l2(x_pred, y),
                "cosine": cosine(x_pred, y),
                "table_bytes_fp16": x_meta["table_bytes_fp16"],
                "center_bytes_fp16": x_meta["center_bytes_fp16"],
                "blocks": x_meta["blocks"],
                "block_dim": args.input_block,
            },
            "swiglu_additive": {
                "rel_l2": rel_l2(h_pred, y),
                "cosine": cosine(h_pred, y),
                "table_bytes_fp16": h_meta["table_bytes_fp16"],
                "center_bytes_fp16": h_meta["center_bytes_fp16"],
                "blocks": h_meta["blocks"],
                "block_dim": args.swiglu_block,
            },
        })

    result = {
        "calibration_root": str(args.calibration_root),
        "k": args.k,
        "rows": rows,
        "notes": [
            "Each block has a small vector codebook; output tables are fitted by coordinate descent.",
            "This is a feasibility baseline for CPU-side additive lookup, not a replacement kernel.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
