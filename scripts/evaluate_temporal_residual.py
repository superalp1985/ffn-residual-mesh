from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cluster_residual import fit_shared_basis


def load(root: Path, item: dict) -> np.ndarray:
    raw = np.fromfile(root / item["file"], dtype=np.float32)
    return raw.reshape(tuple(int(x) for x in item["shape"])[::-1])


def collect(root: Path, layer: int) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        items = {item["name"]: item for item in manifest["tensors"]}
        x_name, y_name = f"attn_post_norm-{layer}", f"ffn_out-{layer}"
        if x_name in items and y_name in items:
            result.append((load(prompt_dir, items[x_name]), load(prompt_dir, items[y_name])))
    return result


def rel_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)


def predict_sequence(x: np.ndarray, y: np.ndarray, basis: np.ndarray, reset_interval: int) -> tuple[np.ndarray, np.ndarray]:
    pred = np.empty_like(y)
    exact = np.zeros(len(y), dtype=bool)
    pred[0] = y[0]
    exact[0] = True
    for t in range(1, len(y)):
        if reset_interval > 0 and t % reset_interval == 0:
            pred[t] = y[t]
            exact[t] = True
        else:
            pred[t] = pred[t - 1] + basis @ (x[t] - x[t - 1])
    return pred, exact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=Path)
    parser.add_argument("test_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default="0,10,18,22,23")
    args = parser.parse_args()
    rows = []
    ranks = (16, 32, 64, 128)
    intervals = (2, 4, 8, 16, 0)
    for layer in [int(x) for x in args.layers.split(",") if x.strip()]:
        train = collect(args.train_root, layer)
        test = collect(args.test_root, layer)
        if not train or not test:
            continue
        dx = np.concatenate([np.diff(x, axis=0) for x, _ in train])
        dy = np.concatenate([np.diff(y, axis=0) for _, y in train])
        u, s, vt = fit_shared_basis(dx, dy)
        layer_row = {"layer": layer, "train_delta_samples": len(dx), "ranks": {}}
        for rank in ranks:
            r = min(rank, len(s))
            basis = (u[:, :r] * s[:r]) @ vt[:r]
            rank_row = {}
            for interval in intervals:
                errors, approx_errors = [], []
                for x, y in test:
                    pred, exact = predict_sequence(x, y, basis, interval)
                    e = rel_error(pred, y)
                    errors.extend(e.tolist())
                    approx_errors.extend(e[~exact].tolist())
                rank_row[str(interval)] = {
                    "approx_fraction": float(len(approx_errors) / max(len(errors), 1)),
                    "approx_rel_l2": float(np.mean(approx_errors)) if approx_errors else 0.0,
                    "expected_full_output_rel_l2": float(np.mean(errors)),
                }
            layer_row["ranks"][str(rank)] = rank_row
        rows.append(layer_row)
    result = {"train_root": str(args.train_root), "test_root": str(args.test_root), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
