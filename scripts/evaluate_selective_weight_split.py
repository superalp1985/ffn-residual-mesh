from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

from evaluate_additive_holdout import load_tensor


def load_data(root: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wanted = [f"attn_post_norm-{layer}", f"ffn_gate-{layer}", f"ffn_up-{layer}", f"ffn_out-{layer}"]
    rows = []
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: load_tensor(prompt_dir, item) for item in manifest["tensors"] if item["name"] in wanted}
        if all(name in tensors for name in wanted):
            rows.append(tuple(tensors[name] for name in wanted))
    if not rows:
        raise ValueError(f"layer {layer} missing from {root}")
    return tuple(np.concatenate([row[i] for row in rows], axis=0) for i in range(4))  # type: ignore[return-value]


def load_weights(model: Path, layer: int) -> tuple[tuple[np.ndarray, int], ...]:
    reader = GGUFReader(str(model))
    result = []
    for name in ("gate", "up", "down"):
        tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_{name}.weight")
        result.append((dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False), int(tensor.n_bytes)))
    return tuple(result)  # type: ignore[return-value]


def prepare_blocks(weight: np.ndarray, quant_bytes: int, block: int, rank: int) -> list[dict]:
    result = []
    n_rows, n_cols = weight.shape
    for start in range(0, n_rows, block):
        end = min(start + block, n_rows)
        part = weight[start:end]
        u, s, vt = np.linalg.svd(part, full_matrices=False)
        r = min(rank, len(s))
        left = (u[:, :r] * s[:r]).astype(np.float32, copy=False)
        right = vt[:r].astype(np.float32, copy=False)
        approx = left @ right
        residual = float(np.linalg.norm(part - approx) / max(np.linalg.norm(part), 1e-6))
        dense_bytes = int(round(quant_bytes * (end - start) / n_rows))
        factor_bytes = int((left.size + right.size) * 2)
        result.append({
            "start": start,
            "end": end,
            "left": left,
            "right": right,
            "residual": residual,
            "dense_bytes": dense_bytes,
            "factor_bytes": factor_bytes,
        })
    return result


def choose(blocks: list[dict], fraction: float) -> list[bool]:
    count = int(round(len(blocks) * fraction))
    order = sorted(range(len(blocks)), key=lambda i: (blocks[i]["residual"], blocks[i]["factor_bytes"] - blocks[i]["dense_bytes"]))
    selected = [False] * len(blocks)
    for i in order[:count]:
        # Never select a block if the chosen factor is larger than its exact Q4 artifact.
        if blocks[i]["factor_bytes"] < blocks[i]["dense_bytes"]:
            selected[i] = True
    return selected


def apply_blocks(x: np.ndarray, weight: np.ndarray, blocks: list[dict], selected: list[bool]) -> np.ndarray:
    out = np.zeros((len(x), weight.shape[0]), dtype=np.float32)
    for block, use_factor in zip(blocks, selected):
        start, end = block["start"], block["end"]
        part = (x @ block["right"].T) @ block["left"].T if use_factor else x @ weight[start:end].T
        out[:, start:end] = part
    return out


def swiglu(g: np.ndarray, u: np.ndarray) -> np.ndarray:
    return (g / (1.0 + np.exp(-g))) * u


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    err = np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)
    return float(err.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective exact-main + low-rank residual FFN weight split")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--fractions", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--families", default="down_only,gate_up_only,all_three")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, _, _, train_y = load_data(args.calibration_root, args.layer)
    test_x, _, _, test_y = load_data(args.holdout_root, args.layer)
    (wg, wg_bytes), (wu, wu_bytes), (wd, wd_bytes) = load_weights(args.model, args.layer)
    exact_train = swiglu(train_x @ wg.T, train_x @ wu.T) @ wd.T
    exact_test = swiglu(test_x @ wg.T, test_x @ wu.T) @ wd.T
    weights = {"gate": (wg, wg_bytes), "up": (wu, wu_bytes), "down": (wd, wd_bytes)}
    rows = []
    for rank in [int(v) for v in args.ranks.split(",") if v.strip()]:
        prepared = {name: prepare_blocks(weight, nbytes, args.block, rank) for name, (weight, nbytes) in weights.items()}
        for fraction in [float(v) for v in args.fractions.split(",") if v.strip()]:
            selections = {name: choose(blocks, fraction) for name, blocks in prepared.items()}
            for family in [v.strip() for v in args.families.split(",") if v.strip()]:
                use = {
                    "gate": family in {"gate_up_only", "all_three"},
                    "up": family in {"gate_up_only", "all_three"},
                    "down": family in {"down_only", "all_three"},
                }
                def run(x: np.ndarray) -> np.ndarray:
                    g = apply_blocks(x, wg, prepared["gate"], selections["gate"]) if use["gate"] else x @ wg.T
                    u = apply_blocks(x, wu, prepared["up"], selections["up"]) if use["up"] else x @ wu.T
                    h = swiglu(g, u)
                    return apply_blocks(h, wd, prepared["down"], selections["down"]) if use["down"] else h @ wd.T

                pred_train, pred_test = run(train_x), run(test_x)
                dense_bytes = sum(nbytes for _, nbytes in weights.values())
                stored_bytes = 0
                detail = {}
                for name, (weight, nbytes) in weights.items():
                    blocks = prepared[name]
                    selected = selections[name] if use[name] else [False] * len(blocks)
                    value = sum(block["factor_bytes"] if flag else block["dense_bytes"] for block, flag in zip(blocks, selected))
                    stored_bytes += value
                    detail[name] = {
                        "selected_blocks": int(sum(selected)),
                        "total_blocks": len(blocks),
                        "selected_fraction": float(sum(selected) / max(len(blocks), 1)),
                        "stored_bytes": int(value),
                        "dense_bytes": int(nbytes),
                        "mean_selected_svd_residual": float(np.mean([b["residual"] for b, flag in zip(blocks, selected) if flag])) if any(selected) else 0.0,
                    }
                rows.append({
                    "family": family,
                    "rank": rank,
                    "requested_fraction": fraction,
                    "train_rel_l2_vs_capture": rel_l2(pred_train, train_y),
                    "holdout_rel_l2_vs_capture": rel_l2(pred_test, test_y),
                    "train_rel_l2_vs_exact_replay": rel_l2(pred_train, exact_train),
                    "holdout_rel_l2_vs_exact_replay": rel_l2(pred_test, exact_test),
                    "dense_weight_bytes": int(dense_bytes),
                    "stored_weight_bytes": int(stored_bytes),
                    "weight_reduction": float(1.0 - stored_bytes / dense_bytes),
                    "blocks": detail,
                })
    result = {
        "experiment": "selective_exact_main_low_rank_ffn_split",
        "formula": "selected W_b ~= U_b V_b; unselected W_b exact; SwiGLU retained; block outputs add in layer space",
        "runtime_properties": {
            "lookup": False,
            "main_path": "exact blocks",
            "split_path": "two matmuls for selected low-rank blocks",
            "merge": "write/add block outputs in intermediate or hidden dimension",
            "selection": "offline choose blocks with lowest SVD residual and factor_bytes < exact Q4 bytes",
        },
        "layer": args.layer,
        "block": args.block,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "dense_weight_bytes": int(wg_bytes + wu_bytes + wd_bytes),
        "exact_weight_replay_rel_l2_train": rel_l2(exact_train, train_y),
        "exact_weight_replay_rel_l2_holdout": rel_l2(exact_test, test_y),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
