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
    for start in range(0, weight.shape[0], block):
        end = min(start + block, weight.shape[0])
        part = weight[start:end]
        u, s, vt = np.linalg.svd(part, full_matrices=False)
        r = min(rank, len(s))
        left = (u[:, :r] * s[:r]).astype(np.float32, copy=False)
        right = vt[:r].astype(np.float32, copy=False)
        dense_bytes = int(round(quant_bytes * (end - start) / weight.shape[0]))
        result.append({
            "start": start,
            "end": end,
            "left": left,
            "right": right,
            "dense_bytes": dense_bytes,
            "factor_bytes": int((left.size + right.size) * 2),
            "svd_residual": float(np.linalg.norm(part - left @ right) / max(np.linalg.norm(part), 1e-6)),
        })
    return result


def apply_all(x: np.ndarray, weight: np.ndarray, blocks: list[dict], selected: list[bool]) -> np.ndarray:
    out = np.zeros((len(x), weight.shape[0]), dtype=np.float32)
    for b, use in zip(blocks, selected):
        start, end = b["start"], b["end"]
        out[:, start:end] = (x @ b["right"].T) @ b["left"].T if use else x @ weight[start:end].T
    return out


def apply_one(x: np.ndarray, block: dict) -> np.ndarray:
    return (x @ block["right"].T) @ block["left"].T


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def relative_output_delta(delta: np.ndarray, exact_y: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(exact_y, axis=1), 1e-6)))


def sensitivity_scores(x: np.ndarray, exact_g: np.ndarray, exact_u: np.ndarray, exact_h: np.ndarray, exact_y: np.ndarray, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, blocks: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for name, weight in (("gate", wg), ("up", wu)):
        values = []
        for b in blocks[name]:
            start, end = b["start"], b["end"]
            changed = apply_one(x, b)
            if name == "gate":
                delta_h = silu(changed) * exact_u[:, start:end] - exact_h[:, start:end]
            else:
                delta_h = silu(exact_g[:, start:end]) * changed - exact_h[:, start:end]
            delta_y = delta_h @ wd[:, start:end].T
            values.append(relative_output_delta(delta_y, exact_y))
        scores[name] = np.asarray(values, dtype=np.float32)

    values = []
    for b in blocks["down"]:
        start, end = b["start"], b["end"]
        exact_part = exact_h @ wd[start:end].T
        approx_part = apply_one(exact_h, b)
        delta = np.zeros_like(exact_y)
        delta[:, start:end] = approx_part - exact_part
        values.append(relative_output_delta(delta, exact_y))
    scores["down"] = np.asarray(values, dtype=np.float32)
    return scores


def choose(blocks: list[dict], scores: np.ndarray, fraction: float) -> list[bool]:
    count = int(round(len(blocks) * fraction))
    order = sorted(range(len(blocks)), key=lambda i: (float(scores[i]), blocks[i]["factor_bytes"] - blocks[i]["dense_bytes"]))
    selected = [False] * len(blocks)
    for i in order[:count]:
        if blocks[i]["factor_bytes"] < blocks[i]["dense_bytes"]:
            selected[i] = True
    return selected


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation-sensitive selective FFN weight split")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--fractions", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--families", default="down_only,gate_up_only,all_three")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, _, _, train_y = load_data(args.calibration_root, args.layer)
    test_x, _, _, test_y = load_data(args.holdout_root, args.layer)
    (wg, wg_bytes), (wu, wu_bytes), (wd, wd_bytes) = load_weights(args.model, args.layer)
    exact_g, exact_u = train_x @ wg.T, train_x @ wu.T
    exact_h = silu(exact_g) * exact_u
    exact_y = exact_h @ wd.T
    holdout_exact = silu(test_x @ wg.T) * (test_x @ wu.T) @ wd.T
    weights = {"gate": (wg, wg_bytes), "up": (wu, wu_bytes), "down": (wd, wd_bytes)}
    rows = []
    for rank in [int(v) for v in args.ranks.split(",") if v.strip()]:
        blocks = {name: prepare_blocks(weight, nbytes, args.block, rank) for name, (weight, nbytes) in weights.items()}
        scores = sensitivity_scores(train_x, exact_g, exact_u, exact_h, exact_y, wg, wu, wd, blocks)
        for fraction in [float(v) for v in args.fractions.split(",") if v.strip()]:
            selections = {name: choose(blocks[name], scores[name], fraction) for name in blocks}
            for family in [v.strip() for v in args.families.split(",") if v.strip()]:
                use = {"gate": family in {"gate_up_only", "all_three"}, "up": family in {"gate_up_only", "all_three"}, "down": family in {"down_only", "all_three"}}

                def run(x: np.ndarray) -> np.ndarray:
                    g = apply_all(x, wg, blocks["gate"], selections["gate"]) if use["gate"] else x @ wg.T
                    u = apply_all(x, wu, blocks["up"], selections["up"]) if use["up"] else x @ wu.T
                    h = silu(g) * u
                    return apply_all(h, wd, blocks["down"], selections["down"]) if use["down"] else h @ wd.T

                pred_train, pred_test = run(train_x), run(test_x)
                dense_bytes = sum(item[1] for item in weights.values())
                stored_bytes = 0
                detail = {}
                for name, (_, nbytes) in weights.items():
                    selected = selections[name] if use[name] else [False] * len(blocks[name])
                    value = sum(b["factor_bytes"] if flag else b["dense_bytes"] for b, flag in zip(blocks[name], selected))
                    stored_bytes += value
                    detail[name] = {
                        "selected_blocks": int(sum(selected)),
                        "total_blocks": len(blocks[name]),
                        "selected_fraction": float(sum(selected) / max(len(blocks[name]), 1)),
                        "stored_bytes": int(value),
                        "dense_bytes": int(nbytes),
                        "mean_activation_score": float(np.mean([scores[name][i] for i, flag in enumerate(selected) if flag])) if any(selected) else 0.0,
                    }
                rows.append({
                    "family": family,
                    "rank": rank,
                    "requested_fraction": fraction,
                    "train_rel_l2_vs_capture": rel_l2(pred_train, train_y),
                    "holdout_rel_l2_vs_capture": rel_l2(pred_test, test_y),
                    "train_rel_l2_vs_exact_replay": rel_l2(pred_train, exact_y),
                    "holdout_rel_l2_vs_exact_replay": rel_l2(pred_test, holdout_exact),
                    "dense_weight_bytes": int(dense_bytes),
                    "stored_weight_bytes": int(stored_bytes),
                    "weight_reduction": float(1.0 - stored_bytes / dense_bytes),
                    "blocks": detail,
                })
    result = {
        "experiment": "activation_sensitive_selective_ffn_weight_split",
        "formula": "W_b ~= U_b V_b only for blocks with low E_x[||delta_y,b||/||y||]; other blocks exact",
        "runtime_properties": {
            "lookup": False,
            "selection": "calibration-set output sensitivity, not Frobenius weight error",
            "goal": "spend extra matmuls only on low-impact blocks to reduce dynamic weight bytes",
        },
        "layer": args.layer,
        "block": args.block,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "dense_weight_bytes": int(wg_bytes + wu_bytes + wd_bytes),
        "exact_weight_replay_rel_l2_train": rel_l2(exact_y, train_y),
        "exact_weight_replay_rel_l2_holdout": rel_l2(holdout_exact, test_y),
        "sensitivity_scores": {name: scores[name].tolist() for name in scores},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
