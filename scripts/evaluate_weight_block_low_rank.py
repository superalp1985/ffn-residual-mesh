from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

from evaluate_additive_holdout import load_tensor


def load_data(root: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    wanted = [f"attn_post_norm-{layer}", f"ffn_gate-{layer}", f"ffn_up-{layer}", f"ffn_out-{layer}"]
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: load_tensor(prompt_dir, item) for item in manifest["tensors"] if item["name"] in wanted}
        if all(name in tensors for name in wanted):
            rows.append(tuple(tensors[name] for name in wanted))
    if not rows:
        raise ValueError(f"layer {layer} missing from {root}")
    return tuple(np.concatenate([row[i] for row in rows], axis=0) for i in range(4))  # type: ignore[return-value]


def load_weights(model: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = GGUFReader(str(model))
    result = []
    for name in ("gate", "up", "down"):
        tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_{name}.weight")
        result.append(dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False))
    return tuple(result)  # type: ignore[return-value]


def factor_matrix(weight: np.ndarray, block: int, rank: int) -> list[tuple[np.ndarray, np.ndarray]]:
    factors = []
    for start in range(0, weight.shape[0], block):
        part = weight[start : start + block]
        u, s, vt = np.linalg.svd(part, full_matrices=False)
        r = min(rank, len(s))
        # Fold singular values into U so runtime only performs two matmuls.
        left = (u[:, :r] * s[:r]).astype(np.float32, copy=False)
        right = vt[:r].astype(np.float32, copy=False)
        factors.append((left, right))
    return factors


def apply_matrix(x: np.ndarray, factors: list[tuple[np.ndarray, np.ndarray]], block: int, out_dim: int) -> np.ndarray:
    out = np.zeros((len(x), out_dim), dtype=np.float32)
    offset = 0
    for left, right in factors:
        width = left.shape[0]
        out[:, offset : offset + width] = (x @ right.T) @ left.T
        offset += width
    return out


def exact_matrix(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x @ weight.T


def swiglu(g: np.ndarray, u: np.ndarray) -> np.ndarray:
    return (g / (1.0 + np.exp(-g))) * u


def artifact_bytes(weight: np.ndarray, factors: list[tuple[np.ndarray, np.ndarray]], dtype_bytes: int = 2) -> int:
    elems = sum(left.size + right.size for left, right in factors)
    return int(elems * dtype_bytes)


def factor_cost(weight: np.ndarray, block: int, rank: int) -> dict[str, int]:
    blocks = (weight.shape[0] + block - 1) // block
    dense = weight.shape[0] * weight.shape[1]
    factored = sum(min(block, weight.shape[0] - i * block) * rank + rank * weight.shape[1] for i in range(blocks))
    return {
        "dense_mac": dense,
        "factored_mac": int(factored),
        "mac_multiplier": float(factored / max(dense, 1)),
        "blocks": blocks,
    }


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    err = np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-6)
    return float(err.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Block low-rank FFN weight decomposition")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--blocks", default="256,512")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_x, train_g, train_u, train_y = load_data(args.calibration_root, args.layer)
    test_x, test_g, test_u, test_y = load_data(args.holdout_root, args.layer)
    wg, wu, wd = load_weights(args.model, args.layer)
    exact_train_y = swiglu(exact_matrix(train_x, wg), exact_matrix(train_x, wu)) @ wd.T
    exact_test_y = swiglu(exact_matrix(test_x, wg), exact_matrix(test_x, wu)) @ wd.T
    rows = []
    for block in [int(v) for v in args.blocks.split(",") if v.strip()]:
        for rank in [int(v) for v in args.ranks.split(",") if v.strip()]:
            fg, fu, fd = factor_matrix(wg, block, rank), factor_matrix(wu, block, rank), factor_matrix(wd, block, rank)
            cases = {
                "down_only": (None, None, fd),
                "gate_up_only": (fg, fu, None),
                "all_three": (fg, fu, fd),
            }
            for family, (cg, cu, cd) in cases.items():
                def run(x: np.ndarray, exact_g: np.ndarray, exact_u: np.ndarray) -> np.ndarray:
                    g = exact_matrix(x, wg) if cg is None else apply_matrix(x, cg, block, wg.shape[0])
                    u = exact_matrix(x, wu) if cu is None else apply_matrix(x, cu, block, wu.shape[0])
                    h = swiglu(g, u)
                    return h @ wd.T if cd is None else apply_matrix(h, cd, block, wd.shape[0])

                pred_train, pred_test = run(train_x, train_g, train_u), run(test_x, test_g, test_u)
                weight_bytes = {
                    "gate": wg.nbytes,
                    "up": wu.nbytes,
                    "down": wd.nbytes,
                }
                stored = {
                    "gate": artifact_bytes(wg, fg) if cg is not None else wg.nbytes,
                    "up": artifact_bytes(wu, fu) if cu is not None else wu.nbytes,
                    "down": artifact_bytes(wd, fd) if cd is not None else wd.nbytes,
                }
                rows.append({
                    "family": family,
                    "block": block,
                    "rank": rank,
                    "train_rel_l2_vs_capture": rel_l2(pred_train, train_y),
                    "holdout_rel_l2_vs_capture": rel_l2(pred_test, test_y),
                    "train_rel_l2_vs_exact_weight_replay": rel_l2(pred_train, exact_train_y),
                    "holdout_rel_l2_vs_exact_weight_replay": rel_l2(pred_test, exact_test_y),
                    "dense_weight_bytes": int(sum(weight_bytes.values())),
                    "factored_weight_bytes": int(sum(stored.values())),
                    "weight_reduction": float(1.0 - sum(stored.values()) / sum(weight_bytes.values())),
                    "mac": {"gate": factor_cost(wg, block, rank) if cg is not None else {"dense_mac": int(wg.size), "factored_mac": int(wg.size), "mac_multiplier": 1.0}, "up": factor_cost(wu, block, rank) if cu is not None else {"dense_mac": int(wu.size), "factored_mac": int(wu.size), "mac_multiplier": 1.0}, "down": factor_cost(wd, block, rank) if cd is not None else {"dense_mac": int(wd.size), "factored_mac": int(wd.size), "mac_multiplier": 1.0}},
                })
    result = {
        "experiment": "block_low_rank_ffn_weight_split",
        "formula": "W_b ~= U_b V_b; FFN(x)=W_down * (SiLU(W_gate*x) * (W_up*x))",
        "runtime_properties": {
            "lookup": False,
            "goal": "increase arithmetic and reduce dynamic weight bytes",
            "split": "factor selected output-row blocks; each factor uses two matmuls",
            "merge": "sum block outputs in hidden space",
            "exact_fallback": "retain original quantized dense weights for any rejected layer/token",
        },
        "layer": args.layer,
        "train_samples": len(train_x),
        "holdout_samples": len(test_x),
        "dense_weight_bytes": int(wg.nbytes + wu.nbytes + wd.nbytes),
        "exact_weight_replay_rel_l2_train": rel_l2(exact_train_y, train_y),
        "exact_weight_replay_rel_l2_holdout": rel_l2(exact_test_y, test_y),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
