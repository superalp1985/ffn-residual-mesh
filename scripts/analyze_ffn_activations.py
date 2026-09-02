from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_tensor(root: Path, item: dict) -> np.ndarray:
    raw = np.fromfile(root / item["file"], dtype=np.float32)
    shape = tuple(int(x) for x in item["shape"])
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"size mismatch for {item['name']}: {raw.size} != {shape}")
    # ggml stores ne[0] as the contiguous dimension; expose rows as tokens.
    return raw.reshape(shape[::-1])


def effective_ranks(values: np.ndarray) -> dict[str, int]:
    matrix = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    total = float(energy.sum())
    if total <= 0:
        return {"90": 0, "95": 0, "99": 0}
    cumulative = np.cumsum(energy) / total
    return {
        str(level): int(np.searchsorted(cumulative, level / 100.0) + 1)
        for level in (90, 95, 99)
    }


def energy_fraction(values: np.ndarray, k: int) -> float:
    flat = np.abs(values).reshape(-1)
    if flat.size == 0:
        return 0.0
    k = min(k, flat.size)
    selected = np.partition(flat, -k)[-k:]
    return float(selected.sum() / max(float(flat.sum()), 1e-12))


def summarize(name: str, values: np.ndarray) -> dict:
    token_norms = np.linalg.norm(values, axis=1)
    delta = np.diff(values, axis=0)
    delta_norms = np.linalg.norm(delta, axis=1) if len(values) > 1 else np.array([], dtype=np.float32)
    mean_abs = np.abs(values).mean(axis=0)
    top_indices = [max(1, int(round(values.shape[1] * ratio))) for ratio in (0.01, 0.05, 0.10, 0.20)]
    return {
        "shape": list(values.shape),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mean_abs": float(np.abs(values).mean()),
        "mean_token_l2": float(token_norms.mean()),
        "std_token_l2": float(token_norms.std()),
        "adjacent_delta_over_current": float(delta_norms.mean() / max(token_norms.mean(), 1e-12)) if len(delta_norms) else 0.0,
        "positive_fraction": float((values > 0).mean()),
        "channel_mean_abs_p95": float(np.percentile(mean_abs, 95)),
        "channel_mean_abs_p99": float(np.percentile(mean_abs, 99)),
        "top_energy_fraction": {
            f"top_{int(ratio * 100)}pct": energy_fraction(values, k)
            for ratio, k in zip((0.01, 0.05, 0.10, 0.20), top_indices)
        },
        "rank_on_tokens": effective_ranks(values),
        "rank_on_adjacent_delta": effective_ranks(delta) if len(delta) > 1 else {"90": 0, "95": 0, "99": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.probe_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
    by_layer: dict[int, dict[str, tuple[dict, np.ndarray]]] = {}
    for item in manifest["tensors"]:
        base, layer_text = item["name"].rsplit("-", 1)
        layer = int(layer_text)
        by_layer.setdefault(layer, {})[base] = (item, load_tensor(args.probe_dir, item))

    layers = []
    for layer, tensors in sorted(by_layer.items()):
        row = {"layer": layer}
        for name, (_, values) in sorted(tensors.items()):
            row[name] = summarize(name, values)
        layers.append(row)

    result = {
        "probe_dir": str(args.probe_dir),
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    swiglu = [row["ffn_swiglu"] for row in layers if "ffn_swiglu" in row]
    out = [row["ffn_out"] for row in layers if "ffn_out" in row]
    print(json.dumps({
        "layers": len(layers),
        "swiglu_adjacent_delta_mean": round(float(np.mean([x["adjacent_delta_over_current"] for x in swiglu])), 6),
        "out_adjacent_delta_mean": round(float(np.mean([x["adjacent_delta_over_current"] for x in out])), 6),
        "swiglu_rank95_mean": round(float(np.mean([x["rank_on_tokens"]["95"] for x in swiglu])), 3),
        "out_rank95_mean": round(float(np.mean([x["rank_on_tokens"]["95"] for x in out])), 3),
    }))


if __name__ == "__main__":
    main()
