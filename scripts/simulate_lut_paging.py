from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def load(root: Path, item: dict) -> np.ndarray:
    raw = np.fromfile(root / item["file"], dtype=np.float32)
    return raw.reshape(tuple(int(x) for x in item["shape"])[::-1])


def collect(root: Path, layer: int) -> list[np.ndarray]:
    values = []
    for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        manifest = json.loads((prompt_dir / "ffn_tensors.json").read_text(encoding="utf-8"))
        items = {item["name"]: item for item in manifest["tensors"]}
        name = f"ffn_swiglu-{layer}"
        if name in items:
            values.append(load(prompt_dir, items[name]))
    return values


def fit_codebooks(train: list[np.ndarray], block_dim: int, k: int, seed: int) -> list[np.ndarray]:
    sample = np.concatenate(train)
    centers = []
    for j, start in enumerate(range(0, sample.shape[1], block_dim)):
        block = sample[:, start:start + block_dim]
        km = MiniBatchKMeans(n_clusters=min(k, len(block)), random_state=seed + j, n_init=2,
                             batch_size=min(256, len(block)), max_iter=60)
        km.fit(block)
        centers.append(km.cluster_centers_.astype(np.float32))
    return centers


def encode(values: np.ndarray, centers: list[np.ndarray], block_dim: int) -> np.ndarray:
    codes = []
    for j, start in enumerate(range(0, values.shape[1], block_dim)):
        block = values[:, start:start + block_dim]
        center = centers[j]
        d2 = ((block[:, None, :] - center[None, :, :]) ** 2).sum(axis=2)
        codes.append(np.argmin(d2, axis=1))
    return np.stack(codes, axis=1)


def simulate(codes: np.ndarray, k: int, page_size: int, block_size: int, vector_bytes: int, layout: str, window: int) -> dict:
    pages_per_vector = max(1, int(np.ceil(vector_bytes / page_size)))
    pages_per_super = max(1, block_size // page_size)
    n_blocks = codes.shape[1]
    requested_pages = set()
    for block, code in enumerate(codes[:window].T):
        for token_code in code:
            entry = block * k + int(token_code) if layout == "block_major" else int(token_code) * n_blocks + block
            start_page = entry * pages_per_vector
            requested_pages.update(range(start_page, start_page + pages_per_vector))
    requested = len(requested_pages) * page_size
    supers = {page // pages_per_super for page in requested_pages}
    transferred = len(supers) * block_size
    table_bytes = codes.shape[1] * k * vector_bytes
    transferred = min(transferred, table_bytes)
    return {
        "tokens": int(len(codes)),
        "requested_bytes": int(requested),
        "transferred_bytes": int(transferred),
        "waste_ratio": float((transferred - requested) / max(requested, 1)),
        "unique_vectors": int(len(requested_pages) // pages_per_vector),
        "superblocks": int(len(supers)),
        "table_bytes": int(table_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default="0,10,18,22,23")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=4096)
    args = parser.parse_args()
    rows = []
    for layer in [int(x) for x in args.layers.split(",") if x.strip()]:
        sequences = collect(args.calibration_root, layer)
        if not sequences:
            continue
        centers = fit_codebooks(sequences, args.block, args.k, 7000 + layer)
        for sequence_id, values in enumerate(sequences):
            codes = encode(values, centers, args.block)
            for window in (1, 4, 8, 16, len(codes)):
                window = min(window, len(codes))
                for layout in ("block_major", "code_major"):
                    for superblock in (64 * 1024, 256 * 1024, 1024 * 1024):
                        rows.append({"layer": layer, "sequence": sequence_id, "window": window, "layout": layout, "superblock": superblock,
                                     "result": simulate(codes, args.k, args.page_size, superblock, 2048 * 2, layout, window)})
    result = {"calibration_root": str(args.calibration_root), "page_size": args.page_size, "k": args.k, "block": args.block, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for layout in ("block_major", "code_major"):
        for window in (1, 4, 8, 16):
            subset = [r["result"] for r in rows if r["layout"] == layout and r["window"] == window and r["superblock"] == 256 * 1024]
            if subset:
                print(layout, window, {"mean_waste": float(np.mean([x["waste_ratio"] for x in subset])), "mean_transfer": float(np.mean([x["transferred_bytes"] for x in subset]))})


if __name__ == "__main__":
    main()
