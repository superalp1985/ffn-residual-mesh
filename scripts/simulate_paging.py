from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def selected_pages(page_count: int, count: int, mode: str, rng: random.Random) -> set[int]:
    if count <= 0:
        return set()
    if mode == "random":
        return set(rng.sample(range(page_count), min(count, page_count)))
    # Clustered requests approximate a router selecting nearby residual blocks.
    width = min(count, page_count)
    start = rng.randrange(0, max(1, page_count - width + 1))
    return set(range(start, start + width))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    page_sizes = [4096, 16384, 65536]
    block_sizes = [256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
    fractions = [0.05, 0.10, 0.20]
    rows = []
    for layer in layout["layers"]:
        layer_bytes = int(layer["bytes"])
        for page_size in page_sizes:
            page_count = math.ceil(layer_bytes / page_size)
            for fraction in fractions:
                count = max(1, math.ceil(page_count * fraction))
                for mode in ("random", "clustered"):
                    for block_size in block_sizes:
                        pages_per_block = max(1, block_size // page_size)
                        rng = random.Random(args.seed + int(layer["layer"]) * 1000 + page_size + int(fraction * 100) + block_size)
                        pages = selected_pages(page_count, count, mode, rng)
                        blocks = {page // pages_per_block for page in pages}
                        requested = min(layer_bytes, len(pages) * page_size)
                        transferred = min(layer_bytes, len(blocks) * block_size)
                        rows.append({
                            "layer": int(layer["layer"]),
                            "layer_bytes": layer_bytes,
                            "page_size": page_size,
                            "fraction": fraction,
                            "mode": mode,
                            "block_size": block_size,
                            "requested_bytes": requested,
                            "transferred_bytes": transferred,
                            "waste_ratio": round((transferred - requested) / max(1, requested), 6),
                            "blocks": len(blocks),
                        })

    result = {
        "seed": args.seed,
        "rows": rows,
        "summary": {
            "best_by_mode": {
                mode: min(
                    (row for row in rows if row["mode"] == mode),
                    key=lambda row: row["waste_ratio"],
                )
                for mode in ("random", "clustered")
            }
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
