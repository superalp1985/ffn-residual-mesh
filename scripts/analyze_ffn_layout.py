from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.metadata.read_text(encoding="utf-8"))
    tensors = payload["tensors"]
    total_bytes = sum(int(t["n_bytes"]) for t in tensors)
    ffn = [t for t in tensors if ".ffn_" in t["name"]]
    by_layer = defaultdict(list)
    for tensor in ffn:
        layer = tensor["name"].split(".")[1]
        by_layer[layer].append(tensor)

    pages = [4096, 16384, 65536]
    superblocks = [256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
    layer_rows = []
    for layer, rows in sorted(by_layer.items(), key=lambda item: int(item[0])):
        layer_bytes = sum(int(t["n_bytes"]) for t in rows)
        layer_rows.append(
            {
                "layer": int(layer),
                "bytes": layer_bytes,
                "MiB": round(layer_bytes / 2**20, 4),
                "tensors": [
                    {
                        "name": t["name"],
                        "shape": t["shape"],
                        "dtype": t["dtype"],
                        "bytes": int(t["n_bytes"]),
                    }
                    for t in rows
                ],
                "logical_pages": {str(p): math.ceil(layer_bytes / p) for p in pages},
                "superblocks": {str(s): math.ceil(layer_bytes / s) for s in superblocks},
            }
        )

    result = {
        "model": payload["model"],
        "tensor_count": len(tensors),
        "total_bytes": total_bytes,
        "total_MiB": round(total_bytes / 2**20, 4),
        "ffn_bytes": sum(int(t["n_bytes"]) for t in ffn),
        "ffn_MiB": round(sum(int(t["n_bytes"]) for t in ffn) / 2**20, 4),
        "ffn_fraction": sum(int(t["n_bytes"]) for t in ffn) / total_bytes,
        "layers": layer_rows,
        "output_vector_bytes": {"fp16": 2048 * 2, "fp32": 2048 * 4},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "total_MiB": result["total_MiB"],
        "ffn_MiB": result["ffn_MiB"],
        "ffn_fraction": round(result["ffn_fraction"], 4),
        "layers": len(layer_rows),
    }))


if __name__ == "__main__":
    main()
