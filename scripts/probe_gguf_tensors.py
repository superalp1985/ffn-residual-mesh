from __future__ import annotations

import argparse
import json
from pathlib import Path

from gguf import GGUFReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--pattern", default="ffn_")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    reader = GGUFReader(str(args.model))
    rows = []
    for tensor in reader.tensors:
        if args.pattern not in tensor.name:
            continue
        row = {
            "name": tensor.name,
            "shape": [int(x) for x in tensor.shape],
            "type": str(tensor.tensor_type),
            "n_bytes": int(tensor.n_bytes),
            "data_offset": int(getattr(tensor, "data_offset", -1)),
        }
        data = getattr(tensor, "data", None)
        if data is not None:
            row["data_type"] = str(getattr(data, "dtype", type(data)))
            row["data_shape"] = list(getattr(data, "shape", ()))
            row["data_nbytes"] = int(getattr(data, "nbytes", 0))
            row["first_values"] = data.reshape(-1)[:16].tolist()
        rows.append(row)
        if len(rows) >= args.limit:
            break
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
