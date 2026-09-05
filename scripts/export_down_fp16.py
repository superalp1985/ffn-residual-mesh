from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one GGUF FFN down projection as row-major fp16")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reader = GGUFReader(str(args.model))
    tensor = next(item for item in reader.tensors if item.name == f"blk.{args.layer}.ffn_down.weight")
    matrix = dequantize(tensor.data, tensor.tensor_type).astype(np.float16, copy=False)
    # gguf exposes down as [hidden, ffn] for this model; keep the file row-major.
    np.ascontiguousarray(matrix).tofile(args.out)
    print({"path": str(args.out), "shape": list(matrix.shape), "bytes": int(matrix.nbytes), "dtype": "float16"})


if __name__ == "__main__":
    main()
