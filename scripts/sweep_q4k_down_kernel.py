from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resident_residual_cuda import DirectQ4Projection  # noqa: E402


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.split(",") if value)
    if not values:
        raise ValueError("at least one integer is required")
    return values


def load_q4k_down(model: Path, layer: int) -> tuple[np.ndarray, int, int]:
    reader = GGUFReader(model)
    try:
        tensor = next(
            item for item in reader.tensors
            if item.name == f"blk.{layer}.ffn_down.weight"
        )
        quant = GGMLQuantizationType(int(tensor.tensor_type))
        if quant is not GGMLQuantizationType.Q4_K:
            raise ValueError(f"Q4_K down tensor required, found {quant.name}")
        # The GGUF reader owns the mmap. Copy before it is closed.
        return (
            np.array(tensor.data, copy=True),
            int(tensor.shape[0]),
            int(tensor.shape[1]),
        )
    finally:
        reader.data._mmap.close()


def measure(
    raw: np.ndarray,
    cols: int,
    rows: int,
    *,
    block_rows: int,
    num_warps: int,
    chunk_cols: int,
    warmup: int,
    repeats: int,
    activation: torch.Tensor,
) -> dict[str, object]:
    projection = DirectQ4Projection(
        raw,
        cols,
        block_rows=block_rows,
        num_warps=num_warps,
        chunk_cols=chunk_cols,
    )
    try:
        for _ in range(warmup):
            projection.launch(activation)
        torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            projection.launch(activation)
            end.record()
            end.synchronize()
            values.append(float(begin.elapsed_time(end)))
        return {
            "block_rows": block_rows,
            "num_warps": num_warps,
            "chunk_cols": chunk_cols,
            "gpu_ms_median": float(np.median(values)),
            "gpu_ms_p95": float(np.percentile(values, 95)),
            "samples_ms": values,
        }
    finally:
        del projection
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 1 or args.repeats < 3:
        raise ValueError("warmup must be positive and repeats must be at least 3")
    raw, cols, rows = load_q4k_down(args.model, args.layer)
    activation = torch.from_numpy(
        np.random.default_rng(args.seed).standard_normal(cols).astype(np.float32)
    ).cuda()
    configurations = []
    for block_rows in parse_int_list(args.block_rows):
        for num_warps in parse_int_list(args.num_warps):
            for chunk_cols in parse_int_list(args.chunk_cols):
                configurations.append(
                    measure(
                        raw,
                        cols,
                        rows,
                        block_rows=block_rows,
                        num_warps=num_warps,
                        chunk_cols=chunk_cols,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        activation=activation,
                    )
                )
    configurations.sort(key=lambda item: item["gpu_ms_median"])
    return {
        "status": "q4k_down_kernel_shape_sweep",
        "layer": args.layer,
        "dimensions": {"rows": rows, "cols": cols},
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "configurations": configurations,
        "best": configurations[0],
        "scope": (
            "Exact Q4_K decode kernel-shape sweep. CUDA event time is a "
            "device-kernel measure, not end-to-end inference throughput."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep exact Q4_K down GEMV launch shapes"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--chunk-cols", default="256,512,1024,2048")
    parser.add_argument("--block-rows", default="1,2,4")
    parser.add_argument("--num-warps", default="2,4,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
