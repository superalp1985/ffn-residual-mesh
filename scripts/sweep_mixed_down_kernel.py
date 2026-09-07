from __future__ import annotations

"""Measure exact mixed-quantized FFN down projections from a GGUF model.

This module also owns the projection factory used by the resident FFN probes,
so format dispatch is tested once rather than duplicated per benchmark.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resident_residual_cuda import (  # noqa: E402
    DirectIQ4NLProjection,
    DirectIQ4XSProjection,
    DirectQ4Projection,
    DirectQ5Projection,
)


def build_projection(
    tensor,
    *,
    block_rows: int = 2,
    num_warps: int = 2,
    block_qblocks: int = 4,
):
    """Build an exact GPU down-projection for a supported GGUF tensor."""
    quant = GGMLQuantizationType(int(tensor.tensor_type))
    cols = int(tensor.shape[0])
    if quant is GGMLQuantizationType.IQ4_NL:
        projection = DirectIQ4NLProjection(tensor.data, cols)
    elif quant is GGMLQuantizationType.Q4_K:
        projection = DirectQ4Projection(
            tensor.data,
            cols,
            kernel="grouped",
            block_rows=block_rows,
            num_warps=num_warps,
            block_qblocks=block_qblocks,
        )
    elif quant is GGMLQuantizationType.Q5_K:
        projection = DirectQ5Projection(
            tensor.data,
            cols,
            block_rows=block_rows,
            num_warps=num_warps,
            block_qblocks=block_qblocks,
        )
    elif quant is GGMLQuantizationType.IQ4_XS:
        projection = DirectIQ4XSProjection(
            tensor.data,
            cols,
            block_rows=block_rows,
            num_warps=num_warps,
            block_qblocks=block_qblocks,
        )
    else:
        raise ValueError(f"unsupported down tensor type: {quant.name}")
    return projection, {
        "quant_type": quant.name,
        "rows": projection.rows,
        "cols": projection.cols,
        "kernel": getattr(projection, "kernel", "iq4_nl_fused"),
    }


def load_down(model: Path, layer: int):
    reader = GGUFReader(model)
    try:
        tensor = next(
            item for item in reader.tensors
            if item.name == f"blk.{layer}.ffn_down.weight"
        )
        # The GGUF reader owns the mmap. Projection storage must outlive it.
        return np.array(tensor.data, copy=True), int(tensor.tensor_type), tuple(tensor.shape)
    finally:
        reader.data._mmap.close()


def measure(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 1 or args.repeats < 3:
        raise ValueError("warmup must be positive and repeats must be at least 3")

    raw, tensor_type, shape = load_down(args.model, args.layer)

    projection, metadata = build_projection(
        SimpleNamespace(data=raw, tensor_type=tensor_type, shape=shape),
        block_rows=args.block_rows,
        num_warps=args.num_warps,
        block_qblocks=args.block_qblocks,
    )
    activation = torch.from_numpy(
        np.random.default_rng(args.seed).standard_normal(projection.cols).astype(np.float32)
    ).cuda()
    try:
        for _ in range(args.warmup):
            projection.launch(activation)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            projection.launch(activation)
            end.record()
            end.synchronize()
            samples.append(float(begin.elapsed_time(end)))
        return {
            "status": "mixed_quant_down_kernel_measurement",
            "layer": args.layer,
            "projection": metadata,
            "gpu_ms_median": float(np.median(samples)),
            "gpu_ms_p95": float(np.percentile(samples, 95)),
            "samples_ms": samples,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "scope": (
                "Exact GGUF quantized down GEMV. CUDA event time is a device "
                "kernel measure, not end-to-end generation throughput."
            ),
        }
    finally:
        del projection
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure an exact mixed-quantized FFN down GEMV"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--block-rows", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--block-qblocks", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = measure(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
