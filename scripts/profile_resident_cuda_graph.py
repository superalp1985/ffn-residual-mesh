from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf import GGUFReader
from resident_residual_cuda import DirectIQ4NLProjection, DirectQ4Projection
from resident_residual_format import ResidentArtifact
from resident_tiled_ffn import TiledResidentGateUp


def load_down(model: Path, layer: int):
    reader = GGUFReader(model)
    try:
        tensor = next(
            item for item in reader.tensors
            if item.name == f"blk.{layer}.ffn_down.weight"
        )
        quant = int(tensor.tensor_type)
        try:
            from gguf import GGMLQuantizationType
            quant_type = GGMLQuantizationType(quant)
        except (ImportError, ValueError):
            quant_type = None
        if quant_type is not None and quant_type.name == "Q4_K":
            return DirectQ4Projection(tensor.data, int(tensor.shape[0]))
        if quant_type is not None and quant_type.name == "IQ4_NL":
            return DirectIQ4NLProjection(tensor.data, int(tensor.shape[0]))
        raise ValueError(f"unsupported down tensor type: {quant_type or quant}")
    finally:
        reader.data._mmap.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one resident Qwen3.8 FFN layer")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--down", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    import torch

    down = None
    if args.down:
        down = load_down(args.model, args.layer)
    with ResidentArtifact.open(args.artifact, verify_hashes=False) as artifact:
        x = np.random.default_rng(6000 + args.layer).standard_normal(
            int(artifact.projections["gate"]["cols"])
        ).astype(np.float32)
        with TiledResidentGateUp(
            artifact,
            tile_rows=int(artifact.projections["gate"]["rows"]),
            persistent=True,
            base_on_gpu=True,
            use_cuda_graph=args.graph,
        ) as runner:
            for _ in range(args.warmup):
                runner.run(x, down=down, return_outputs=False)
            torch.cuda.synchronize()
            samples = [
                runner.run(x, down=down, return_outputs=False)
                for _ in range(args.repeats)
            ]
            report = {
                "layer": args.layer,
                "graph": args.graph,
                "down": args.down,
                "kernel_mode": samples[-1]["kernel_mode"],
                "wall_ms_median": float(np.median([item["wall_ms"] for item in samples])),
                "wall_ms_p95": float(np.percentile([item["wall_ms"] for item in samples], 95)),
                "graph_replay_ms_median": float(np.median([
                    item.get("cuda_graph_replay_ms", 0.0) for item in samples
                ])),
                "tile_kernel_ms_median": float(np.median([
                    item["tile_kernel_ms"] for item in samples
                ])),
                "base_h2d_bytes": samples[-1]["base_h2d_bytes"],
                "base_resident_bytes": samples[-1]["base_resident_bytes"],
            }
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
