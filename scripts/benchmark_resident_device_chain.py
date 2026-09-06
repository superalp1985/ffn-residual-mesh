from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from gguf import GGUFReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resident_residual_cuda import DirectIQ4NLProjection  # noqa: E402
from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_tiled_ffn import TiledResidentGateUp  # noqa: E402


def make_down(model: Path, layer: int) -> DirectIQ4NLProjection:
    reader = GGUFReader(model)
    try:
        tensor = next(
            item for item in reader.tensors
            if item.name == f"blk.{layer}.ffn_down.weight"
        )
        return DirectIQ4NLProjection(tensor.data, int(tensor.shape[0]))
    finally:
        reader.data._mmap.close()


def benchmark(
    first_artifact: Path,
    first_layer: int,
    second_artifact: Path,
    second_layer: int,
    model: Path,
    *,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if warmup < 1 or repeats < 3:
        raise ValueError("warmup must be positive and repeats must be at least 3")
    with ResidentArtifact.open(first_artifact, verify_hashes=False) as first_meta, \
            ResidentArtifact.open(second_artifact, verify_hashes=False) as second_meta:
        if first_meta.projections["gate"]["cols"] != second_meta.projections["gate"]["cols"]:
            raise ValueError("device chain requires matching hidden widths")
        first = TiledResidentGateUp(
            first_meta,
            tile_rows=int(first_meta.projections["gate"]["rows"]),
            persistent=True,
            base_on_gpu=True,
        )
        second = TiledResidentGateUp(
            second_meta,
            tile_rows=int(second_meta.projections["gate"]["rows"]),
            persistent=True,
            base_on_gpu=True,
        )
        down_first = make_down(model, first_layer)
        down_second = make_down(model, second_layer)
        try:
            rng = np.random.default_rng(seed)
            x = torch.from_numpy(
                rng.standard_normal(first.cols).astype(np.float32)
            ).cuda()
            stream = torch.cuda.Stream()

            for _ in range(warmup):
                first.run_device(
                    x,
                    down=down_first,
                    return_outputs=False,
                    measure_events=False,
                )
                second.run_device(
                    down_first.output,
                    down=down_second,
                    return_outputs=False,
                    measure_events=False,
                )

            host_sync_samples: list[float] = []
            host_sync_gpu_samples: list[float] = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                with torch.cuda.stream(torch.cuda.current_stream()):
                    start.record()
                first.run_device(
                    x,
                    down=down_first,
                    return_outputs=False,
                    measure_events=False,
                )
                second.run_device(
                    down_first.output,
                    down=down_second,
                    return_outputs=False,
                    measure_events=False,
                )
                end.record()
                end.synchronize()
                host_sync_samples.append((time.perf_counter() - begin) * 1000)
                host_sync_gpu_samples.append(float(start.elapsed_time(end)))

            tail_sync_gpu_samples: list[float] = []
            tail_sync_wall_samples: list[float] = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                with torch.cuda.stream(stream):
                    start.record()
                    first.run_device(
                        x,
                        down=down_first,
                        stream=stream,
                        return_outputs=False,
                        synchronize=False,
                        measure_events=False,
                    )
                    second.run_device(
                        down_first.output,
                        down=down_second,
                        stream=stream,
                        return_outputs=False,
                        synchronize=False,
                        measure_events=False,
                    )
                    end.record()
                end.synchronize()
                tail_sync_wall_samples.append((time.perf_counter() - begin) * 1000)
                tail_sync_gpu_samples.append(float(start.elapsed_time(end)))

            async_gpu_samples: list[float] = []
            async_enqueue_samples: list[float] = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                with torch.cuda.stream(stream):
                    start.record()
                    first.run_device(
                        x,
                        down=down_first,
                        stream=stream,
                        return_outputs=False,
                        synchronize=False,
                        measure_events=False,
                    )
                    second.run_device(
                        down_first.output,
                        down=down_second,
                        stream=stream,
                        return_outputs=False,
                        synchronize=False,
                        measure_events=False,
                    )
                    end.record()
                async_enqueue_samples.append((time.perf_counter() - begin) * 1000)
                end.synchronize()
                async_gpu_samples.append(float(start.elapsed_time(end)))

            return {
                "status": "measured_two_layer_gpu_resident_device_chain",
                "layers": [first_layer, second_layer],
                "warmup": warmup,
                "repeats": repeats,
                "host_sync_chain_wall_ms_median": float(np.median(host_sync_samples)),
                "host_sync_chain_wall_ms_p95": float(np.percentile(host_sync_samples, 95)),
                "host_sync_chain_gpu_ms_median": float(np.median(host_sync_gpu_samples)),
                "host_sync_chain_gpu_ms_p95": float(np.percentile(host_sync_gpu_samples, 95)),
                "tail_sync_chain_wall_ms_median": float(np.median(tail_sync_wall_samples)),
                "tail_sync_chain_wall_ms_p95": float(np.percentile(tail_sync_wall_samples, 95)),
                "tail_sync_chain_gpu_ms_median": float(np.median(tail_sync_gpu_samples)),
                "tail_sync_chain_gpu_ms_p95": float(np.percentile(tail_sync_gpu_samples, 95)),
                "async_chain_gpu_ms_median": float(np.median(async_gpu_samples)),
                "async_chain_gpu_ms_p95": float(np.percentile(async_gpu_samples, 95)),
                "async_cpu_enqueue_ms_median": float(np.median(async_enqueue_samples)),
                "activation_h2d_bytes_per_layer": [0, 0],
                "activation_d2d_bytes_per_layer": [0, 0],
                "base_h2d_bytes_per_layer": [0, 0],
                "resident_weight_h2d_bytes_per_layer": [0, 0],
                "first_kernel_mode": "device_activation_fused_base_residual_swiglu",
                "second_kernel_mode": "device_activation_fused_base_residual_swiglu",
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "note": (
                    "Host-sync, tail-sync and explicit async modes are measured "
                    "separately. All device paths use zero H2D activation/base "
                    "traffic; this is not end-to-end generation."
                ),
            }
        finally:
            first.close()
            second.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a GPU-resident FFN device-activation chain"
    )
    parser.add_argument("--first-artifact", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, required=True)
    parser.add_argument("--second-artifact", type=Path, required=True)
    parser.add_argument("--second-layer", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.first_artifact,
        args.first_layer,
        args.second_artifact,
        args.second_layer,
        args.model,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
