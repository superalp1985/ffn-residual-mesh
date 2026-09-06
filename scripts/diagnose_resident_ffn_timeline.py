from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf import GGUFReader
from resident_residual_cuda import (
    DirectIQ4NLProjection,
    launch_fused_gate_up_base_residual,
)
from resident_residual_format import ResidentArtifact
from resident_tiled_ffn import TiledResidentGateUp


def gpu_snapshot() -> dict[str, str]:
    query = (
        "pstate,clocks.sm,clocks.mem,power.draw,utilization.gpu,"
        "utilization.memory"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = [part.strip() for part in result.stdout.strip().split(",")]
        names = query.split(",")
        return dict(zip(names, fields))
    except (OSError, subprocess.SubprocessError):
        return {}


def elapsed(begin: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(begin.elapsed_time(end))


def timed_op(stream: torch.cuda.Stream, operation) -> tuple[float, float]:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_begin = time.perf_counter()
    with torch.cuda.stream(stream):
        begin.record()
        operation()
        end.record()
    end.synchronize()
    return (time.perf_counter() - host_begin) * 1000.0, elapsed(begin, end)


def load_down(model: Path, layer: int) -> DirectIQ4NLProjection:
    reader = GGUFReader(model)
    try:
        tensor = next(
            item for item in reader.tensors
            if item.name == f"blk.{layer}.ffn_down.weight"
        )
        return DirectIQ4NLProjection(tensor.data, int(tensor.shape[0]))
    finally:
        reader.data._mmap.close()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    with ResidentArtifact.open(args.artifact, verify_hashes=False) as artifact:
        down = load_down(args.model, args.layer) if args.down else None
        runner = TiledResidentGateUp(
            artifact,
            tile_rows=int(artifact.projections["gate"]["rows"]),
            persistent=True,
            base_on_gpu=True,
        )
        try:
            x = torch.randn(
                runner.cols, device=runner.device, dtype=torch.float32
            )
            stream = torch.cuda.current_stream(runner.device)
            group_sums = torch.empty(
                runner.cols // 32, device=runner.device, dtype=torch.float32
            )
            package = runner.cache.package(0)
            assert package is not None

            # Keep the device hot long enough for laptop boost clocks to settle.
            for _ in range(args.boost_iters):
                with torch.cuda.stream(stream):
                    torch.sum(x.view(-1, 32), dim=1, out=group_sums)
                    launch_fused_gate_up_base_residual(
                        package["gate.residual"],
                        package["gate.alpha"],
                        package["up.residual"],
                        package["up.alpha"],
                        runner.base_resident["gate"],
                        runner.base_resident["up"],
                        group_sums,
                        x,
                        runner.output["gate"],
                        runner.output["up"],
                        runner.output["swiglu"],
                        rows=runner.rows,
                        cols=runner.cols,
                        block_rows=runner.block_rows,
                        num_warps=runner.num_warps,
                    )
            torch.cuda.synchronize()
            before = gpu_snapshot()

            stages: dict[str, dict[str, float]] = {}
            for _ in range(args.warmup):
                timed_op(
                    stream,
                    lambda: torch.sum(
                        x.view(-1, 32), dim=1, out=group_sums
                    ),
                )
                timed_op(
                    stream,
                    lambda: launch_fused_gate_up_base_residual(
                        package["gate.residual"],
                        package["gate.alpha"],
                        package["up.residual"],
                        package["up.alpha"],
                        runner.base_resident["gate"],
                        runner.base_resident["up"],
                        group_sums,
                        x,
                        runner.output["gate"],
                        runner.output["up"],
                        runner.output["swiglu"],
                        rows=runner.rows,
                        cols=runner.cols,
                        block_rows=runner.block_rows,
                        num_warps=runner.num_warps,
                    ),
                )
                if down is not None:
                    timed_op(stream, lambda: down.launch(runner.output["swiglu"]))

            for stage, operation in (
                (
                    "group_sum",
                    lambda: torch.sum(
                        x.view(-1, 32), dim=1, out=group_sums
                    ),
                ),
                (
                    "fused_gate_up_base_residual_swiglu",
                    lambda: launch_fused_gate_up_base_residual(
                        package["gate.residual"],
                        package["gate.alpha"],
                        package["up.residual"],
                        package["up.alpha"],
                        runner.base_resident["gate"],
                        runner.base_resident["up"],
                        group_sums,
                        x,
                        runner.output["gate"],
                        runner.output["up"],
                        runner.output["swiglu"],
                        rows=runner.rows,
                        cols=runner.cols,
                        block_rows=runner.block_rows,
                        num_warps=runner.num_warps,
                    ),
                ),
            ):
                host_values: list[float] = []
                gpu_values: list[float] = []
                for _ in range(args.repeats):
                    host_ms, gpu_ms = timed_op(stream, operation)
                    host_values.append(host_ms)
                    gpu_values.append(gpu_ms)
                stages[stage] = {
                    "gpu_median_ms": float(np.median(gpu_values)),
                    "gpu_p95_ms": float(np.percentile(gpu_values, 95)),
                    "host_wall_median_ms": float(np.median(host_values)),
                    "host_wall_p95_ms": float(np.percentile(host_values, 95)),
                }

            if down is not None:
                host_values = []
                gpu_values = []
                for _ in range(args.repeats):
                    host_ms, gpu_ms = timed_op(
                        stream, lambda: down.launch(runner.output["swiglu"])
                    )
                    host_values.append(host_ms)
                    gpu_values.append(gpu_ms)
                stages["down_iq4_nl"] = {
                    "gpu_median_ms": float(np.median(gpu_values)),
                    "gpu_p95_ms": float(np.percentile(gpu_values, 95)),
                    "host_wall_median_ms": float(np.median(host_values)),
                    "host_wall_p95_ms": float(np.percentile(host_values, 95)),
                }

            # Mark each boundary on one stream to expose any launch gap. The
            # GPU event span excludes host-side Python enqueue time.
            chain_begin = torch.cuda.Event(enable_timing=True)
            group_end = torch.cuda.Event(enable_timing=True)
            fused_end = torch.cuda.Event(enable_timing=True)
            down_end = torch.cuda.Event(enable_timing=True) if down else None
            chain_host_begin = time.perf_counter()
            with torch.cuda.stream(stream):
                chain_begin.record()
                torch.sum(x.view(-1, 32), dim=1, out=group_sums)
                group_end.record()
                launch_fused_gate_up_base_residual(
                    package["gate.residual"],
                    package["gate.alpha"],
                    package["up.residual"],
                    package["up.alpha"],
                    runner.base_resident["gate"],
                    runner.base_resident["up"],
                    group_sums,
                    x,
                    runner.output["gate"],
                    runner.output["up"],
                    runner.output["swiglu"],
                    rows=runner.rows,
                    cols=runner.cols,
                    block_rows=runner.block_rows,
                    num_warps=runner.num_warps,
                )
                fused_end.record()
                if down is not None:
                    down.launch(runner.output["swiglu"])
                    assert down_end is not None
                    down_end.record()
            (down_end or fused_end).synchronize()
            chain_host_ms = (time.perf_counter() - chain_host_begin) * 1000.0
            span_ms = elapsed(chain_begin, down_end or fused_end)

            after = gpu_snapshot()
            return {
                "status": "resident_ffn_timeline",
                "layer": args.layer,
                "dimensions": {"rows": runner.rows, "cols": runner.cols},
                "block_rows": runner.block_rows,
                "num_warps": runner.num_warps,
                "down": args.down,
                "stages": stages,
                "chain": {
                    "group_sum_ms": elapsed(chain_begin, group_end),
                    "fused_ms": elapsed(group_end, fused_end),
                    "down_ms": (
                        elapsed(fused_end, down_end)
                        if down_end is not None else 0.0
                    ),
                    "gpu_span_ms": span_ms,
                    "host_enqueue_and_wait_ms": chain_host_ms,
                    "launch_gap_proxy_ms": max(
                        0.0,
                        span_ms - (
                            elapsed(chain_begin, group_end)
                            + elapsed(group_end, fused_end)
                            + (
                                elapsed(fused_end, down_end)
                                if down_end is not None else 0.0
                            )
                        ),
                    ),
                },
                "gpu_before": before,
                "gpu_after": after,
                "resident_bytes": int(
                    sum(value.numel() * value.element_size()
                        for value in runner.base_resident.values())
                    + sum(value.numel() * value.element_size()
                          for value in runner.cache.package(0).values()
                          if isinstance(value, torch.Tensor))
                ),
                "note": (
                    "GPU event timings measure device execution. This does not "
                    "claim Nsight occupancy or hardware DMA counters."
                ),
            }
        finally:
            runner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--down", action="store_true")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--boost-iters", type=int, default=40)
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
