from __future__ import annotations

"""Compare GPU and CPU placement for the post-residual FFN merge.

The CPU path deliberately includes the required device round trip:

    GPU residual -> D2H -> CPU add -> H2D merged gate/up -> GPU SwiGLU/down

The GPU path keeps base and residual vectors on device and merges them before
SwiGLU.  This benchmark is about the complete critical path, not isolated
``numpy.add`` or kernel timings.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resident_residual_cuda import (  # noqa: E402
    launch_fused_gate_up_residual_tile,
    launch_merge_swiglu,
)
from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_tiled_ffn import TiledResidentGateUp  # noqa: E402
from benchmark_resident_device_chain import make_down  # noqa: E402


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _base_from_host(
    artifact: ResidentArtifact,
    activation: np.ndarray,
    projection: str,
) -> np.ndarray:
    grouped = activation.reshape(-1, 32).sum(axis=1, dtype=np.float64)
    coefficient = np.asarray(artifact.arrays[projection]["coefficient"])
    return np.asarray(coefficient @ grouped, dtype=np.float32)


def _prepare(
    artifact_path: Path,
    model: Path,
    layer: int,
    seed: int,
) -> tuple[
    ResidentArtifact,
    TiledResidentGateUp,
    object,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    artifact = ResidentArtifact.open(artifact_path, verify_hashes=False)
    runner = TiledResidentGateUp(
        artifact,
        tile_rows=int(artifact.projections["gate"]["rows"]),
        persistent=True,
        base_on_gpu=True,
    )
    if len(runner.plan.tile_slices()) != 1:
        runner.close()
        artifact.close()
        raise ValueError("merge benchmark requires a one-tile resident artifact")
    down, _ = make_down(model, layer)
    rng = np.random.default_rng(seed)
    activation_np = rng.standard_normal(runner.cols).astype(np.float32)
    activation = torch.from_numpy(activation_np).cuda()

    rows = runner.rows
    base_gate_np = _base_from_host(artifact, activation_np, "gate")
    base_up_np = _base_from_host(artifact, activation_np, "up")
    host_base_gate = torch.from_numpy(base_gate_np).pin_memory()
    host_base_up = torch.from_numpy(base_up_np).pin_memory()
    base_gate = torch.from_numpy(base_gate_np).cuda()
    base_up = torch.from_numpy(base_up_np).cuda()
    residual_gate = torch.empty(rows, dtype=torch.float32, device="cuda")
    residual_up = torch.empty(rows, dtype=torch.float32, device="cuda")
    gate_output = torch.empty_like(residual_gate)
    up_output = torch.empty_like(residual_up)
    swiglu_output = torch.empty_like(residual_gate)

    runner.cache.acquire(0)
    package = runner.cache.package(0)
    setup_stream = torch.cuda.Stream()
    with torch.cuda.stream(setup_stream):
        launch_fused_gate_up_residual_tile(
            package["gate.residual"],
            package["gate.alpha"],
            package["up.residual"],
            package["up.alpha"],
            activation,
            residual_gate,
            residual_up,
            rows=rows,
            cols=runner.cols,
            block_rows=runner.block_rows,
            num_warps=runner.num_warps,
            gate_bits=runner.bits["gate"],
            up_bits=runner.bits["up"],
            block_groups=runner.residual_block_groups,
        )
    setup_stream.synchronize()
    runner.cache.release(0)
    return (
        artifact,
        runner,
        down,
        activation,
        base_gate,
        base_up,
        residual_gate,
        residual_up,
        swiglu_output,
        host_base_gate,
        host_base_up,
    )


def _run_gpu_merge(
    *,
    base_gate: torch.Tensor,
    base_up: torch.Tensor,
    residual_gate: torch.Tensor,
    residual_up: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    swiglu_output: torch.Tensor,
    down: object,
    rows: int,
) -> dict[str, float]:
    stream = torch.cuda.Stream()
    begin = torch.cuda.Event(enable_timing=True)
    merge_end = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_begin = time.perf_counter()
    with torch.cuda.stream(stream):
        begin.record()
        launch_merge_swiglu(
            residual_gate,
            residual_up,
            base_gate,
            base_up,
            gate_output,
            up_output,
            swiglu_output,
            rows=rows,
        )
        merge_end.record()
        down.launch(swiglu_output)
        end.record()
    end.synchronize()
    return {
        "wall_ms": (time.perf_counter() - host_begin) * 1000.0,
        "gpu_merge_ms": float(begin.elapsed_time(merge_end)),
        "gpu_swiglu_down_ms": float(merge_end.elapsed_time(end)),
        "critical_ms": float(begin.elapsed_time(end)),
        "d2h_bytes": 0.0,
        "h2d_bytes": 0.0,
        "cpu_merge_ms": 0.0,
    }


def _run_cpu_merge(
    *,
    base_gate: torch.Tensor,
    base_up: torch.Tensor,
    host_base_gate: torch.Tensor,
    host_base_up: torch.Tensor,
    residual_gate: torch.Tensor,
    residual_up: torch.Tensor,
    down: object,
    rows: int,
    host_gate_residual: torch.Tensor,
    host_up_residual: torch.Tensor,
    host_gate_merged: torch.Tensor,
    host_up_merged: torch.Tensor,
    device_gate_merged: torch.Tensor,
    device_up_merged: torch.Tensor,
    zero_gate: torch.Tensor,
    zero_up: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    swiglu_output: torch.Tensor,
) -> dict[str, float]:
    stream = torch.cuda.Stream()
    begin = torch.cuda.Event(enable_timing=True)
    d2h_begin = torch.cuda.Event(enable_timing=True)
    d2h_end = torch.cuda.Event(enable_timing=True)
    h2d_begin = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    gpu_begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_begin = time.perf_counter()
    with torch.cuda.stream(stream):
        begin.record()
        d2h_begin.record()
        host_gate_residual.copy_(residual_gate, non_blocking=True)
        host_up_residual.copy_(residual_up, non_blocking=True)
        d2h_end.record()
    # CPU cannot merge until both D2H copies have completed.
    d2h_end.synchronize()
    cpu_begin = time.perf_counter()
    np.add(
        host_gate_residual.numpy(),
        host_base_gate.numpy(),
        out=host_gate_merged.numpy(),
    )
    np.add(
        host_up_residual.numpy(),
        host_base_up.numpy(),
        out=host_up_merged.numpy(),
    )
    cpu_merge_ms = (time.perf_counter() - cpu_begin) * 1000.0
    with torch.cuda.stream(stream):
        h2d_begin.record()
        device_gate_merged.copy_(host_gate_merged, non_blocking=True)
        device_up_merged.copy_(host_up_merged, non_blocking=True)
        h2d_end.record()
        gpu_begin.record()
        # Feed already merged vectors through the same SwiGLU kernel.  The
        # zero residual tensors avoid changing the nonlinear expression.
        launch_merge_swiglu(
            zero_gate,
            zero_up,
            device_gate_merged,
            device_up_merged,
            gate_output,
            up_output,
            swiglu_output,
            rows=rows,
        )
        down.launch(swiglu_output)
        end.record()
    end.synchronize()
    return {
        "wall_ms": (time.perf_counter() - host_begin) * 1000.0,
        "gpu_merge_ms": 0.0,
        "gpu_swiglu_down_ms": float(gpu_begin.elapsed_time(end)),
        "critical_ms": float(begin.elapsed_time(end)),
        "d2h_ms": float(d2h_begin.elapsed_time(d2h_end)),
        "h2d_ms": float(h2d_begin.elapsed_time(h2d_end)),
        "d2h_bytes": float(2 * rows * 4),
        "h2d_bytes": float(2 * rows * 4),
        "cpu_merge_ms": cpu_merge_ms,
    }


def benchmark_one(
    artifact_path: Path,
    model: Path,
    layer: int,
    *,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    (
        artifact,
        runner,
        down,
        activation,
        base_gate,
        base_up,
        residual_gate,
        residual_up,
        swiglu_output,
        host_base_gate,
        host_base_up,
    ) = _prepare(artifact_path, model, layer, seed)
    rows = runner.rows
    host_gate_residual = torch.empty(rows, dtype=torch.float32, pin_memory=True)
    host_up_residual = torch.empty(rows, dtype=torch.float32, pin_memory=True)
    host_gate_merged = torch.empty(rows, dtype=torch.float32, pin_memory=True)
    host_up_merged = torch.empty(rows, dtype=torch.float32, pin_memory=True)
    device_gate_merged = torch.empty_like(base_gate)
    device_up_merged = torch.empty_like(base_up)
    zero_gate = torch.zeros_like(base_gate)
    zero_up = torch.zeros_like(base_up)
    gate_output = torch.empty_like(base_gate)
    up_output = torch.empty_like(base_up)

    try:
        # Warm up both merge paths and the down projection.
        for _ in range(warmup):
            _run_gpu_merge(
                base_gate=base_gate,
                base_up=base_up,
                residual_gate=residual_gate,
                residual_up=residual_up,
                gate_output=gate_output,
                up_output=up_output,
                swiglu_output=swiglu_output,
                down=down,
                rows=rows,
            )
            _run_cpu_merge(
                base_gate=base_gate,
                base_up=base_up,
                host_base_gate=host_base_gate,
                host_base_up=host_base_up,
                residual_gate=residual_gate,
                residual_up=residual_up,
                down=down,
                rows=rows,
                host_gate_residual=host_gate_residual,
                host_up_residual=host_up_residual,
                host_gate_merged=host_gate_merged,
                host_up_merged=host_up_merged,
                device_gate_merged=device_gate_merged,
                device_up_merged=device_up_merged,
                zero_gate=zero_gate,
                zero_up=zero_up,
                gate_output=gate_output,
                up_output=up_output,
                swiglu_output=swiglu_output,
            )

        gpu_samples = [
            _run_gpu_merge(
                base_gate=base_gate,
                base_up=base_up,
                residual_gate=residual_gate,
                residual_up=residual_up,
                gate_output=gate_output,
                up_output=up_output,
                swiglu_output=swiglu_output,
                down=down,
                rows=rows,
            )
            for _ in range(repeats)
        ]
        cpu_samples = [
            _run_cpu_merge(
                base_gate=base_gate,
                base_up=base_up,
                host_base_gate=host_base_gate,
                host_base_up=host_base_up,
                residual_gate=residual_gate,
                residual_up=residual_up,
                down=down,
                rows=rows,
                host_gate_residual=host_gate_residual,
                host_up_residual=host_up_residual,
                host_gate_merged=host_gate_merged,
                host_up_merged=host_up_merged,
                device_gate_merged=device_gate_merged,
                device_up_merged=device_up_merged,
                zero_gate=zero_gate,
                zero_up=zero_up,
                gate_output=gate_output,
                up_output=up_output,
                swiglu_output=swiglu_output,
            )
            for _ in range(repeats)
        ]

        # Validate one fresh sample from each path against the other.
        _run_gpu_merge(
            base_gate=base_gate,
            base_up=base_up,
            residual_gate=residual_gate,
            residual_up=residual_up,
            gate_output=gate_output,
            up_output=up_output,
            swiglu_output=swiglu_output,
            down=down,
            rows=rows,
        )
        gpu_gate = gate_output.detach().cpu().numpy().copy()
        gpu_up = up_output.detach().cpu().numpy().copy()
        gpu_swiglu = swiglu_output.detach().cpu().numpy().copy()
        gpu_down = down.output.detach().cpu().numpy().copy()
        _run_cpu_merge(
            base_gate=base_gate,
            base_up=base_up,
            host_base_gate=host_base_gate,
            host_base_up=host_base_up,
            residual_gate=residual_gate,
            residual_up=residual_up,
            down=down,
            rows=rows,
            host_gate_residual=host_gate_residual,
            host_up_residual=host_up_residual,
            host_gate_merged=host_gate_merged,
            host_up_merged=host_up_merged,
            device_gate_merged=device_gate_merged,
            device_up_merged=device_up_merged,
            zero_gate=zero_gate,
            zero_up=zero_up,
            gate_output=gate_output,
            up_output=up_output,
            swiglu_output=swiglu_output,
        )
        cpu_gate = gate_output.detach().cpu().numpy().copy()
        cpu_up = up_output.detach().cpu().numpy().copy()
        cpu_swiglu = swiglu_output.detach().cpu().numpy().copy()
        cpu_down = down.output.detach().cpu().numpy().copy()
        error = {
            "gate_rel_l2": float(
                np.linalg.norm(gpu_gate - cpu_gate)
                / max(np.linalg.norm(gpu_gate), 1e-12)
            ),
            "up_rel_l2": float(
                np.linalg.norm(gpu_up - cpu_up)
                / max(np.linalg.norm(gpu_up), 1e-12)
            ),
            "swiglu_rel_l2": float(
                np.linalg.norm(gpu_swiglu - cpu_swiglu)
                / max(np.linalg.norm(gpu_swiglu), 1e-12)
            ),
            "down_rel_l2": float(
                np.linalg.norm(gpu_down - cpu_down)
                / max(np.linalg.norm(gpu_down), 1e-12)
            ),
        }

        def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
            keys = samples[0].keys()
            return {
                f"{key}_median": _median([float(sample[key]) for sample in samples])
                for key in keys
            } | {
                "critical_ms_p95": _percentile(
                    [float(sample["critical_ms"]) for sample in samples], 95
                )
            }

        return {
            "layer": layer,
            "artifact": str(artifact_path),
            "rows": rows,
            "cols": runner.cols,
            "warmup": warmup,
            "repeats": repeats,
            "gpu_merge": summarize(gpu_samples),
            "cpu_merge_roundtrip": summarize(cpu_samples),
            "speedup_gpu_over_cpu_critical": (
                _median([s["critical_ms"] for s in cpu_samples])
                / max(_median([s["critical_ms"] for s in gpu_samples]), 1e-12)
            ),
            "output_error": error,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "note": (
                "GPU path keeps base and residual on CUDA. CPU path includes "
                "2*rows*fp32 D2H, CPU gate/up add, 2*rows*fp32 H2D, then "
                "SwiGLU/down. Residual production is held constant and "
                "measured separately from merge placement."
            ),
        }
    finally:
        runner.close()
        artifact.close()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU versus GPU FFN merge placement")
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if len(args.artifact) != len(args.layer):
        parser.error("--artifact and --layer must be repeated equally")
    reports = [
        benchmark_one(
            artifact,
            args.model,
            layer,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed + index,
        )
        for index, (artifact, layer) in enumerate(zip(args.artifact, args.layer))
    ]
    report = {
        "status": "measured_ffn_merge_location",
        "reports": reports,
        "scope": (
            "Post-residual merge placement only. This is not a full model "
            "generation throughput result."
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
