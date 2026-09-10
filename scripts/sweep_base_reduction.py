from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resident_residual_cuda import launch_fused_gate_up_base_residual  # noqa: E402
from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_tiled_ffn import TiledResidentGateUp  # noqa: E402
from sweep_mixed_down_kernel import build_projection  # noqa: E402


def _load_down(model: Path, layer: int):
    reader = GGUFReader(model)
    try:
        tensor = next(
            item for item in reader.tensors
            if item.name == f"blk.{layer}.ffn_down.weight"
        )
        raw = np.array(tensor.data, copy=True)
        tensor_type = int(tensor.tensor_type)
        shape = tuple(tensor.shape)
    finally:
        reader.data._mmap.close()
    projection, metadata = build_projection(
        SimpleNamespace(data=raw, tensor_type=tensor_type, shape=shape),
        block_rows=2,
        num_warps=2,
        block_qblocks=4,
    )
    return projection, metadata, raw, tensor_type


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    repeats: int,
) -> list[float]:
    values: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            begin.record()
            graph.replay()
            end.record()
        end.synchronize()
        values.append(float(begin.elapsed_time(end)))
    return values


def _capture_kernel_graph(
    runner: TiledResidentGateUp,
    x: torch.Tensor,
    groups: int,
    *,
    down=None,
) -> tuple[torch.cuda.CUDAGraph, float]:
    package = runner.cache.package(0)
    assert package is not None and runner.device_group_sums is not None
    with torch.cuda.stream(runner.stream):
        torch.sum(x.view(-1, 32), dim=1, dtype=torch.float32, out=runner.device_group_sums)
        launch_fused_gate_up_base_residual(
            package["gate.residual"], package["gate.alpha"],
            package["up.residual"], package["up.alpha"],
            runner.base_resident["gate"], runner.base_resident["up"],
            runner.device_group_sums, x,
            runner.output["gate"], runner.output["up"], runner.output["swiglu"],
            rows=runner.rows, cols=runner.cols, block_rows=runner.block_rows,
            num_warps=runner.num_warps, block_groups=groups,
        )
        if down is not None:
            down.launch(runner.output["swiglu"])
    runner.stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = time.perf_counter()
    with torch.cuda.graph(graph, stream=runner.stream):
        torch.sum(x.view(-1, 32), dim=1, dtype=torch.float32, out=runner.device_group_sums)
        launch_fused_gate_up_base_residual(
            package["gate.residual"], package["gate.alpha"],
            package["up.residual"], package["up.alpha"],
            runner.base_resident["gate"], runner.base_resident["up"],
            runner.device_group_sums, x,
            runner.output["gate"], runner.output["up"], runner.output["swiglu"],
            rows=runner.rows, cols=runner.cols, block_rows=runner.block_rows,
            num_warps=runner.num_warps, block_groups=groups,
        )
        if down is not None:
            down.launch(runner.output["swiglu"])
    runner.stream.synchronize()
    return graph, (time.perf_counter() - start) * 1000


def benchmark(
    artifact_path: Path,
    *,
    candidates: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    repeats: int = 31,
    inner: int = 3,
    warmup_seconds: float = 2.0,
    seed: int = 6802,
) -> dict[str, object]:
    allowed = {8, 16, 32, 64, 128, 256}
    if repeats < 1 or inner < 1:
        raise ValueError("repeats and inner must be positive")
    if not np.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("warmup_seconds must be finite and nonnegative")
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(group not in allowed for group in candidates)
        or 256 not in candidates
    ):
        raise ValueError(
            "candidates must be unique values from 8,16,32,64,128,256 and include 256"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    with ResidentArtifact.open(artifact_path, verify_hashes=True) as artifact:
        source = Path(artifact.manifest["source"]["path"])
        layer = int(artifact.manifest["layer"])
        down, down_metadata, down_raw, down_tensor_type = _load_down(source, layer)
        reference_reader = GGUFReader(source)
        try:
            tensors = {item.name: item for item in reference_reader.tensors}
            rng = np.random.default_rng(seed)
            x_host = rng.standard_normal(int(artifact.projections["gate"]["cols"])).astype(np.float32)
            gate = dequantize(
                tensors[f"blk.{layer}.ffn_gate.weight"].data,
                tensors[f"blk.{layer}.ffn_gate.weight"].tensor_type,
            ).astype(np.float64) @ x_host
            up = dequantize(
                tensors[f"blk.{layer}.ffn_up.weight"].data,
                tensors[f"blk.{layer}.ffn_up.weight"].tensor_type,
            ).astype(np.float64) @ x_host
            swiglu = gate * np.exp(-np.logaddexp(0, -gate)) * up
            reference = dequantize(down_raw, down_tensor_type).astype(np.float64) @ swiglu
        finally:
            reference_reader.data._mmap.close()

        x = torch.from_numpy(x_host).cuda()
        execution_orders = []
        variants = []
        total_warmup_elapsed = 0.0
        total_warmup_cycles = 0
        resident_payload_bytes = 0
        for pass_index in range(3):
            order = list(candidates)
            rng.shuffle(order)
            execution_orders.append(order)
            for groups in order:
                with TiledResidentGateUp(
                    artifact, tile_rows=int(artifact.projections["gate"]["rows"]),
                    persistent=True, base_on_gpu=True, base_block_groups=groups,
                    auto_tune_q5_base=False,
                ) as runner:
                    kernel_graph, capture_ms = _capture_kernel_graph(runner, x, groups)
                    full_graph, full_capture_ms = _capture_kernel_graph(
                        runner, x, groups, down=down
                    )
                    warmup_begin = time.perf_counter()
                    warmup_cycles = 0
                    while time.perf_counter() - warmup_begin < warmup_seconds:
                        with torch.cuda.stream(runner.stream):
                            full_graph.replay()
                        runner.stream.synchronize()
                        warmup_cycles += 1
                    warmup_elapsed = time.perf_counter() - warmup_begin
                    total_warmup_elapsed += warmup_elapsed
                    total_warmup_cycles += warmup_cycles
                    samples = []
                    for round_index in range(repeats):
                        ffn_raw = _time_graph(full_graph, runner.stream, inner)
                        kernel_raw = _time_graph(kernel_graph, runner.stream, inner)
                        samples.append({
                            "pass_index": pass_index,
                            "round_index": round_index,
                            "kernel_graph_ms": float(np.mean(kernel_raw)),
                            "ffn_graph_ms": float(np.mean(ffn_raw)),
                            "kernel_graph_raw_ms": kernel_raw,
                            "ffn_graph_raw_ms": ffn_raw,
                        })
                    actual = down.output.detach().cpu().numpy().astype(np.float64)
                    delta = actual - reference
                    package = runner.cache.package(0)
                    compiled = launch_fused_gate_up_base_residual(
                        package["gate.residual"],
                        package["gate.alpha"],
                        package["up.residual"],
                        package["up.alpha"],
                        runner.base_resident["gate"],
                        runner.base_resident["up"],
                        runner.device_group_sums,
                        x,
                        runner.output["gate"],
                        runner.output["up"],
                        runner.output["swiglu"],
                        rows=runner.rows,
                        cols=runner.cols,
                        block_rows=runner.block_rows,
                        num_warps=runner.num_warps,
                        block_groups=groups,
                    )
                    runner.stream.synchronize()
                    resident_payload_bytes = (
                        runner.cache.cache.device_bytes()
                        + sum(
                            value.numel() * value.element_size()
                            for value in runner.base_resident.values()
                        )
                        + down.raw.numel() * down.raw.element_size()
                    )
                    variants.append({
                        "base_block_groups": groups,
                        "registers_per_thread": int(compiled.n_regs),
                        "spill_count": int(compiled.n_spills),
                        "shared_bytes": int(compiled.metadata.shared),
                        "kernel_capture_ms": capture_ms,
                        "ffn_capture_ms": full_capture_ms,
                        "kernel_graph_ms_median": float(np.median(
                            [sample["kernel_graph_ms"] for sample in samples]
                        )),
                        "ffn_graph_ms_median": float(np.median(
                            [sample["ffn_graph_ms"] for sample in samples]
                        )),
                        "samples": samples,
                        "output_rel_l2": float(
                            np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-20)
                        ),
                        "output_max_abs": float(np.max(np.abs(delta))),
                        "warmup_elapsed_seconds": warmup_elapsed,
                        "warmup_cycles": warmup_cycles,
                        "pass_index": pass_index,
                    })
                    runner.cache.release(0)
        by_groups = {}
        for variant in variants:
            by_groups.setdefault(variant["base_block_groups"], []).append(variant)
        compact = []
        for groups in candidates:
            entries = by_groups[groups]
            samples = [
                sample
                for entry in entries
                for sample in entry["samples"]
            ]
            compact.append({
                "base_block_groups": groups,
                "registers_per_thread": entries[-1]["registers_per_thread"],
                "spill_count": entries[-1]["spill_count"],
                "shared_bytes": entries[-1]["shared_bytes"],
                "kernel_graph_ms_median": float(np.median(
                    [sample["kernel_graph_ms"] for sample in samples]
                )),
                "ffn_graph_ms_median": float(np.median(
                    [sample["ffn_graph_ms"] for sample in samples]
                )),
                "output_rel_l2": max(e["output_rel_l2"] for e in entries),
                "output_max_abs": max(e["output_max_abs"] for e in entries),
                "passes": entries,
                "samples": samples,
            })
        return {
            "status": "measured_base_reduction_sweep",
            "layer": layer,
            "dimensions": {
                "hidden": int(artifact.projections["gate"]["cols"]),
                "ffn": int(artifact.projections["gate"]["rows"]),
            },
            "baseline_block_groups": 256,
            "candidates": list(candidates),
            "execution_order": execution_orders,
            "variants": compact,
            "warmup_seconds_per_variant_pass": warmup_seconds,
            "warmup_elapsed_seconds": total_warmup_elapsed,
            "warmup_cycles": total_warmup_cycles,
            "down_projection": down_metadata,
            "resident_payload_bytes": resident_payload_bytes,
            "dynamic_h2d_bytes_per_graph_ffn": 0,
            "tokens_per_second": None,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "notes": [
                "CUDA graph spans isolate repeated device work; they are not SM occupancy.",
                "The complete graph contains the synthetic down projection.",
                "Graphs consume a fixed CUDA activation, so dynamic H2D is zero by construction.",
                "This is a single synthetic activation and not generation throughput.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--candidates", default="8,16,32,64,128,256")
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(
        args.artifact,
        candidates=tuple(int(x) for x in args.candidates.split(",") if x),
        repeats=args.repeats,
        inner=args.inner,
        warmup_seconds=args.warmup_seconds,
    )
    text = json.dumps(report, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
