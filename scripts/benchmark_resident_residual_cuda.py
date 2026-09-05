from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resident_residual_cuda import DirectQ4Projection, ResidentGateUp  # noqa: E402
from resident_residual_format import ResidentArtifact  # noqa: E402


def error(actual: np.ndarray, expected: np.ndarray) -> dict:
    delta = np.asarray(actual, dtype=np.float64) - expected
    return {"rel_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(expected)), 1e-20)),
            "max_abs": float(np.max(np.abs(delta)))}


def graph_ms(launch, stream, repeats=20) -> float:
    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    warm_until = time.perf_counter() + 0.5
    with torch.cuda.stream(stream):
        while time.perf_counter() < warm_until:
            for _ in range(repeats):
                graph.replay()
            stream.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        with torch.cuda.stream(stream):
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)


def benchmark(
    artifact_path: Path, *, repeats: int = 9, cpu_threads=(1, 2, 4, 8),
    launch_shapes=((1, 4), (2, 4), (4, 4), (8, 4), (4, 8)),
) -> dict:
    if repeats < 3 or not cpu_threads or not launch_shapes or any(n < 1 for n in cpu_threads):
        raise ValueError("at least 3 repeats and positive CPU thread counts are required")
    with ResidentArtifact.open(artifact_path, verify_hashes=True) as artifact:
        return _benchmark(artifact, repeats, cpu_threads, launch_shapes)


def _benchmark(artifact, repeats, cpu_threads, launch_shapes):
    source = artifact.manifest["source"]
    model = Path(source["path"])
    stat = model.stat()
    if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
        raise ValueError("source GGUF changed since artifact compilation")
    runner = ResidentGateUp(artifact)
    rng = np.random.default_rng(5152)
    inputs = rng.standard_normal((repeats + 1, runner.cols)).astype(np.float32)
    reader = GGUFReader(model)
    try:
        tensors = {item.name: item for item in reader.tensors}
        oracle = {}
        baseline = {}
        for p in ("gate", "up"):
            tensor = tensors[f"blk.{artifact.manifest['layer']}.ffn_{p}.weight"]
            baseline[p] = DirectQ4Projection(tensor.data, runner.cols)
            oracle[p] = np.empty(runner.rows, dtype=np.float64)
            with threadpool_limits(limits=4):
                for start in range(0, runner.rows, 128):
                    stop = min(start + 128, runner.rows)
                    weight = dequantize(tensor.data[start:stop], tensor.tensor_type)
                    np.testing.assert_array_equal(
                        artifact.reconstruct_weights(p, start, stop).view(np.uint32),
                        weight.view(np.uint32),
                    )
                    oracle[p][start:stop] = weight.astype(np.float64) @ inputs[0]
    finally:
        reader.data._mmap.close()
    with threadpool_limits(limits=4):
        runner.run(inputs[0], return_outputs=False)
        validation = runner.run(inputs[0])
    # Stable float64 sigmoid reference, including saturated real-model outputs.
    g = oracle["gate"]
    sigmoid = np.empty_like(g)
    positive = g >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-g[positive]))
    exp_g = np.exp(g[~positive])
    sigmoid[~positive] = exp_g / (1 + exp_g)
    oracle["swiglu"] = g * sigmoid * oracle["up"]
    correctness = {p: error(validation[p], oracle[p]) for p in ("gate", "up", "swiglu")}
    for p in ("gate", "up"):
        with torch.cuda.stream(runner.stream):
            baseline[p].launch(runner.device_x)
        runner.stream.synchronize()
        correctness[f"raw_q4_{p}"] = error(baseline[p].output.cpu().numpy(), oracle[p])
    if any(not np.isfinite(e["rel_l2"]) or e["rel_l2"] > 1e-4 for e in correctness.values()):
        raise ValueError(f"numerical gate failed: {correctness}")
    rows = []
    for block_rows, warps in launch_shapes:
        runner.block_rows, runner.num_warps = block_rows, warps
        with torch.cuda.stream(runner.stream):
            runner.launch_residuals()
        runner.stream.synchronize()
        residual_ms = graph_ms(runner.launch_residuals, runner.stream)
        for p in ("gate", "up"):
            baseline[p].block_rows, baseline[p].num_warps = block_rows, warps
        def direct():
            for name in ("gate", "up"):
                baseline[name].launch(runner.device_x)
        direct_ms = graph_ms(direct, runner.stream)
        for threads in cpu_threads:
            samples = []
            with threadpool_limits(limits=threads):
                for _ in range(2):
                    runner.run(inputs[1], return_outputs=False)
                for x in inputs[1:]:
                    samples.append(runner.run(x, return_outputs=False)["timing"])
            medians = {key: statistics.median(row[key] for row in samples) for key in samples[0]}
            rows.append({
                "block_rows": block_rows, "num_warps": warps, "cpu_threads": threads,
                "median": medians, "samples": samples,
                "residual_graph_ms": residual_ms, "raw_q4_graph_ms": direct_ms,
                "kernel_resources": runner.kernel_resources,
                "logical_resident_read_gbps": runner.resident_bytes / residual_ms / 1e6,
                "kernel_span_fraction_proxy": (
                    medians["residual_stream_span_ms"] + medians["merge_stream_span_ms"]
                ) / max(medians["stream_span_ms"], 1e-20),
            })
    best = min(rows, key=lambda row: row["median"]["wall_ms"])
    # A deliberately serialized offload comparison; not an optimized paging baseline.
    raw_host = {p: baseline[p].raw.cpu().pin_memory() for p in ("gate", "up")}
    for p in ("gate", "up"):
        baseline[p].block_rows, baseline[p].num_warps = best["block_rows"], best["num_warps"]
    copy_events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    streamed = []
    for trial in range(repeats + 2):
        with torch.cuda.stream(runner.stream):
            runner.device_x.copy_(runner.host_x, non_blocking=True)
            copy_events[0].record()
            for p in ("gate", "up"):
                baseline[p].raw.copy_(raw_host[p], non_blocking=True)
            copy_events[1].record()
            for p in ("gate", "up"):
                baseline[p].launch(runner.device_x)
            copy_events[2].record()
        copy_events[2].synchronize()
        if trial >= 2:
            streamed.append({
                "weight_h2d_ms": copy_events[0].elapsed_time(copy_events[1]),
                "compute_stream_ms": copy_events[1].elapsed_time(copy_events[2]),
                "total_stream_ms": copy_events[0].elapsed_time(copy_events[2]),
            })
    torch.cuda.synchronize()
    return {
        "status": "measured_gate_up_swiglu_only", "timestamp": datetime.now().astimezone().isoformat(),
        "artifact": str(artifact.directory), "layer": artifact.manifest["layer"],
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "dimensions": {"rows": runner.rows, "cols": runner.cols},
        "input": "fixed-seed synthetic FP32 vectors; NOT captured model activations or quality evaluation",
        "exactness": "all gate/up reconstructed FP32 weight bit patterns checked against gguf decoder",
        "correctness": correctness, "resident_weight_bytes": runner.resident_bytes,
        "cpu_base_coefficient_bytes": artifact.manifest["byte_ledger"]["host_base_bytes"],
        "raw_q4_gate_up_bytes": artifact.manifest["byte_ledger"]["source_gate_up_bytes"],
        "dynamic_h2d_bytes_per_run": 4 * (runner.cols + runner.rows * 2),
        "residual_weight_h2d_bytes_per_run": 0,
        "traffic_counters_including_warmups": runner.traffic,
        "gpu_allocated_bytes_including_baseline": torch.cuda.memory_allocated(),
        "configurations": rows, "best_configuration": {k: v for k, v in best.items() if k != "samples"},
        "streamed_raw_q4_baseline": {
            "scope": "serialized pinned Q4_K weight H2D + gate/up only; excludes SwiGLU/activation copy",
            "weight_h2d_bytes_per_run": sum(t.numel() * t.element_size() for t in raw_host.values()),
            "median": {key: statistics.median(row[key] for row in streamed) for key in streamed[0]},
            "samples": streamed,
        },
        "tokens_per_second": None,
        "unmeasured": ["down", "attention", "KV", "window_paging", "generation"],
        "measurement_limits": [
            "CUDA event spans can include host launch gaps; span fraction is NOT SM utilization or occupancy.",
            "Logical weight bytes / graph time is NOT measured physical DRAM traffic.",
            "Raw Q4 baseline is this Triton prototype, NOT optimized llama.cpp.",
            "One-layer residency does NOT imply all 64 layers fit or avoid per-token window transfers.",
            "Source SHA256 has not been matched to a trusted upstream digest.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real resident gate/up, not full-model speed")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--cpu-threads", default="1,2,4,8")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.artifact, repeats=args.repeats,
                       cpu_threads=tuple(int(x) for x in args.cpu_threads.split(",")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "configurations"}, indent=2))


if __name__ == "__main__":
    main()
