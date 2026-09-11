from __future__ import annotations

"""Compare original quantized and exact-split resident FFN device paths.

This is a paired latency diagnostic. Both paths keep their weights on CUDA
after setup, use fixed activation addresses, and are replayed through CUDA
Graphs. It does not claim end-to-end model throughput or hardware counters.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resident_residual_cuda import (  # noqa: E402
    launch_fused_gate_up_base_residual,
    launch_merge_swiglu,
)
from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_tiled_ffn import TiledResidentGateUp  # noqa: E402
from sweep_mixed_down_kernel import build_projection  # noqa: E402


_BLOCK_ROWS = {1, 2, 4, 8}
_WARPS = {2, 4, 8}
_GROUPS = {8, 16, 32, 64, 128, 256}


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_host = actual.detach().cpu().numpy().astype(np.float64, copy=False)
    reference_host = reference.detach().cpu().numpy().astype(np.float64, copy=False)
    return float(
        np.linalg.norm(actual_host - reference_host)
        / max(float(np.linalg.norm(reference_host)), 1e-20)
    )


def _validate_protocol(
    *,
    repeats: int,
    inner: int,
    warmup_seconds: float,
    configs: tuple[tuple[int, int, int], ...],
) -> None:
    if repeats < 1 or inner < 1:
        raise ValueError("repeats and inner must be positive")
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("warmup_seconds must be finite and nonnegative")
    if not configs:
        raise ValueError("at least one split kernel configuration is required")
    if len(set(configs)) != len(configs):
        raise ValueError("split kernel configurations must be unique")
    for block_rows, num_warps, block_groups in configs:
        if block_rows not in _BLOCK_ROWS:
            raise ValueError("block_rows must be one of 1, 2, 4, 8")
        if num_warps not in _WARPS:
            raise ValueError("num_warps must be one of 2, 4, 8")
        if block_groups not in _GROUPS:
            raise ValueError("block_groups must be one of 8, 16, 32, 64, 128, 256")


def _load_projection_inputs(
    source: Path,
    layer: int,
) -> tuple[dict[str, SimpleNamespace], SimpleNamespace]:
    reader = GGUFReader(source)
    try:
        tensors = {item.name: item for item in reader.tensors}
        selected = {}
        for name in ("gate", "up", "down"):
            tensor = tensors[f"blk.{layer}.ffn_{name}.weight"]
            selected[name] = SimpleNamespace(
                data=np.array(tensor.data, copy=True),
                tensor_type=int(tensor.tensor_type),
                shape=tuple(int(value) for value in tensor.shape),
            )
        return (
            {"gate": selected["gate"], "up": selected["up"]},
            selected["down"],
        )
    finally:
        reader.data._mmap.close()


def _reference_ffn(
    projections: dict[str, SimpleNamespace],
    down: SimpleNamespace,
    x: np.ndarray,
    *,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Independent, untimed reference with bounded dequantization memory."""
    def dot(tensor: SimpleNamespace, vector: np.ndarray) -> np.ndarray:
        result = np.empty(int(tensor.shape[1]), dtype=np.float64)
        for start in range(0, len(result), chunk_rows):
            stop = min(start + chunk_rows, len(result))
            weights = dequantize(
                tensor.data[start:stop], tensor.tensor_type
            ).astype(np.float64)
            result[start:stop] = weights @ vector
        return result

    value = np.asarray(x, dtype=np.float64)
    gate = dot(projections["gate"], value)
    up = dot(projections["up"], value)
    return dot(down, gate * np.exp(-np.logaddexp(0, -gate)) * up)


def _capture_native_graph(
    gate,
    up,
    down,
    x: torch.Tensor,
    *,
    stage_events: dict[str, torch.cuda.Event] | None = None,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream, torch.Tensor, dict[str, torch.Tensor]]:
    rows = int(gate.rows)
    stream = torch.cuda.Stream()
    zero_gate = torch.zeros(rows, dtype=torch.float32, device="cuda")
    zero_up = torch.zeros(rows, dtype=torch.float32, device="cuda")
    gate_output = torch.empty(rows, dtype=torch.float32, device="cuda")
    up_output = torch.empty(rows, dtype=torch.float32, device="cuda")
    swiglu_output = torch.empty(rows, dtype=torch.float32, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())

    def launch() -> None:
        if stage_events is not None:
            stage_events["begin"].record()
        gate.launch(x)
        up.launch(x)
        launch_merge_swiglu(
            zero_gate,
            zero_up,
            gate.output,
            up.output,
            gate_output,
            up_output,
            swiglu_output,
            rows=rows,
        )
        if stage_events is not None:
            stage_events["gate_up_end"].record()
        down.launch(swiglu_output)
        if stage_events is not None:
            stage_events["end"].record()

    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    # CUDA Graph replays retain device addresses, not Python ownership of each
    # temporary. Keep every captured intermediate alive for the graph lifetime.
    return graph, stream, down.output, {
        "zero_gate": zero_gate,
        "zero_up": zero_up,
        "gate_output": gate_output,
        "up_output": up_output,
        "swiglu_output": swiglu_output,
    }


def _capture_split_graph(
    runner: TiledResidentGateUp,
    down,
    x: torch.Tensor,
    *,
    stage_events: dict[str, torch.cuda.Event] | None = None,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream, torch.Tensor]:
    package = runner.cache.package(0)
    if package is None or runner.device_group_sums is None:
        raise RuntimeError("persistent split runner did not retain its resident package")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch() -> None:
        if stage_events is not None:
            stage_events["begin"].record()
        torch.sum(
            x.view(-1, 32),
            dim=1,
            dtype=torch.float32,
            out=runner.device_group_sums,
        )
        launch_fused_gate_up_base_residual(
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
            block_groups=runner.base_block_groups,
            gate_bits=runner.bits["gate"],
            up_bits=runner.bits["up"],
        )
        if stage_events is not None:
            stage_events["gate_up_end"].record()
        down.launch(runner.output["swiglu"])
        if stage_events is not None:
            stage_events["end"].record()

    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    return graph, stream, down.output


def _graph_ms(
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    *,
    inner: int,
) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        begin.record()
        for _ in range(inner):
            graph.replay()
        end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / inner


def _replay_with_input(
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    target: torch.Tensor,
    value: np.ndarray,
) -> None:
    source = torch.from_numpy(np.ascontiguousarray(value))
    with torch.cuda.stream(stream):
        target.copy_(source, non_blocking=True)
        graph.replay()
    stream.synchronize()


def _profile_stages(
    native,
    split_entries: dict[str, dict[str, object]],
    *,
    repeats: int,
    warmup_seconds: float,
    seed: int,
) -> dict[str, object]:
    """Separate instrumented pass; headline timings contain no event nodes."""
    graphs = {}
    buffers = []
    for name in ("native", *split_entries):
        events = {
            key: torch.cuda.Event(enable_timing=True, external=True)
            for key in ("begin", "gate_up_end", "end")
        }
        if name == "native":
            graph, stream, _, owned = _capture_native_graph(
                *native, stage_events=events
            )
            buffers.append(owned)
        else:
            entry = split_entries[name]
            graph, stream, _ = _capture_split_graph(
                entry["runner"], entry["down"], entry["x"], stage_events=events
            )
        graphs[name] = (graph, stream, events)
    warmup_end = time.perf_counter() + warmup_seconds
    while time.perf_counter() < warmup_end:
        for graph, stream, _ in graphs.values():
            _graph_ms(graph, stream, inner=4)

    rng = np.random.default_rng(seed)
    samples = {name: [] for name in graphs}
    orders = []
    for _ in range(repeats):
        order = list(graphs)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            graph, stream, events = graphs[name]
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            samples[name].append({
                "gate_up_swiglu_ms": float(
                    events["begin"].elapsed_time(events["gate_up_end"])
                ),
                "down_ms": float(events["gate_up_end"].elapsed_time(events["end"])),
                "span_ms": float(events["begin"].elapsed_time(events["end"])),
            })
    return {
        "includes_event_nodes": True,
        "scope": (
            "Separate instrumented CUDA Graph pass; gate/up includes split group "
            "sums where applicable. Event nodes perturb latency. These are not "
            "the uninstrumented headline measurements or hardware counters."
        ),
        "execution_orders": orders,
        "variants": {
            name: {
                "samples": values,
                "median_ms": {
                    key: float(np.median([item[key] for item in values]))
                    for key in ("gate_up_swiglu_ms", "down_ms", "span_ms")
                },
            }
            for name, values in samples.items()
        },
    }


def benchmark(
    artifact_path: Path,
    *,
    repeats: int = 15,
    inner: int = 4,
    warmup_seconds: float = 1.0,
    configs: tuple[tuple[int, int, int], ...] = ((4, 2, 16),),
    seed: int = 20260911,
    validation_inputs: int = 2,
    profile_stages: bool = False,
) -> dict[str, object]:
    _validate_protocol(
        repeats=repeats,
        inner=inner,
        warmup_seconds=warmup_seconds,
        configs=configs,
    )
    if validation_inputs < 1:
        raise ValueError("validation_inputs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with ResidentArtifact.open(artifact_path, verify_hashes=False) as artifact:
        if not {"gate", "up"}.issubset(artifact.projections):
            raise ValueError("comparison requires both gate and up compiled in the artifact")
        source = Path(artifact.manifest["source"]["path"])
        layer = int(artifact.manifest["layer"])
        gate_up_raw, down_raw = _load_projection_inputs(source, layer)
        native_gate, _ = build_projection(gate_up_raw["gate"])
        native_up, _ = build_projection(gate_up_raw["up"])
        native_down, down_metadata = build_projection(down_raw)

        rows = int(artifact.projections["gate"]["rows"])
        cols = int(artifact.projections["gate"]["cols"])
        rng = np.random.default_rng(seed)
        timed_input = rng.standard_normal(cols).astype(np.float32)
        validation = [
            rng.standard_normal(cols).astype(np.float32)
            for _ in range(validation_inputs)
        ]
        native_x = torch.from_numpy(timed_input.copy()).cuda()
        native_graph, native_stream, native_output, native_buffers = _capture_native_graph(
            native_gate, native_up, native_down, native_x
        )

        split_entries: dict[str, dict[str, object]] = {}
        runners: list[TiledResidentGateUp] = []
        try:
            for block_rows, num_warps, block_groups in configs:
                runner = TiledResidentGateUp(
                    artifact,
                    tile_rows=rows,
                    persistent=True,
                    base_on_gpu=True,
                    block_rows=block_rows,
                    num_warps=num_warps,
                    base_block_groups=block_groups,
                    auto_tune_q5_base=False,
                )
                runners.append(runner)
                split_down, _ = build_projection(down_raw)
                split_x = torch.from_numpy(timed_input.copy()).cuda()
                graph, stream, output = _capture_split_graph(runner, split_down, split_x)
                name = f"split_r{block_rows}_w{num_warps}_g{block_groups}"
                split_entries[name] = {
                    "runner": runner,
                    "down": split_down,
                    "x": split_x,
                    "graph": graph,
                    "stream": stream,
                    "output": output,
                    "config": {
                        "block_rows": block_rows,
                        "num_warps": num_warps,
                        "base_block_groups": block_groups,
                    },
                }

            graphs: dict[str, tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]] = {
                "native": (native_graph, native_stream)
            }
            graphs.update(
                {
                    name: (entry["graph"], entry["stream"])
                    for name, entry in split_entries.items()
                }
            )
            all_names = list(graphs)
            warmup_end = time.perf_counter() + warmup_seconds
            warmup_cycles = 0
            while time.perf_counter() < warmup_end:
                for name in all_names:
                    graph, stream = graphs[name]
                    with torch.cuda.stream(stream):
                        graph.replay()
                    stream.synchronize()
                warmup_cycles += 1

            samples: dict[str, list[float]] = {name: [] for name in all_names}
            execution_orders: list[list[str]] = []
            for _ in range(repeats):
                order = list(all_names)
                rng.shuffle(order)
                execution_orders.append(order)
                for name in order:
                    graph, stream = graphs[name]
                    samples[name].append(_graph_ms(graph, stream, inner=inner))

            stage_profile = (
                _profile_stages(
                    (native_gate, native_up, native_down, native_x),
                    split_entries,
                    repeats=repeats,
                    warmup_seconds=warmup_seconds,
                    seed=seed + 1,
                )
                if profile_stages else None
            )

            native_errors: list[float] = []
            split_errors: dict[str, list[float]] = {
                name: [] for name in split_entries
            }
            reference_errors: dict[str, list[float]] = {
                name: [] for name in all_names
            }
            for value in validation:
                reference = torch.from_numpy(
                    _reference_ffn(gate_up_raw, down_raw, value)
                )
                _replay_with_input(native_graph, native_stream, native_x, value)
                native_result = native_output.detach().clone()
                native_errors.append(_relative_l2(native_result, reference))
                reference_errors["native"].append(native_errors[-1])
                for name, entry in split_entries.items():
                    _replay_with_input(
                        entry["graph"],
                        entry["stream"],
                        entry["x"],
                        value,
                    )
                    split_errors[name].append(
                        _relative_l2(entry["output"], native_result)
                    )
                    reference_errors[name].append(
                        _relative_l2(entry["output"], reference)
                    )

            variants: dict[str, dict[str, object]] = {
                "native": {
                    "kind": "original_packed_quantized",
                    "median_ms": float(np.median(samples["native"])),
                    "p95_ms": float(np.percentile(samples["native"], 95)),
                    "samples_ms": samples["native"],
                    "max_relative_l2": max(native_errors),
                    "max_reference_relative_l2": max(reference_errors["native"]),
                    "comparison_reference": "gguf_dequantized_fp64",
                    "validation_inputs": validation_inputs,
                    "weight_bytes": int(
                        gate_up_raw["gate"].data.nbytes
                        + gate_up_raw["up"].data.nbytes
                        + down_raw.data.nbytes
                    ),
                }
            }
            for name, entry in split_entries.items():
                runner = entry["runner"]
                split_weight_bytes = int(
                    runner.cache.cache.scheduler.layer_bytes(0)
                    + sum(
                        value.numel() * value.element_size()
                        for value in runner.base_resident.values()
                    )
                    + entry["down"].raw.numel() * entry["down"].raw.element_size()
                    + (
                        entry["down"].kvalues.numel() * entry["down"].kvalues.element_size()
                        if hasattr(entry["down"], "kvalues")
                        else 0
                    )
                )
                variants[name] = {
                    "kind": "exact_centered_affine_split",
                    "config": entry["config"],
                    "median_ms": float(np.median(samples[name])),
                    "p95_ms": float(np.percentile(samples[name], 95)),
                    "samples_ms": samples[name],
                    "max_relative_l2": max(split_errors[name]),
                    "max_reference_relative_l2": max(reference_errors[name]),
                    "comparison_reference": "native_gpu",
                    "validation_inputs": validation_inputs,
                    "weight_bytes": split_weight_bytes,
                }

            native_weight_bytes = int(
                gate_up_raw["gate"].data.nbytes
                + gate_up_raw["up"].data.nbytes
                + down_raw.data.nbytes
            )
            split_weight_bytes = int(variants[next(iter(split_entries))]["weight_bytes"])
            return {
                "status": "paired_resident_ffn_latency_diagnostic",
                "layer": layer,
                "dimensions": {"hidden": cols, "ffn": rows},
                "quantization": {
                    "gate": artifact.projections["gate"]["source"]["type_name"],
                    "up": artifact.projections["up"]["source"]["type_name"],
                    "down": down_metadata["quant_type"],
                },
                "protocol": {
                    "repeats": repeats,
                    "inner": inner,
                    "warmup_seconds": warmup_seconds,
                    "warmup_cycles": warmup_cycles,
                    "randomized_variant_order": True,
                    "fixed_cuda_addresses": True,
                    "activation_h2d_in_timed_graph": False,
                },
                "execution_orders": execution_orders,
                "variants": variants,
                "stage_profile": stage_profile,
                "reference_kind": "gguf_dequantized_fp64",
                "validation_passed": all(
                    math.isfinite(value) and value < 1e-4
                    for errors in reference_errors.values()
                    for value in errors
                ),
                "native_weight_bytes": native_weight_bytes,
                "split_weight_bytes": split_weight_bytes,
                "runtime_h2d_bytes": 0,
                "weight_h2d_bytes_per_invocation": 0,
                "inner_invocations": inner,
                "tokens_per_second": None,
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            "scope": (
                "One fixed-shape FFN comparison. It excludes attention, KV cache, "
                "layer paging, host orchestration, input-update transfer and "
                "end-to-end generation."
            ),
        }
        finally:
            for runner in runners:
                runner.close()
            torch.cuda.empty_cache()


def _parse_configs(value: str) -> tuple[tuple[int, int, int], ...]:
    configs = []
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "each config must use block_rows:num_warps:base_block_groups"
            )
        try:
            configs.append(tuple(int(part) for part in parts))
        except ValueError as error:
            raise argparse.ArgumentTypeError("config values must be integers") from error
    return tuple(configs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired original-versus-split resident FFN latency diagnostic"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--inner", type=int, default=4)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--configs", type=_parse_configs, default=((4, 2, 16),))
    parser.add_argument("--validation-inputs", type=int, default=2)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(
        args.artifact,
        repeats=args.repeats,
        inner=args.inner,
        warmup_seconds=args.warmup_seconds,
        configs=args.configs,
        validation_inputs=args.validation_inputs,
        seed=args.seed,
        profile_stages=args.profile_stages,
    )
    text = json.dumps(report, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
