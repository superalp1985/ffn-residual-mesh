from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_tiled_ffn import TiledResidentGateUp  # noqa: E402
from benchmark_resident_device_chain import make_down  # noqa: E402


def _run_chain(
    runners: list[TiledResidentGateUp],
    downs: list[object],
    activation: torch.Tensor,
    *,
    stream: torch.cuda.Stream,
    synchronize_each: bool,
    record_markers: bool,
) -> tuple[list[torch.cuda.Event], torch.cuda.Event]:
    markers: list[torch.cuda.Event] = []
    with torch.cuda.stream(stream):
        for runner, down in zip(runners, downs):
            runner.run_device(
                activation,
                down=down,
                stream=stream,
                return_outputs=False,
                synchronize=synchronize_each,
                measure_events=False,
            )
            if record_markers:
                marker = torch.cuda.Event(enable_timing=True)
                marker.record(stream)
                markers.append(marker)
            activation = down.output
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
    return markers, end


def benchmark(
    artifacts: list[Path],
    layers: list[int],
    model: Path,
    *,
    warmup: int,
    repeats: int,
    seed: int,
    down_configs: list[dict[str, int]] | None = None,
) -> dict[str, object]:
    if len(artifacts) != len(layers) or len(layers) < 2:
        raise ValueError("at least two matching artifact/layer entries are required")
    if warmup < 1 or repeats < 3:
        raise ValueError("warmup must be positive and repeats must be at least 3")
    if down_configs is None:
        down_configs = [{} for _ in layers]
    if len(down_configs) != len(layers):
        raise ValueError("down_configs must match the layer count")

    with ExitStack() as stack:
        metas = [
            stack.enter_context(
                ResidentArtifact.open(path, verify_hashes=False)
            )
            for path in artifacts
        ]
        hidden_widths = {
            int(meta.projections["gate"]["cols"]) for meta in metas
        }
        if len(hidden_widths) != 1:
            raise ValueError("device chain requires matching hidden widths")
        runners = [
            TiledResidentGateUp(
                meta,
                tile_rows=int(meta.projections["gate"]["rows"]),
                persistent=True,
                base_on_gpu=True,
            )
            for meta in metas
        ]
        downs: list[object] = []
        down_metadata: list[dict[str, object]] = []
        try:
            for layer, config in zip(layers, down_configs):
                down, metadata = make_down(model, layer, **config)
                downs.append(down)
                down_metadata.append(metadata)

            rng = np.random.default_rng(seed)
            activation = torch.from_numpy(
                rng.standard_normal(runners[0].cols).astype(np.float32)
            ).cuda()
            work_stream = torch.cuda.Stream()

            for _ in range(warmup):
                _run_chain(
                    runners,
                    downs,
                    activation,
                    stream=work_stream,
                    synchronize_each=False,
                    record_markers=False,
                )[1].synchronize()

            host_sync_wall: list[float] = []
            host_sync_gpu: list[float] = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                with torch.cuda.stream(torch.cuda.current_stream()):
                    start.record()
                    _run_chain(
                        runners,
                        downs,
                        activation,
                        stream=torch.cuda.current_stream(),
                        synchronize_each=True,
                        record_markers=False,
                    )
                    end.record()
                end.synchronize()
                host_sync_wall.append((time.perf_counter() - begin) * 1000)
                host_sync_gpu.append(float(start.elapsed_time(end)))

            tail_wall: list[float] = []
            tail_gpu: list[float] = []
            tail_layer_spans: list[list[float]] = []
            for _ in range(repeats):
                begin = time.perf_counter()
                with torch.cuda.stream(work_stream):
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    markers, end = _run_chain(
                        runners,
                        downs,
                        activation,
                        stream=work_stream,
                        synchronize_each=False,
                        record_markers=True,
                    )
                end.synchronize()
                tail_wall.append((time.perf_counter() - begin) * 1000)
                tail_gpu.append(float(start.elapsed_time(end)))
                previous = start
                spans: list[float] = []
                for marker in markers:
                    spans.append(float(previous.elapsed_time(marker)))
                    previous = marker
                tail_layer_spans.append(spans)

            async_enqueue: list[float] = []
            async_gpu: list[float] = []
            for _ in range(repeats):
                begin = time.perf_counter()
                with torch.cuda.stream(work_stream):
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _, end = _run_chain(
                        runners,
                        downs,
                        activation,
                        stream=work_stream,
                        synchronize_each=False,
                        record_markers=False,
                    )
                async_enqueue.append((time.perf_counter() - begin) * 1000)
                end.synchronize()
                async_gpu.append(float(start.elapsed_time(end)))

            median_layer_spans = np.median(
                np.asarray(tail_layer_spans, dtype=np.float64), axis=0
            )
            return {
                "status": "measured_multi_layer_gpu_resident_device_chain",
                "layers": layers,
                "layer_count": len(layers),
                "warmup": warmup,
                "repeats": repeats,
                "host_sync_chain_wall_ms_median": float(np.median(host_sync_wall)),
                "host_sync_chain_gpu_ms_median": float(np.median(host_sync_gpu)),
                "tail_sync_chain_wall_ms_median": float(np.median(tail_wall)),
                "tail_sync_chain_gpu_ms_median": float(np.median(tail_gpu)),
                "tail_sync_layer_span_ms_median": [
                    float(value) for value in median_layer_spans
                ],
                "async_chain_gpu_ms_median": float(np.median(async_gpu)),
                "async_cpu_enqueue_ms_median": float(np.median(async_enqueue)),
                "activation_h2d_bytes_per_layer": [0] * len(layers),
                "activation_d2d_bytes_per_layer": [0] * len(layers),
                "base_h2d_bytes_per_layer": [0] * len(layers),
                "resident_weight_h2d_bytes_per_layer": [0] * len(layers),
                "down_kernel_config_per_layer": down_metadata,
                "kernel_mode": "device_activation_fused_base_residual_swiglu",
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "note": (
                    "Same-stream multi-layer device chain. Layer spans are "
                    "CUDA event intervals between markers; this is not an "
                    "SM-occupancy, DMA-engine, or generation-throughput metric."
                ),
            }
        finally:
            for runner in runners:
                runner.close()
            del downs
            torch.cuda.empty_cache()


def _parse_config(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in text.split(","):
        key, separator, value = item.partition("=")
        if not separator or key not in {
            "block_rows",
            "num_warps",
            "block_qblocks",
        }:
            raise ValueError(
                "down config must be comma-separated block_rows, num_warps, "
                "and block_qblocks assignments"
            )
        values[key] = int(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark an N-layer GPU-resident FFN device chain"
    )
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--down-config", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if len(args.artifact) != len(args.layer):
        parser.error("--artifact and --layer must be repeated the same number of times")
    if args.down_config and len(args.down_config) != len(args.layer):
        parser.error("--down-config must match the layer count")
    configs = [_parse_config(item) for item in args.down_config]
    if not configs:
        configs = [{} for _ in args.layer]
    report = benchmark(
        args.artifact,
        args.layer,
        args.model,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        down_configs=configs,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
