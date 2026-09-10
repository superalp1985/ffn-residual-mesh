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

from benchmark_resident_device_chain_multi import make_down  # noqa: E402
from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_tiled_ffn import TiledResidentGateUp  # noqa: E402


def _enqueue_chain(
    runners: list[TiledResidentGateUp],
    downs: list[object],
    activation: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    for runner, down in zip(runners, downs):
        runner.run_device(
            activation,
            down=down,
            stream=stream,
            return_outputs=False,
            synchronize=False,
            measure_events=False,
        )
        activation = down.output


def benchmark(
    artifacts: list[Path],
    layers: list[int],
    model: Path,
    *,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if len(artifacts) != len(layers) or len(layers) < 2:
        raise ValueError("at least two matching artifact/layer entries are required")
    if warmup < 1 or repeats < 3:
        raise ValueError("warmup must be positive and repeats must be at least 3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with ExitStack() as stack:
        metas = [
            stack.enter_context(ResidentArtifact.open(path, verify_hashes=False))
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
        try:
            for layer in layers:
                down, _ = make_down(model, layer)
                downs.append(down)

            rng = np.random.default_rng(seed)
            activation = torch.from_numpy(
                rng.standard_normal(runners[0].cols).astype(np.float32)
            ).cuda()
            stream = torch.cuda.Stream()

            for _ in range(warmup):
                with torch.cuda.stream(stream):
                    _enqueue_chain(runners, downs, activation, stream)
                    warmup_done = torch.cuda.Event()
                    warmup_done.record(stream)
                warmup_done.synchronize()

            graph = torch.cuda.CUDAGraph()
            stream.synchronize()
            capture_begin = time.perf_counter()
            with torch.cuda.graph(graph, stream=stream):
                _enqueue_chain(runners, downs, activation, stream)
            stream.synchronize()
            capture_ms = (time.perf_counter() - capture_begin) * 1000.0

            replay_ms: list[float] = []
            enqueue_ms: list[float] = []
            for _ in range(repeats):
                begin = time.perf_counter()
                with torch.cuda.stream(stream):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    graph.replay()
                    end.record(stream)
                enqueue_ms.append((time.perf_counter() - begin) * 1000.0)
                end.synchronize()
                replay_ms.append(float(start.elapsed_time(end)))

            return {
                "status": "measured_multi_layer_cuda_graph_resident_device_chain",
                "layers": layers,
                "layer_count": len(layers),
                "warmup": warmup,
                "repeats": repeats,
                "cuda_graph_capture_ms": capture_ms,
                "cuda_graph_replay_ms_median": float(np.median(replay_ms)),
                "cuda_graph_replay_ms_samples": replay_ms,
                "cuda_graph_cpu_enqueue_ms_median": float(np.median(enqueue_ms)),
                "cuda_graph_cpu_enqueue_ms_samples": enqueue_ms,
                "activation_h2d_bytes_per_layer": [0] * len(layers),
                "activation_d2d_bytes_per_layer": [0] * len(layers),
                "base_h2d_bytes_per_layer": [0] * len(layers),
                "resident_weight_h2d_bytes_per_layer": [0] * len(layers),
                "gate_up_kernel_config_per_layer": [
                    {
                        "block_rows": runner.block_rows,
                        "num_warps": runner.num_warps,
                        "base_block_groups": runner.base_block_groups,
                        "auto_tuned": runner.base_schedule_auto_tuned,
                    }
                    for runner in runners
                ],
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "note": (
                    "The graph replays the fixed-address two-layer dependency chain "
                    "for one resident activation. This isolates launch-gap reduction; "
                    "it is not an end-to-end generation throughput measurement."
                ),
            }
        finally:
            for runner in runners:
                runner.close()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.artifact) != len(args.layer):
        parser.error("--artifact and --layer must be repeated the same number of times")
    report = benchmark(
        args.artifact,
        args.layer,
        args.model,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
