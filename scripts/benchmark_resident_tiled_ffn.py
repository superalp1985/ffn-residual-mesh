from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_tiled_ffn import TiledResidentGateUp  # noqa: E402
from sweep_mixed_down_kernel import build_projection  # noqa: E402


def reference_dot(tensor, x: np.ndarray) -> np.ndarray:
    output = np.empty(int(tensor.shape[1]), dtype=np.float64)
    for start in range(0, len(output), 128):
        stop = min(start + 128, len(output))
        output[start:stop] = (
            dequantize(tensor.data[start:stop], tensor.tensor_type).astype(np.float64) @ x
        )
    return output


def benchmark(
    artifact_path: Path,
    model: Path,
    *,
    warmup: int,
    repeats: int,
    seed: int,
    cpu_threads: int,
    block_rows: int,
    num_warps: int,
    base_block_groups: int,
    down_block_rows: int,
    down_num_warps: int,
    down_block_qblocks: int,
) -> dict[str, object]:
    if warmup < 1 or repeats < 3:
        raise ValueError("warmup must be positive and repeats must be at least 3")
    with ResidentArtifact.open(artifact_path, verify_hashes=True) as artifact:
        source = Path(artifact.manifest["source"]["path"])
        layer = int(artifact.manifest["layer"])
        reader = GGUFReader(source)
        try:
            tensors = {item.name: item for item in reader.tensors}
            down_tensor = tensors[f"blk.{layer}.ffn_down.weight"]
            down, down_meta = build_projection(
                down_tensor,
                block_rows=down_block_rows,
                num_warps=down_num_warps,
                block_qblocks=down_block_qblocks,
            )
            rng = np.random.default_rng(seed)
            inputs = rng.standard_normal((repeats + 1, artifact.projections["gate"]["cols"])).astype(
                np.float32
            )
            with threadpool_limits(limits=cpu_threads):
                gate_ref = reference_dot(tensors[f"blk.{layer}.ffn_gate.weight"], inputs[0])
                up_ref = reference_dot(tensors[f"blk.{layer}.ffn_up.weight"], inputs[0])
                hidden_ref = gate_ref * np.exp(-np.logaddexp(0, -gate_ref)) * up_ref
                down_ref = reference_dot(down_tensor, hidden_ref)
        finally:
            reader.data._mmap.close()

        runner = TiledResidentGateUp(
            artifact,
            tile_rows=int(artifact.projections["gate"]["rows"]),
            persistent=True,
            base_on_gpu=True,
            block_rows=block_rows,
            num_warps=num_warps,
            base_block_groups=base_block_groups,
            auto_tune_q5_base=True,
        )
        try:
            x = torch.from_numpy(inputs[0]).cuda()
            for _ in range(warmup):
                runner.run_device(
                    x,
                    down=down,
                    return_outputs=False,
                    measure_events=False,
                )
            validation = runner.run_device(
                x,
                down=down,
                return_outputs=True,
                measure_events=True,
            )
            delta = validation["down"].astype(np.float64) - down_ref
            samples: list[dict[str, object]] = []
            for index in range(1, repeats + 1):
                device_x = torch.from_numpy(inputs[index]).cuda()
                begin = time.perf_counter()
                result = runner.run_device(
                    device_x,
                    down=down,
                    return_outputs=False,
                    measure_events=True,
                )
                samples.append(
                    {
                        "host_wall_ms": (time.perf_counter() - begin) * 1000.0,
                        "gpu_fused_base_residual_swiglu_ms": result[
                            "fused_base_residual_swiglu_ms"
                        ],
                        "gpu_base_reduce_ms": result["base_compute_ms"],
                        "gpu_down_ms": result["down_stream_ms"],
                    }
                )
            median = {
                key: float(np.median([float(sample[key]) for sample in samples]))
                for key in samples[0]
            }
            return {
                "status": "measured_single_layer_full_ffn_gpu_resident_q5",
                "layer": layer,
                "dimensions": {
                    "hidden": runner.cols,
                    "ffn": runner.rows,
                    "down_out": down.rows,
                },
                "gate_up_quant_types": {
                    name: artifact.projections[name]["source"]["type_name"]
                    for name in ("gate", "up")
                },
                "residual_bits": dict(runner.bits),
                "requested_gate_up_config": {
                    "block_rows": block_rows,
                    "num_warps": num_warps,
                    "base_block_groups": base_block_groups,
                },
                "resolved_gate_up_config": {
                    "block_rows": runner.block_rows,
                    "num_warps": runner.num_warps,
                    "base_block_groups": runner.base_block_groups,
                    "auto_tuned": runner.base_schedule_auto_tuned,
                },
                "down_kernel_config": down_meta,
                "warmup": warmup,
                "repeats": repeats,
                "cpu_threads": cpu_threads,
                "timing_median_ms": median,
                "samples": samples,
                "critical_gpu_ms": float(
                    median["gpu_fused_base_residual_swiglu_ms"] + median["gpu_down_ms"]
                ),
                "host_wall_ms": median["host_wall_ms"],
                "dynamic_h2d_bytes_per_token": 0,
                "activation_d2d_bytes_per_token": 0,
                "resident_weight_h2d_bytes_per_token": 0,
                "resident_payload_bytes": int(
                    runner.cache.cache.scheduler.layer_bytes(0)
                    + sum(
                        value.numel() * value.element_size()
                        for value in runner.base_resident.values()
                    )
                    + down.raw.numel()
                    + (down.kvalues.numel() * 4 if hasattr(down, "kvalues") else 0)
                ),
                "resident_vram_peak_bytes": int(torch.cuda.max_memory_allocated()),
                "output_rel_l2": float(
                    np.linalg.norm(delta) / max(float(np.linalg.norm(down_ref)), 1e-20)
                ),
                "output_max_abs": float(np.abs(delta).max()),
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "source_path": str(source),
                "artifact_path": str(artifact.directory),
                "quality_scope": "synthetic_one_token_input_exact_weight_reconstruction",
                "unmeasured": [
                    "attention",
                    "KV",
                    "layer paging",
                    "CPU model orchestration",
                    "end-to-end generation throughput",
                ],
            }
        finally:
            runner.close()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--block-rows", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--base-block-groups", type=int, choices=(8, 16, 32, 64, 128, 256), default=256)
    parser.add_argument("--down-block-rows", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--down-num-warps", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--down-block-qblocks", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(
        args.artifact,
        args.model,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        cpu_threads=args.cpu_threads,
        block_rows=args.block_rows,
        num_warps=args.num_warps,
        base_block_groups=args.base_block_groups,
        down_block_rows=args.down_block_rows,
        down_num_warps=args.down_num_warps,
        down_block_qblocks=args.down_block_qblocks,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
