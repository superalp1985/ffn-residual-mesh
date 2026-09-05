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
from resident_residual_cuda import DirectIQ4NLProjection, DirectQ4Projection, ResidentGateUp  # noqa: E402
from resident_residual_format import ResidentArtifact  # noqa: E402


def reference_dot(tensor, x):
    output = np.empty(int(tensor.shape[1]), dtype=np.float64)
    for start in range(0, len(output), 128):
        stop = min(start + 128, len(output))
        output[start:stop] = dequantize(tensor.data[start:stop], tensor.tensor_type).astype(np.float64) @ x
    return output


def run_resident_ffn(layer_artifact: Path, *, repeats: int = 9, seed: int = 20260905,
                     cpu_threads: int = 8) -> dict:
    if repeats < 3 or cpu_threads < 1:
        raise ValueError("at least 3 repeats and a positive CPU thread count required")
    torch.cuda.reset_peak_memory_stats()
    with ResidentArtifact.open(layer_artifact, verify_hashes=True) as artifact:
        source = Path(artifact.manifest["source"]["path"])
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (artifact.manifest["source"]["bytes"],
                                               artifact.manifest["source"]["mtime_ns"]):
            raise ValueError("source GGUF changed after compilation")
        layer = int(artifact.manifest["layer"])
        rng = np.random.default_rng(seed)
        runner = ResidentGateUp(artifact)
        inputs = rng.standard_normal((repeats + 1, runner.cols)).astype(np.float32)
        reader = GGUFReader(source)
        try:
            tensors = {item.name: item for item in reader.tensors}
            tensor = tensors[f"blk.{layer}.ffn_down.weight"]
            quant = GGMLQuantizationType(int(tensor.tensor_type))
            if quant is GGMLQuantizationType.IQ4_NL:
                down = DirectIQ4NLProjection(tensor.data, int(tensor.shape[0]))
            elif quant is GGMLQuantizationType.Q4_K:
                down = DirectQ4Projection(tensor.data, int(tensor.shape[0]))
            else:
                raise ValueError(f"unsupported down tensor in v1 pipeline: {quant.name}")
            with threadpool_limits(limits=cpu_threads):
                g = reference_dot(tensors[f"blk.{layer}.ffn_gate.weight"], inputs[0])
                u = reference_dot(tensors[f"blk.{layer}.ffn_up.weight"], inputs[0])
                # np.logaddexp avoids overflow for large negative gates.
                h = g * np.exp(-np.logaddexp(0, -g)) * u
                reference = reference_dot(tensor, h)
        finally:
            reader.data._mmap.close()

        samples = []
        with threadpool_limits(limits=cpu_threads):
            # Compile kernels and raise clocks before timing.
            warm_until = time.perf_counter() + 0.5
            while time.perf_counter() < warm_until:
                runner.run(inputs[0], down=down, return_outputs=False)
            result = runner.run(inputs[0], down=down)
            delta = result["down"].astype(np.float64) - reference
            rel_l2 = float(np.linalg.norm(delta) / max(float(np.linalg.norm(reference)), 1e-20))
            if not np.isfinite(rel_l2) or rel_l2 > 1e-4:
                raise ValueError(f"final FFN numerical gate failed: {rel_l2}")
            for x in inputs[1:]:
                result = runner.run(x, down=down, return_outputs=False)
                samples.append(result["timing"])
        median = {
            key: float(np.median([sample[key] for sample in samples]))
            for key in samples[0]
        }
        return {
            "status": "measured_single_layer_full_ffn",
            "layer": layer,
            "dimensions": {"hidden": runner.cols, "ffn": runner.rows},
            "merge_order": "gate_up_before_swiglu",
            "cpu_base_ms": median["cpu_base_ms"],
            "resident_residual_kernel_ms": median["residual_stream_span_ms"],
            "swiglu_down_ms": median["merge_stream_span_ms"] + median["down_stream_span_ms"],
            "critical_ms": median["wall_ms"],
            "dynamic_h2d_bytes": 4 * (runner.cols + 2 * runner.rows),
            "resident_payload_bytes": runner.resident_bytes + down.raw.numel()
                                      + (down.kvalues.numel() * 4 if hasattr(down, "kvalues") else 0),
            "resident_vram_peak_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "vram_note": "allocator process peak; reserved is not whole-device usage",
            "residual_weight_h2d_bytes_per_token": 0,
            "timing": median,
            "samples": samples,
            "down_quant_type": quant.name,
            "output_rel_l2": rel_l2, "output_max_abs": float(np.abs(delta).max()),
            "quality_scope": "synthetic_inputs_not_model_quality",
            "cpu_threads": cpu_threads,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "source_path": str(source), "artifact_path": str(artifact.directory),
            "input_seed": seed,
            "tokens_per_second": None,
            "unmeasured": ["attention", "KV", "window_paging", "generation", "real activation quality"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one complete resident FFN layer")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_resident_ffn(args.artifact, repeats=args.repeats, cpu_threads=args.cpu_threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
