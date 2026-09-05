from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resident_package_cache import ResidentPackageCache  # noqa: E402
from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_window_scheduler import ResidentWindowScheduler  # noqa: E402


def benchmark(artifact_path: Path, *, compute_repeats: int = 8) -> dict:
    if compute_repeats < 1:
        raise ValueError("compute_repeats must be positive")
    with ResidentArtifact.open(artifact_path, verify_hashes=True) as artifact:
        package = {
            "gate_residual": artifact.arrays["gate"]["residual"],
            "gate_alpha": artifact.arrays["gate"]["alpha"],
            "up_residual": artifact.arrays["up"]["residual"],
            "up_alpha": artifact.arrays["up"]["alpha"],
        }
        package_bytes = sum(value.nbytes for value in package.values())
        packages = {0: package, 1: package}
        scheduler = ResidentWindowScheduler(
            window_layers=1,
            vram_budget_bytes=package_bytes * 2 + 64 * 2**20,
            layer_bytes={0: package_bytes, 1: package_bytes},
        )
        with ResidentPackageCache(packages, scheduler=scheduler) as cache:
            cache.acquire(0)
            cache.release(0)
            compute = torch.cuda.Stream()
            left = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
            right = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
            with torch.cuda.stream(compute):
                warm = left @ right
            compute.synchronize()

            begin = time.perf_counter()
            ticket = cache.prefetch_async([1])[0]
            issue_ms = (time.perf_counter() - begin) * 1000
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(compute):
                compute_start.record()
                value = left @ right
                for _ in range(compute_repeats - 1):
                    value = value @ right
                compute_end.record()
            cache.wait_prefetch(1, stream=compute)
            with torch.cuda.stream(compute):
                dependent = cache.package(1)["gate_residual"].sum()
            compute_end_2 = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(compute):
                compute_end_2.record()
            compute_end_2.synchronize()
            compute_ms = compute_start.elapsed_time(compute_end)
            gated_ms = compute_start.elapsed_time(compute_end_2)
            cache.wait_prefetch(1)
            return {
                "status": "measured_async_prefetch_overlap_probe",
                "artifact": str(artifact.directory),
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "package_bytes": package_bytes,
                "prefetch_issue_ms": issue_ms,
                "copy_ms": ticket["copy_ms"],
                "compute_before_wait_ms": compute_ms,
                "compute_plus_dependency_ms": gated_ms,
                "exposed_wait_proxy_ms": max(0.0, gated_ms - compute_ms),
                "weight_h2d_bytes": cache.traffic["weight_h2d_bytes"],
                "resident_hit_weight_h2d_bytes": 0,
                "dependent_checksum": float(dependent.detach().cpu()),
                "limits": [
                    "The duplicate package is the same real layer artifact; this is not a two-layer model run.",
                    "The matmul workload is a synthetic overlap source, not an FFN kernel.",
                    "No token throughput, attention, KV cache, or end-to-end generation is measured.",
                ],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure async residual prefetch overlap")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--compute-repeats", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.artifact, compute_repeats=args.compute_repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
