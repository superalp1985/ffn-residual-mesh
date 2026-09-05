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
from resident_residual_cuda import ResidentGateUp  # noqa: E402
from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_window_scheduler import ResidentWindowScheduler  # noqa: E402


def benchmark(artifact_path: Path, *, repeats: int = 5) -> dict:
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    with ResidentArtifact.open(artifact_path, verify_hashes=True) as artifact:
        runner = ResidentGateUp(artifact)
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
        x = np.random.default_rng(540).standard_normal(runner.cols).astype(np.float32)
        runner.run(x, return_outputs=False)
        rows = []
        with ResidentPackageCache(packages, scheduler=scheduler) as cache:
            cache.acquire(0)
            cache.release(0)
            for _ in range(repeats):
                ticket = cache.prefetch_async([1])[0]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                gated_end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(runner.stream):
                    start.record()
                    runner.launch_residuals()
                    end.record()
                cache.wait_prefetch(1, stream=runner.stream)
                with torch.cuda.stream(runner.stream):
                    dependent = cache.package(1)["gate_residual"].sum()
                    gated_end.record()
                gated_end.synchronize()
                cache.wait_prefetch(1)
                rows.append({
                    "copy_ms": float(ticket["copy_ms"]),
                    "residual_kernel_ms": float(start.elapsed_time(end)),
                    "kernel_plus_dependency_ms": float(start.elapsed_time(gated_end)),
                    "exposed_extension_ms": max(
                        0.0,
                        float(start.elapsed_time(gated_end) - start.elapsed_time(end)),
                    ),
                    "dependent_checksum": float(dependent.detach().cpu()),
                })
                cache.release(1)
                cache.evict(1)
            median = {
                key: float(np.median([row[key] for row in rows]))
                for key in rows[0] if key != "dependent_checksum"
            }
            return {
                "status": "measured_async_resident_ffn_overlap_probe",
                "artifact": str(artifact.directory),
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "package_bytes": package_bytes,
                "weight_h2d_bytes": cache.traffic["weight_h2d_bytes"],
                "resident_hit_weight_h2d_bytes": 0,
                "median": median,
                "samples": rows,
                "limits": [
                    "The copied package is a duplicate of one real layer artifact.",
                    "The residual kernel is real, but this is not a two-layer model run.",
                    "No end-to-end generation, attention, KV, or token/s is measured.",
                ],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure async residual-copy overlap with real Triton kernel")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.artifact, repeats=args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
