from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resident_residual_format import ResidentArtifact  # noqa: E402


def parse_threads(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or any(item < 1 for item in values):
        raise ValueError("threads must be a nonempty list of positive integers")
    return values


def median_projection_ms(
    coefficient: np.ndarray,
    group_sums: np.ndarray,
    *,
    threads: int,
    warmup: int,
    repeats: int,
) -> float:
    output = np.empty(coefficient.shape[0], dtype=coefficient.dtype)
    with threadpool_limits(limits=threads):
        for _ in range(warmup):
            np.matmul(coefficient, group_sums, out=output)
        samples = []
        for _ in range(repeats):
            begin = time.perf_counter()
            np.matmul(coefficient, group_sums, out=output)
            samples.append((time.perf_counter() - begin) * 1000.0)
    return float(np.median(samples))


def run(
    artifact_path: Path,
    *,
    threads: tuple[int, ...],
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    with ResidentArtifact.open(artifact_path, verify_hashes=False) as artifact:
        cols = int(artifact.projections["gate"]["cols"])
        rows = int(artifact.projections["gate"]["rows"])
        values = np.random.default_rng(seed).standard_normal(cols).astype(np.float32)
        sums64 = values.astype(np.float64).reshape(-1, 32).sum(axis=1)
        sums32 = values.reshape(-1, 32).sum(axis=1, dtype=np.float32)
        coefficient64 = {
            name: np.asarray(artifact.arrays[name]["coefficient"], dtype=np.float64)
            for name in ("gate", "up")
        }
        coefficient32 = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in coefficient64.items()
        }
        output64 = {
            name: coefficient64[name] @ sums64
            for name in ("gate", "up")
        }
        output32 = {
            name: coefficient32[name] @ sums32
            for name in ("gate", "up")
        }
        precision = {}
        for name in ("gate", "up"):
            delta = output32[name].astype(np.float64) - output64[name]
            precision[name] = {
                "relative_l2_vs_fp64": float(
                    np.linalg.norm(delta)
                    / max(float(np.linalg.norm(output64[name])), 1.0e-20)
                ),
                "max_abs_vs_fp64": float(np.max(np.abs(delta))),
            }
        timings = []
        for count in threads:
            fp64_ms = sum(
                median_projection_ms(
                    coefficient64[name],
                    sums64,
                    threads=count,
                    warmup=warmup,
                    repeats=repeats,
                )
                for name in ("gate", "up")
            )
            fp32_ms = sum(
                median_projection_ms(
                    coefficient32[name],
                    sums32,
                    threads=count,
                    warmup=warmup,
                    repeats=repeats,
                )
                for name in ("gate", "up")
            )
            timings.append({
                "threads": count,
                "gate_up_fp64_sum_ms": fp64_ms,
                "gate_up_fp32_sum_ms": fp32_ms,
            })
        timings.sort(key=lambda item: item["gate_up_fp32_sum_ms"])
        return {
            "status": "cpu_base_precision_thread_sweep",
            "layer": int(artifact.manifest["layer"]),
            "dimensions": {
                "rows": rows,
                "cols": cols,
                "group_count": cols // 32,
            },
            "coefficient_storage": {
                "fp64_gate_up_bytes": sum(
                    item.nbytes for item in coefficient64.values()
                ),
                "fp32_gate_up_bytes": sum(
                    item.nbytes for item in coefficient32.values()
                ),
            },
            "precision": precision,
            "timings": timings,
            "best_fp32": timings[0],
            "scope": (
                "CPU base GEMV only. This does not measure residual GPU work, "
                "H2D, SwiGLU, down, attention, or token throughput."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep exact base GEMV precision and CPU thread count"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--threads", default="1,2,4,8,12,16,24")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 3:
        raise ValueError("warmup must be positive and repeats must be at least 3")
    report = run(
        args.artifact,
        threads=parse_threads(args.threads),
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
