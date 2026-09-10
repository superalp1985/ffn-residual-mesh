from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resident_residual_format import ResidentArtifact  # noqa: E402
from resident_tiled_ffn import TiledResidentGateUp  # noqa: E402


def measure(
    artifact_path: Path,
    *,
    block_rows: int,
    num_warps: int,
    block_groups: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    with ResidentArtifact.open(artifact_path, verify_hashes=False) as artifact:
        runner = TiledResidentGateUp(
            artifact,
            tile_rows=int(artifact.projections["gate"]["rows"]),
            persistent=True,
            base_on_gpu=True,
            block_rows=block_rows,
            num_warps=num_warps,
            base_block_groups=block_groups,
            auto_tune_q5_base=False,
        )
        try:
            rng = np.random.default_rng(seed)
            x = torch.from_numpy(
                rng.standard_normal(runner.cols).astype(np.float32)
            ).cuda()
            for _ in range(warmup):
                runner.run_device(
                    x,
                    return_outputs=False,
                    measure_events=True,
                )
            samples: list[float] = []
            total_samples: list[float] = []
            for _ in range(repeats):
                begin = time.perf_counter()
                report = runner.run_device(
                    x,
                    return_outputs=False,
                    measure_events=True,
                )
                total_samples.append((time.perf_counter() - begin) * 1000)
                samples.append(float(report["fused_base_residual_swiglu_ms"]))
            return {
                "block_rows": block_rows,
                "num_warps": num_warps,
                "block_groups": block_groups,
                "warmup": warmup,
                "repeats": repeats,
                "fused_ms_median": float(np.median(samples)),
                "fused_ms_p95": float(np.percentile(samples, 95)),
                "wall_ms_median": float(np.median(total_samples)),
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
        finally:
            runner.close()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    configs = [
        (1, 2, 16),
        (1, 4, 16),
        (1, 8, 16),
        (2, 2, 16),
        (2, 4, 16),
        (2, 8, 16),
        (4, 2, 16),
        (4, 4, 16),
        (4, 8, 16),
        (2, 2, 32),
        (2, 4, 32),
        (2, 8, 32),
        (4, 2, 32),
        (4, 4, 32),
        (4, 8, 32),
        (2, 2, 64),
        (2, 4, 64),
        (2, 8, 64),
        (4, 2, 64),
        (4, 4, 64),
        (4, 8, 64),
        (2, 2, 128),
        (2, 4, 128),
        (2, 8, 128),
        (4, 2, 128),
        (4, 4, 128),
        (4, 8, 128),
        (2, 2, 256),
        (2, 4, 256),
        (2, 8, 256),
        (4, 2, 256),
        (4, 4, 256),
        (4, 8, 256),
    ]
    results: list[dict[str, object]] = []
    for index, (block_rows, num_warps, block_groups) in enumerate(configs):
        print(
            f"[{index + 1}/{len(configs)}] "
            f"rows={block_rows} warps={num_warps} groups={block_groups}",
            flush=True,
        )
        result = measure(
            args.artifact,
            block_rows=block_rows,
            num_warps=num_warps,
            block_groups=block_groups,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
        results.append(result)
        print(json.dumps(result), flush=True)

    results.sort(key=lambda item: float(item["fused_ms_median"]))
    report = {
        "status": "swept_resident_q5_base_kernel",
        "artifact": str(args.artifact),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
