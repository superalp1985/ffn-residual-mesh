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
from resident_residual_cuda import launch_residual_tile  # noqa: E402
from resident_tile_cache import ResidentTileCache  # noqa: E402
from resident_tile_plan import TilePlan  # noqa: E402


def benchmark(artifact_path: Path, tile_rows: int, *, repeats: int = 3) -> dict:
    with ResidentArtifact.open(artifact_path, verify_hashes=True) as artifact:
        arrays = {
            name: {
                "residual": artifact.arrays[name]["residual"],
                "alpha": artifact.arrays[name]["alpha"],
            }
            for name in ("gate", "up")
        }
        rows = int(artifact.projections["gate"]["rows"])
        cols = int(artifact.projections["gate"]["cols"])
        plan = TilePlan(rows=rows, tile_rows=tile_rows, projections=("gate", "up"))
        tile_bytes = plan.tile_bytes(cols=cols, alpha_cols=cols // 32)
        cache = ResidentTileCache(
            arrays,
            plan=plan,
            vram_budget_bytes=tile_bytes * 2 + 32 * 2**20,
        )
        x = np.random.default_rng(551).standard_normal(cols).astype(np.float32)
        device_x = torch.from_numpy(x).cuda()
        full_gate = torch.zeros(rows, device="cuda", dtype=torch.float32)
        full_up = torch.zeros(rows, device="cuda", dtype=torch.float32)
        copy_events: list[float] = []
        tile_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        start_wall = time.perf_counter()
        try:
            for tile, (start, stop) in enumerate(plan.tile_slices()):
                cache.wait_prefetch(tile, stream=torch.cuda.current_stream()) if tile in cache.pending_layers() else None
                if tile not in cache.device_layers():
                    cache.prefetch_async([tile])
                    cache.wait_prefetch(tile, stream=torch.cuda.current_stream())
                if tile + 1 < len(plan.tile_slices()):
                    cache.prefetch_async([tile + 1])
                package = cache.package(tile)
                tile_rows_actual = stop - start
                gate_out = torch.empty(tile_rows_actual, device="cuda")
                up_out = torch.empty(tile_rows_actual, device="cuda")
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(torch.cuda.current_stream()):
                    begin.record()
                    launch_residual_tile(
                        package["gate.residual"],
                        package["gate.alpha"],
                        device_x,
                        gate_out,
                        rows=tile_rows_actual,
                        cols=cols,
                    )
                    launch_residual_tile(
                        package["up.residual"],
                        package["up.alpha"],
                        device_x,
                        up_out,
                        rows=tile_rows_actual,
                        cols=cols,
                    )
                    end.record()
                    full_gate[start:stop].copy_(gate_out)
                    full_up[start:stop].copy_(up_out)
                tile_events.append((begin, end))
                cache.release(tile)
                if tile > 0:
                    if cache.try_finalize(tile - 1):
                        cache.evict(tile - 1)
            torch.cuda.synchronize()
            for tile in list(cache.pending_layers()):
                cache.wait_prefetch(tile)
            tile_times = [float(begin.elapsed_time(end)) for begin, end in tile_events]
            wall_ms = (time.perf_counter() - start_wall) * 1000
            gate_base, gate_residual = artifact.project_parts("gate", x)
            up_base, up_residual = artifact.project_parts("up", x)
            np.testing.assert_allclose(full_gate.cpu().numpy(), gate_residual, rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(full_up.cpu().numpy(), up_residual, rtol=1e-4, atol=1e-3)
            return {
                "status": "measured_tiled_residual_overlap",
                "artifact": str(artifact.directory),
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "rows": rows,
                "cols": cols,
                "tile_rows": tile_rows,
                "tile_count": len(plan.tile_slices()),
                "tile_bytes": tile_bytes,
                "wall_ms": wall_ms,
                "tile_kernel_ms_total": float(sum(tile_times)),
                "tile_kernel_ms_median": float(np.median(tile_times)),
                "weight_h2d_bytes": cache.traffic["weight_h2d_bytes"],
                "copy_ms_total": cache.traffic["copy_ms"],
                "resident_hit_weight_h2d_bytes": 0,
                "traffic": cache.traffic,
                "limits": [
                    "This is a real residual tile kernel, not a complete FFN layer.",
                    "The base term, SwiGLU, down projection, attention, KV, and generation are excluded.",
                    "The reported wall time includes tile launch and synchronization overhead.",
                ],
            }
        finally:
            cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure tiled residual kernel with next-tile prefetch")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--tile-rows", default="256,512,1024")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        str(tile): [benchmark(args.artifact, tile) for _ in range(args.repeats)]
        for tile in (int(value) for value in args.tile_rows.split(","))
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
