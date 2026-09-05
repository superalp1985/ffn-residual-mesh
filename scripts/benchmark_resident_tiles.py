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
from resident_tile_cache import ResidentTileCache  # noqa: E402
from resident_tile_plan import TilePlan  # noqa: E402


def benchmark(artifact_path: Path, tile_rows: int, *, repeats: int = 2) -> dict:
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
        alpha_cols = cols // 32
        plan = TilePlan(rows=rows, tile_rows=tile_rows, projections=("gate", "up"))
        tile_bytes = plan.tile_bytes(cols=cols, alpha_cols=alpha_cols)
        samples = []
        for _ in range(repeats):
            cache = ResidentTileCache(
                arrays,
                plan=plan,
                vram_budget_bytes=tile_bytes * 2,
            )
            begin = time.perf_counter()
            copy_ms = []
            try:
                for tile in range(len(plan.ranges())):
                    result = cache.acquire(tile)
                    copy_ms.append(float(result["copy_ms"]))
                    package = cache.package(tile)
                    # A small device-side consumer keeps the tile live while
                    # the next tile is selected; this is not an FFN kernel.
                    checksum = sum(value.sum() for value in package.values())
                    torch.cuda.current_stream().synchronize()
                    cache.release(tile)
                    if tile + 1 < len(plan.ranges()):
                        cache.evict(tile)
                samples.append({
                    "wall_ms": (time.perf_counter() - begin) * 1000,
                    "copy_ms_total": float(sum(copy_ms)),
                    "copy_ms_median": float(np.median(copy_ms)),
                    "copy_ms_max": float(max(copy_ms)),
                    "checksum": float(checksum.detach().cpu()),
                    "weight_h2d_bytes": cache.traffic["weight_h2d_bytes"],
                })
            finally:
                cache.close()
        return {
            "status": "measured_resident_tile_transfer_sweep",
            "artifact": str(artifact.directory),
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "rows": rows,
            "cols": cols,
            "tile_rows": tile_rows,
            "tile_count": len(plan.ranges()),
            "tile_bytes": tile_bytes,
            "full_gate_up_package_bytes": artifact.gate_up_bytes(),
            "samples": samples,
            "limits": [
                "This measures tile package transfer and a checksum consumer, not a tiled FFN kernel.",
                "All tiles are reloaded in sequence; no end-to-end generation is measured.",
                "Exactness of the underlying artifact is established by the resident artifact tests.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep resident residual tile transfer sizes")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--tile-rows", default="256,512,1024")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        str(tile): benchmark(args.artifact, tile, repeats=args.repeats)
        for tile in (int(value) for value in args.tile_rows.split(","))
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
