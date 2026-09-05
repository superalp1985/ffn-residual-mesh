from __future__ import annotations

import time

import numpy as np
import torch

from resident_residual_cuda import launch_residual_tile
from resident_residual_format import ResidentArtifact
from resident_tile_cache import ResidentTileCache
from resident_tile_plan import TilePlan


class TiledResidentGateUp:
    """Exact gate/up + SwiGLU path using row-tiled resident residuals."""

    def __init__(
        self,
        artifact: ResidentArtifact,
        *,
        tile_rows: int = 1024,
        persistent: bool = False,
        device: str | torch.device = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TiledResidentGateUp requires CUDA")
        self.artifact = artifact
        self.device = torch.device(device)
        self.rows = int(artifact.projections["gate"]["rows"])
        self.cols = int(artifact.projections["gate"]["cols"])
        self.plan = TilePlan(
            rows=self.rows,
            tile_rows=tile_rows,
            projections=("gate", "up"),
        )
        arrays = {
            name: {
                "residual": artifact.arrays[name]["residual"],
                "alpha": artifact.arrays[name]["alpha"],
            }
            for name in ("gate", "up")
        }
        tile_bytes = self.plan.tile_bytes(
            cols=self.cols,
            alpha_cols=self.cols // 32,
        )
        self.cache = ResidentTileCache(
            arrays,
            plan=self.plan,
            vram_budget_bytes=(
                self.plan.total_bytes(cols=self.cols, alpha_cols=self.cols // 32)
                if persistent else tile_bytes * 2
            ),
            device=self.device,
        )
        if persistent:
            self.cache.cache.initialize_persistent(
                list(range(len(self.plan.tile_slices())))
            )
        self.stream = torch.cuda.Stream(device=self.device)
        self.device_x = torch.empty(self.cols, device=self.device, dtype=torch.float32)
        self._group_sums = np.empty(self.cols // 32, dtype=np.float64)

    def run(self, activation: np.ndarray, *, down=None) -> dict[str, object]:
        x = np.asarray(activation, dtype=np.float32)
        if x.shape != (self.cols,) or not np.isfinite(x).all():
            raise ValueError("finite one-token activation required")
        if down is not None and (down.cols != self.rows or down.rows != self.cols):
            raise ValueError("down projection dimensions must reverse gate/up dimensions")
        begin = time.perf_counter()
        grouped = x.astype(np.float64).reshape(-1, 32)
        self._group_sums[:] = grouped.sum(axis=1)
        base = {
            name: artifact_arrays["coefficient"] @ self._group_sums
            for name, artifact_arrays in self.artifact.arrays.items()
            if name in ("gate", "up")
        }
        cpu_base_ms = (time.perf_counter() - begin) * 1000
        with torch.cuda.stream(self.stream):
            self.device_x.copy_(torch.from_numpy(x), non_blocking=True)
        gate_residual = torch.empty(self.rows, device=self.device, dtype=torch.float32)
        up_residual = torch.empty_like(gate_residual)
        tile_kernel_ms = 0.0
        tile_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        down_begin = torch.cuda.Event(enable_timing=True) if down is not None else None
        down_end = torch.cuda.Event(enable_timing=True) if down is not None else None
        for tile, (start, stop) in enumerate(self.plan.tile_slices()):
            acquired = self.cache.acquire(tile)
            package = self.cache.package(tile)
            gate_out = torch.empty(stop - start, device=self.device, dtype=torch.float32)
            up_out = torch.empty_like(gate_out)
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.stream):
                begin_event.record()
                launch_residual_tile(
                    package["gate.residual"],
                    package["gate.alpha"],
                    self.device_x,
                    gate_out,
                    rows=stop - start,
                    cols=self.cols,
                )
                launch_residual_tile(
                    package["up.residual"],
                    package["up.alpha"],
                    self.device_x,
                    up_out,
                    rows=stop - start,
                    cols=self.cols,
                )
                gate_residual[start:stop].copy_(gate_out)
                up_residual[start:stop].copy_(up_out)
                end_event.record()
            tile_events.append((begin_event, end_event))
            self.cache.release(tile)
        with torch.cuda.stream(self.stream):
            gate_base = torch.from_numpy(base["gate"].astype(np.float32)).to(self.device)
            up_base = torch.from_numpy(base["up"].astype(np.float32)).to(self.device)
            gate = gate_residual + gate_base
            up = up_residual + up_base
            swiglu = gate * torch.sigmoid(gate) * up
            if down is not None:
                down_begin.record()
                down.launch(swiglu)
                down_end.record()
        self.stream.synchronize()
        tile_kernel_ms = float(sum(start.elapsed_time(end) for start, end in tile_events))
        result = {
            "gate": gate.cpu().numpy(),
            "up": up.cpu().numpy(),
            "swiglu": swiglu.cpu().numpy(),
            "cpu_base_ms": cpu_base_ms,
            "tile_kernel_ms": tile_kernel_ms,
            "wall_ms": (time.perf_counter() - begin) * 1000,
            "activation_h2d_bytes": self.cols * 4,
            "base_h2d_bytes": self.rows * 2 * 4,
            "weight_h2d_bytes": self.cache.traffic["weight_h2d_bytes"],
            "resident_weight_h2d_bytes": 0,
            "tile_count": len(self.plan.tile_slices()),
        }
        if down is not None:
            result["down"] = down.output.cpu().numpy()
            result["down_stream_ms"] = float(down_begin.elapsed_time(down_end))
        return result

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> TiledResidentGateUp:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
