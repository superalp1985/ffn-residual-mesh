from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from resident_package_cache import ResidentPackageCache
from resident_tile_plan import TilePlan
from resident_window_scheduler import ResidentWindowScheduler


class ResidentTileCache:
    """Tile-granular adapter for residual packages.

    A tile contains row slices for every projection and is independently
    resident, prefetched, leased, and accounted. The adapter intentionally
    leaves kernel launch policy to the caller.
    """

    def __init__(
        self,
        arrays: Mapping[str, Mapping[str, np.ndarray]],
        *,
        plan: TilePlan,
        device: str | torch.device = "cuda",
        vram_budget_bytes: int | None = None,
        reserve_bytes: int = 0,
    ) -> None:
        if tuple(plan.projections) != tuple(arrays):
            raise ValueError("plan projections must match array projections")
        self.plan = plan
        self._keys: dict[int, tuple[int, int]] = {}
        packages: dict[int, dict[str, np.ndarray]] = {}
        layer_bytes: dict[int, int] = {}
        for tile_index, (start, stop) in enumerate(plan.tile_slices()):
            package: dict[str, np.ndarray] = {}
            for projection in plan.projections:
                values = arrays[projection]
                for name in ("residual", "alpha"):
                    if name not in values:
                        raise ValueError(f"{projection} missing {name}")
                    package[f"{projection}.{name}"] = np.asarray(values[name][start:stop])
            packages[tile_index] = package
            layer_bytes[tile_index] = sum(value.nbytes for value in package.values())
            self._keys[tile_index] = (start, stop)
        scheduler = ResidentWindowScheduler(
            window_layers=1,
            vram_budget_bytes=(
                sum(layer_bytes.values())
                if vram_budget_bytes is None
                else int(vram_budget_bytes)
            ),
            reserve_bytes=int(reserve_bytes),
            layer_bytes=layer_bytes,
        )
        self.cache = ResidentPackageCache(packages, scheduler=scheduler, device=device)

    def acquire(self, tile: int) -> dict[str, object]:
        return self.cache.acquire(int(tile))

    def prefetch_async(self, tiles: list[int] | tuple[int, ...]) -> list[dict[str, object]]:
        return self.cache.prefetch_async(tiles)

    def wait_prefetch(
        self,
        tile: int,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.cuda.Event:
        return self.cache.wait_prefetch(int(tile), stream=stream)

    def package(self, tile: int) -> dict[str, torch.Tensor]:
        return self.cache.package(int(tile))

    def tile_range(self, tile: int) -> tuple[int, int]:
        try:
            return self._keys[int(tile)]
        except KeyError as exc:
            raise KeyError(f"unknown tile: {tile}") from exc

    def release(self, tile: int) -> None:
        self.cache.release(int(tile))

    def evict(self, tile: int) -> None:
        self.cache.evict(int(tile))

    def pending_layers(self) -> list[int]:
        return self.cache.pending_layers()

    def device_layers(self) -> list[int]:
        return self.cache.device_layers()

    def try_finalize(self, tile: int) -> bool:
        return self.cache.try_finalize(int(tile))

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> ResidentTileCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def traffic(self) -> dict[str, object]:
        return self.cache.traffic
