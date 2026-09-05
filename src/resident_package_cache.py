from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from resident_window_scheduler import ResidentWindowScheduler


class ResidentPackageCache:
    """CUDA-backed package cache driven by the residency policy.

    Copies use a dedicated stream and pinned host buffers. ``acquire`` waits
    for the copy event before returning, so this is a synchronous correctness
    path with real H2D accounting. A future pipeline can use the same package
    storage with a compute-stream event wait for exposed-overlap measurement.
    """

    def __init__(
        self,
        packages: Mapping[int, Mapping[str, np.ndarray]],
        *,
        scheduler: ResidentWindowScheduler,
        device: str | torch.device = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("ResidentPackageCache requires a CUDA device")
        if not packages:
            raise ValueError("at least one package is required")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("device must be CUDA")
        self.scheduler = scheduler
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.host: dict[int, dict[str, torch.Tensor]] = {}
        self._device: dict[int, dict[str, torch.Tensor]] = {}
        for layer, values in packages.items():
            layer = int(layer)
            if not values:
                raise ValueError(f"package {layer} is empty")
            self.host[layer] = {}
            for name, value in values.items():
                array = np.asarray(value)
                if not array.flags.c_contiguous or not array.flags.writeable:
                    array = np.array(array, copy=True, order="C")
                self.host[layer][str(name)] = torch.from_numpy(array).pin_memory()
            expected = sum(
                value.numel() * value.element_size()
                for value in self.host[layer].values()
            )
            try:
                declared = scheduler.layer_bytes(layer)
            except KeyError as exc:
                raise ValueError(f"package {layer} is not registered") from exc
            if expected != declared:
                raise ValueError(
                    f"package {layer} byte ledger mismatch: "
                    f"payload={expected}, declared={declared}"
                )
        self.traffic = {
            "weight_h2d_bytes": 0,
            "resident_hits": 0,
            "residual_misses": 0,
            "copy_ms": 0.0,
        }

    def acquire(self, layer: int) -> dict[str, object]:
        layer = int(layer)
        self._require_package(layer)
        if layer not in self.scheduler.resident_layers() and layer in self._device:
            self._device.pop(layer, None)
        if layer in self._device:
            transfer = self.scheduler.acquire_layer(layer)
            self.traffic["resident_hits"] += 1
            return {**transfer, "copy_ms": 0.0, "copy_event": None}
        transfer, copy_ms, event = self._upload(layer, activate=True)
        return {**transfer, "copy_ms": copy_ms, "copy_event": event}

    def prefetch(self, layers: list[int] | tuple[int, ...]) -> list[dict[str, object]]:
        results = []
        for layer in layers:
            layer = int(layer)
            self._require_package(layer)
            if layer in self._device:
                continue
            transfer, copy_ms, event = self._upload(layer, activate=False)
            results.append({**transfer, "copy_ms": copy_ms, "copy_event": event})
        return results

    def initialize_persistent(self, layers: list[int] | tuple[int, ...]) -> None:
        layers = tuple(int(layer) for layer in layers)
        self.scheduler.set_persistent_layers(layers, mark_resident=False)
        self.prefetch(list(layers))

    def _upload(
        self,
        layer: int,
        *,
        activate: bool,
    ) -> tuple[dict[str, object], float, torch.cuda.Event]:
        report = self.scheduler.prefetch_layers([layer])
        self._drop_evicted(report["evicted_layers"])
        self.scheduler.begin_prefetch(layer)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        device_values: dict[str, torch.Tensor] = {}
        with torch.cuda.stream(self.copy_stream):
            start.record()
            for name, host_value in self.host[layer].items():
                target = torch.empty_like(host_value, device=self.device)
                target.copy_(host_value, non_blocking=True)
                device_values[name] = target
            end.record()
        end.synchronize()
        copy_ms = float(start.elapsed_time(end))
        transfer = self.scheduler.complete_prefetch(layer)
        if activate:
            self.scheduler.activate_layer(layer)
        self._device[layer] = device_values
        self.traffic["weight_h2d_bytes"] += int(transfer["weight_h2d_bytes"])
        self.traffic["residual_misses"] += 1
        self.traffic["copy_ms"] += copy_ms
        return transfer, copy_ms, end

    def _drop_evicted(self, layers: object) -> None:
        if isinstance(layers, (list, tuple)):
            for layer in layers:
                self._device.pop(int(layer), None)

    def _require_package(self, layer: int) -> None:
        if layer not in self.host:
            raise KeyError(f"unknown package: {layer}")

    def package(self, layer: int) -> dict[str, torch.Tensor]:
        try:
            return self._device[int(layer)]
        except KeyError as exc:
            raise RuntimeError(f"package {layer} is not resident") from exc

    def release(self, layer: int) -> None:
        self.scheduler.release_layer(int(layer))

    def evict(self, layer: int) -> None:
        layer = int(layer)
        if layer not in self._device:
            self.scheduler.evict_layer(layer)
            return
        self.copy_stream.synchronize()
        self.scheduler.evict_layer(layer)
        del self._device[layer]

    def lease(self, layer: int, *, stream: torch.cuda.Stream | None = None):
        cache = self
        layer = int(layer)

        class _Lease:
            def __enter__(self):
                self.result = cache.acquire(layer)
                cache.scheduler.begin_kernel(layer)
                return cache.package(layer)

            def __exit__(self, exc_type, exc, tb):
                if stream is not None:
                    event = torch.cuda.Event()
                    stream.record_event(event)
                    event.synchronize()
                cache.scheduler.end_kernel(layer)
                cache.release(layer)
                return False

        return _Lease()

    def device_layers(self) -> list[int]:
        return sorted(self._device)

    def device_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for package in self._device.values()
            for tensor in package.values()
        )

    def close(self) -> None:
        self.copy_stream.synchronize()
        self._device.clear()

    def __enter__(self) -> ResidentPackageCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
