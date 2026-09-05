from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class ResidencyError(RuntimeError):
    """The requested residency transition would violate the VRAM contract."""


@dataclass
class _Layer:
    layer: int
    bytes: int
    persistent: bool = False
    resident: bool = False
    active: bool = False
    in_flight: int = 0
    copy_in_flight: bool = False


class ResidentWindowScheduler:
    """Capacity and traffic ledger for persistent residual residency.

    This class does not perform CUDA copies. It is the deterministic policy
    layer that a CUDA backend can drive: pending packages represent a
    double-buffer staging area, and `complete_prefetch` is the copy-complete
    callback.
    """

    def __init__(
        self,
        window_layers: int,
        vram_budget_bytes: int,
        *,
        reserve_bytes: int = 0,
        layer_bytes: Mapping[int, int] | None = None,
        persistent_layers: tuple[int, ...] | list[int] = (),
    ) -> None:
        if window_layers < 1:
            raise ValueError("window_layers must be positive")
        if vram_budget_bytes < 0 or reserve_bytes < 0:
            raise ValueError("memory budgets must be non-negative")
        if reserve_bytes > vram_budget_bytes:
            raise ValueError("reserve exceeds VRAM budget")
        self.window_layers = int(window_layers)
        self.vram_budget_bytes = int(vram_budget_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.capacity_bytes = self.vram_budget_bytes - self.reserve_bytes
        self._layers: dict[int, _Layer] = {}
        self._pending: set[int] = set()
        self._persistent: set[int] = set()
        self.traffic = {
            "weight_h2d_bytes": 0,
            "planned_h2d_bytes": 0,
            "resident_hits": 0,
            "residual_misses": 0,
            "evictions": 0,
            "blocked_prefetches": 0,
            "inflight_eviction_attempts": 0,
        }
        for layer, size in (layer_bytes or {}).items():
            self.register_layer(int(layer), int(size))
        if persistent_layers:
            self.set_persistent_layers(persistent_layers)

    def register_layer(self, layer: int, residual_bytes: int, down_bytes: int = 0) -> None:
        if layer < 0 or residual_bytes < 0 or down_bytes < 0:
            raise ValueError("invalid layer package")
        if layer in self._layers:
            raise ValueError(f"layer already registered: {layer}")
        total = int(residual_bytes) + int(down_bytes)
        if total == 0:
            raise ValueError("layer package must have positive bytes")
        self._layers[layer] = _Layer(layer=layer, bytes=total)

    def set_persistent_layers(
        self,
        layers: tuple[int, ...] | list[int],
        *,
        mark_resident: bool = True,
    ) -> None:
        """Declare already uploaded persistent packages (cold-start ledger)."""
        requested = {int(layer) for layer in layers}
        self._require_known(requested)
        if requested & self._pending:
            raise ResidencyError("pending packages cannot be declared resident")
        total = sum(self._layers[layer].bytes for layer in requested | self._persistent)
        if total > self.capacity_bytes:
            raise ResidencyError(
                f"persistent packages need {total} bytes, capacity is {self.capacity_bytes}"
            )
        extra = sum(self._layers[layer].bytes for layer in requested - self._resident())
        self._make_room(extra, protected=requested)
        for layer in requested:
            entry = self._layers[layer]
            entry.persistent = True
            if mark_resident:
                entry.resident = True
        self._persistent.update(requested)

    def prefetch_window(self, first_layer: int) -> dict[str, object]:
        if first_layer < 0:
            raise ValueError("first_layer must be non-negative")
        return self.prefetch_layers(self.window(first_layer))

    def window(self, first_layer: int) -> list[int]:
        self._require_known({first_layer})
        layers = sorted(self._layers)
        start = layers.index(first_layer)
        return layers[start:start + self.window_layers]

    def prefetch_layers(self, layers: list[int]) -> dict[str, object]:
        target = set(layers)
        self._require_known(target)
        needed = target - self._resident() - self._pending
        extra = sum(self._layers[layer].bytes for layer in needed)
        try:
            evicted = self._make_room(extra, protected=target)
        except ResidencyError:
            self.traffic["blocked_prefetches"] += 1
            raise
        self._pending.update(needed)
        planned = sum(self._layers[layer].bytes for layer in needed)
        self.traffic["planned_h2d_bytes"] += planned
        return {
            "target_layers": sorted(target),
            "pending_layers": sorted(self._pending),
            "resident_layers": self.resident_layers(),
            "planned_h2d_bytes": planned,
            "used_bytes": self.used_bytes(),
            "free_bytes": self.capacity_bytes - self.used_bytes(include_reserve=False),
            "evicted_layers": evicted,
        }

    def complete_prefetch(self, layer: int) -> dict[str, object]:
        result = self.publish_prefetch(layer)
        entry = self._layers[layer]
        entry.copy_in_flight = False
        size = int(result["weight_h2d_bytes"])
        self.traffic["weight_h2d_bytes"] += size
        self.traffic["residual_misses"] += 1
        return result

    def publish_prefetch(self, layer: int) -> dict[str, object]:
        self._require_known({layer})
        if layer not in self._pending:
            if self._layers[layer].resident:
                return {"layer": layer, "resident_hit": True, "weight_h2d_bytes": 0}
            raise ResidencyError(f"layer {layer} is not pending")
        self._pending.remove(layer)
        self._layers[layer].resident = True
        size = self._layers[layer].bytes
        return {"layer": layer, "resident_hit": False, "weight_h2d_bytes": size}

    def finish_prefetch(self, layer: int, transfer: dict[str, object] | None = None) -> dict[str, object]:
        self._require_known({layer})
        entry = self._layers[layer]
        if not entry.copy_in_flight:
            raise ResidencyError(f"layer {layer} has no in-flight copy")
        entry.copy_in_flight = False
        if transfer is None:
            transfer = {
                "layer": layer,
                "resident_hit": False,
                "weight_h2d_bytes": entry.bytes,
            }
        size = int(transfer["weight_h2d_bytes"])
        self.traffic["weight_h2d_bytes"] += size
        self.traffic["residual_misses"] += 1
        return transfer

    def begin_prefetch(self, layer: int) -> None:
        self._require_known({layer})
        if layer not in self._pending:
            raise ResidencyError(f"layer {layer} is not pending")
        if self._layers[layer].copy_in_flight:
            raise ResidencyError(f"layer {layer} already has an in-flight copy")
        self._layers[layer].copy_in_flight = True

    def acquire_layer(self, layer: int) -> dict[str, object]:
        self._require_known({layer})
        entry = self._layers[layer]
        if not entry.resident:
            if layer not in self._pending:
                self.prefetch_window(layer)
            if entry.copy_in_flight:
                raise ResidencyError(f"layer {layer} copy is still in flight")
            transfer = self.complete_prefetch(layer)
        else:
            self.traffic["resident_hits"] += 1
            transfer = {"layer": layer, "resident_hit": True, "weight_h2d_bytes": 0}
        entry.active = True
        return transfer

    def activate_layer(self, layer: int) -> None:
        """Mark a completed resident copy as active without charging traffic."""
        self._require_known({layer})
        entry = self._layers[layer]
        if not entry.resident or layer in self._pending:
            raise ResidencyError(f"layer {layer} is not ready for activation")
        entry.active = True

    def release_layer(self, layer: int) -> None:
        self._require_known({layer})
        self._layers[layer].active = False

    def release_window(self, first_layer: int) -> None:
        # Releasing a compute window does not evict packages. A later hit must
        # remain a zero-transfer operation until the policy explicitly trims.
        for layer in self.window(first_layer):
            self._layers[layer].active = False

    def begin_kernel(self, layer: int) -> None:
        self._require_known({layer})
        entry = self._layers[layer]
        if not entry.resident or not entry.active:
            raise ResidencyError(f"layer {layer} is not an acquired resident layer")
        entry.in_flight += 1

    def end_kernel(self, layer: int) -> None:
        self._require_known({layer})
        entry = self._layers[layer]
        if entry.in_flight < 1:
            raise ResidencyError(f"layer {layer} has no in-flight kernel")
        entry.in_flight -= 1

    def evict_layer(self, layer: int) -> None:
        self._require_known({layer})
        entry = self._layers[layer]
        if entry.persistent:
            raise ResidencyError(f"persistent layer {layer} cannot be evicted")
        if entry.in_flight:
            self.traffic["inflight_eviction_attempts"] += 1
            raise ResidencyError(f"layer {layer} has an in-flight kernel")
        if entry.copy_in_flight:
            self.traffic["inflight_eviction_attempts"] += 1
            raise ResidencyError(f"layer {layer} has an in-flight copy")
        if entry.active:
            raise ResidencyError(f"active layer {layer} cannot be evicted")
        if layer in self._pending:
            self._pending.remove(layer)
            return
        if entry.resident:
            entry.resident = False
            self.traffic["evictions"] += 1

    def trim_to_budget(self) -> list[int]:
        evicted: list[int] = []
        while self.used_bytes(include_reserve=False) > self.capacity_bytes:
            candidates = [
                entry for entry in self._layers.values()
                if entry.resident and not entry.persistent and not entry.active and not entry.in_flight
            ]
            if not candidates:
                raise ResidencyError("no safe transient package available for eviction")
            victim = max(candidates, key=lambda entry: entry.bytes)
            self.evict_layer(victim.layer)
            evicted.append(victim.layer)
        return evicted

    def active_layers(self) -> list[int]:
        return sorted(entry.layer for entry in self._layers.values() if entry.active)

    def layer_bytes(self, layer: int) -> int:
        self._require_known({int(layer)})
        return self._layers[int(layer)].bytes

    def pending_layers(self) -> list[int]:
        return sorted(self._pending)

    def resident_layers(self) -> list[int]:
        return sorted(self._resident())

    def used_bytes(self, *, include_reserve: bool = True) -> int:
        value = sum(self._layers[layer].bytes for layer in self._resident() | self._pending)
        return value + (self.reserve_bytes if include_reserve else 0)

    def budget_report(self) -> dict[str, object]:
        return {
            "vram_budget_bytes": self.vram_budget_bytes,
            "reserve_bytes": self.reserve_bytes,
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": sum(self._layers[layer].bytes for layer in self._resident()),
            "pending_bytes": sum(self._layers[layer].bytes for layer in self._pending),
            "used_bytes": self.used_bytes(),
            "free_bytes": max(0, self.capacity_bytes - self.used_bytes(include_reserve=False)),
            "persistent_layers": sorted(self._persistent),
            "active_layers": self.active_layers(),
            "pending_layers": self.pending_layers(),
            "resident_layers": self.resident_layers(),
            "traffic": dict(self.traffic),
        }

    def _resident(self) -> set[int]:
        return {layer for layer, entry in self._layers.items() if entry.resident}

    def _require_known(self, layers: set[int]) -> None:
        unknown = layers.difference(self._layers)
        if unknown:
            raise KeyError(f"unregistered layers: {sorted(unknown)}")

    def _make_room(self, extra: int, *, protected: set[int]) -> list[int]:
        if extra < 0:
            raise ValueError("extra bytes must be non-negative")
        required = self.used_bytes(include_reserve=False) + extra - self.capacity_bytes
        if required <= 0:
            return []
        candidates = [
            entry for entry in self._layers.values()
            if entry.resident
            and entry.layer not in protected
            and not entry.persistent
            and not entry.active
            and not entry.in_flight
            and not entry.copy_in_flight
        ]
        candidates.sort(key=lambda entry: entry.bytes, reverse=True)
        if sum(entry.bytes for entry in candidates) < required:
            raise ResidencyError(
                "prefetch would overcommit VRAM and no safe transient eviction exists"
            )
        reclaimed = 0
        evicted: list[int] = []
        for victim in candidates:
            self.evict_layer(victim.layer)
            reclaimed += victim.bytes
            evicted.append(victim.layer)
            if reclaimed >= required:
                return evicted
        return evicted
