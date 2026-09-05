from __future__ import annotations

from collections.abc import Mapping


def select_persistent_layers(
    layer_bytes: Mapping[int, int],
    *,
    capacity_bytes: int,
    access_weights: Mapping[int, float] | None = None,
) -> dict[str, object]:
    """Select a deterministic high-hit persistent set.

    The selector uses weighted-hit density (access weight per resident byte),
    which scales to full-model byte budgets without a giant byte-indexed DP
    table. Ties are resolved by layer id, making manifests reproducible.
    """
    if capacity_bytes < 0:
        raise ValueError("capacity_bytes must be non-negative")
    normalized = {int(layer): int(size) for layer, size in layer_bytes.items()}
    if any(size <= 0 for size in normalized.values()):
        raise ValueError("layer package bytes must be positive")
    weights = {
        layer: float((access_weights or {}).get(layer, 1.0))
        for layer in normalized
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("access weights must be non-negative")
    ranked = sorted(
        normalized,
        key=lambda layer: (
            -(weights[layer] / normalized[layer] if normalized[layer] else 0.0),
            layer,
        ),
    )
    selected: list[int] = []
    used = 0
    for layer in ranked:
        size = normalized[layer]
        if used + size <= capacity_bytes:
            selected.append(layer)
            used += size
    selected.sort()
    unselected = sorted(set(normalized) - set(selected))
    return {
        "strategy": "weighted_hit_density_greedy",
        "capacity_bytes": int(capacity_bytes),
        "selected_layers": selected,
        "unselected_layers": unselected,
        "selected_bytes": used,
        "free_bytes": int(capacity_bytes - used),
        "weighted_hit_score": sum(weights[layer] for layer in selected),
        "total_weighted_hit_score": sum(weights.values()),
        "coverage": (
            sum(weights[layer] for layer in selected) / sum(weights.values())
            if sum(weights.values()) else 1.0
        ),
    }
