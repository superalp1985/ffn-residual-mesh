from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persistent_layer_selector import select_persistent_layers  # noqa: E402


def layer_bytes(manifest: dict, *, compiled_multiplier: float = 1.0) -> dict[int, int]:
    grouped: dict[int, dict[str, int]] = {}
    for tensor in manifest["tensors"]:
        layer = int(tensor["layer"])
        projection = str(tensor["projection"])
        grouped.setdefault(layer, {})[projection] = int(tensor["bytes"])
    compiled = {int(layer) for layer in manifest.get("compilable_gate_up_layers", [])}
    result: dict[int, int] = {}
    for layer, values in grouped.items():
        total = sum(values.values())
        if layer in compiled:
            gate_up = values.get("ffn_gate", 0) + values.get("ffn_up", 0)
            down = values.get("ffn_down", 0)
            total = int(gate_up * compiled_multiplier + down)
        result[layer] = total
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select persistent Qwen3.8 FFN layers by weighted hit density")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--vram-gib", type=float, default=8.0)
    parser.add_argument("--reserve-gib", type=float, default=1.0)
    parser.add_argument("--compiled-multiplier", type=float, default=1.176)
    parser.add_argument("--window-layers", type=int, default=4)
    parser.add_argument("--trace", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    packages = layer_bytes(manifest, compiled_multiplier=args.compiled_multiplier)
    trace = [int(value) for value in args.trace.split(",") if value.strip()]
    weights = {layer: trace.count(layer) for layer in packages} if trace else None
    capacity = int((args.vram_gib - args.reserve_gib) * 2**30)
    if args.window_layers < 1:
        raise ValueError("window-layers must be positive")
    ordered_layers = sorted(packages)
    window_working = 0
    for index in range(len(ordered_layers)):
        active = ordered_layers[index:index + args.window_layers]
        pending = ordered_layers[index + args.window_layers:index + 2 * args.window_layers]
        window_working = max(
            window_working,
            sum(packages[layer] for layer in active + pending),
        )
    persistent_capacity = max(0, capacity - window_working)
    result = select_persistent_layers(
        packages,
        capacity_bytes=persistent_capacity,
        access_weights=weights,
    )
    result.update({
        "model_id": manifest.get("model_id"),
        "capacity_basis": "VRAM minus reserve",
        "vram_gib": args.vram_gib,
        "reserve_gib": args.reserve_gib,
        "window_layers": args.window_layers,
        "window_working_set_bytes": window_working,
        "persistent_capacity_bytes": persistent_capacity,
        "compiled_gate_up_layers": sorted(int(layer) for layer in manifest.get("compilable_gate_up_layers", [])),
        "layer_package_bytes": packages,
        "access_trace": trace,
        "limits": [
            "Compiled multiplier is an estimate until every layer has an artifact.",
            "Uncompiled layers use original GGUF FFN bytes as exact fallback placeholders.",
            "Selection maximizes weighted byte density heuristically; it is not a throughput claim.",
        ],
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
