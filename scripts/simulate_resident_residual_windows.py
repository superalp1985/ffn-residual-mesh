from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resident_window_scheduler import ResidentWindowScheduler


def layer_package_bytes(manifest: dict) -> dict[int, int]:
    packages: dict[int, int] = {}
    for tensor in manifest["tensors"]:
        packages.setdefault(int(tensor["layer"]), 0)
        packages[int(tensor["layer"])] += int(tensor["bytes"])
    return packages


def simulate(
    manifest: dict,
    *,
    vram_bytes: int,
    reserve_bytes: int,
    window_layers: int,
    persistent_layers: tuple[int, ...],
    sweeps: int,
) -> dict:
    packages = layer_package_bytes(manifest)
    scheduler = ResidentWindowScheduler(
        window_layers=window_layers,
        vram_budget_bytes=vram_bytes,
        reserve_bytes=reserve_bytes,
        layer_bytes=packages,
        persistent_layers=persistent_layers,
    )
    trace: list[dict] = []
    layers = sorted(packages)
    if not layers:
        raise ValueError("manifest has no FFN layers")
    for _ in range(sweeps):
        for first in range(0, len(layers), window_layers):
            window = layers[first:first + window_layers]
            scheduler.prefetch_window(window[0])
            for layer in window:
                scheduler.acquire_layer(layer)
                scheduler.begin_kernel(layer)
                scheduler.end_kernel(layer)
            scheduler.release_window(window[0])
            trace.append({
                "window": window,
                "resident_layers": scheduler.resident_layers(),
                "pending_layers": scheduler.pending_layers(),
                "traffic": dict(scheduler.traffic),
            })
    report = scheduler.budget_report()
    report.update({
        "status": "residency_simulation_not_runtime_measurement",
        "sweeps": sweeps,
        "window_layers": window_layers,
        "persistent_layers": list(persistent_layers),
        "trace": trace,
        "tokens_per_second": None,
        "limits": [
            "No CUDA copy is issued; traffic is a policy ledger.",
            "No GPU kernel, CPU base computation, attention, KV or deadline is measured.",
            "Package bytes are original manifest bytes; compiled residual artifacts must be supplied separately.",
        ],
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate persistent residual residency and swap policy")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--vram-gib", type=float, default=8)
    parser.add_argument("--reserve-gib", type=float, default=1)
    parser.add_argument("--window-layers", type=int, default=1)
    parser.add_argument("--persistent", default="")
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    persistent = tuple(int(item) for item in args.persistent.split(",") if item.strip())
    result = simulate(
        manifest,
        vram_bytes=int(args.vram_gib * 2**30),
        reserve_bytes=int(args.reserve_gib * 2**30),
        window_layers=args.window_layers,
        persistent_layers=persistent,
        sweeps=args.sweeps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
