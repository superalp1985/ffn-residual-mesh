from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mib(value: float) -> float:
    return value / 2.0**20


def network_ms(value_bytes: float, gbps: float, latency_ms: float) -> float:
    if gbps <= 0:
        return float("inf")
    return latency_ms + value_bytes * 8.0 / (gbps * 1.0e9) * 1000.0


def model_weight_bytes(hidden: int, ffn: int, layers: int, weight_bits: int) -> int:
    # gate + up + down; ignores small scales and metadata so this is favorable
    # to the old full-weight-transfer design.
    values_per_layer = 3 * hidden * ffn
    return int(math.ceil(values_per_layer * weight_bits / 8.0) * layers)


def parse_counts(value: str) -> list[int]:
    counts = [max(1, int(item.strip())) for item in value.split(",") if item.strip()]
    if not counts:
        raise ValueError("phone-counts must contain at least one positive integer")
    return counts


def simulate(args: argparse.Namespace) -> dict[str, object]:
    rows = max(1, args.rows)
    hidden = max(1, args.hidden)
    ffn = max(1, args.ffn)
    layers = max(1, args.layers)
    batch = max(1, args.batch)

    fp16_boundary_bytes = rows * hidden * args.activation_bytes
    full_ffn_roundtrip_bytes = 2 * fp16_boundary_bytes
    all_boundaries_bytes = full_ffn_roundtrip_bytes * layers * batch
    full_layer_weight_bytes = model_weight_bytes(hidden, ffn, layers, args.weight_bits)
    direct_weight_bytes = full_layer_weight_bytes * args.steps

    non_ffn_ms = args.cpu_full_model_ms * max(0.0, 1.0 - args.ffn_share)
    old_weight_transfer_ms = network_ms(direct_weight_bytes, args.network_gbps, args.network_latency_ms)
    per_boundary_network_ms = network_ms(
        full_ffn_roundtrip_bytes * batch,
        args.network_gbps,
        args.network_latency_ms,
    )

    scenarios: list[dict[str, object]] = []
    for phones in parse_counts(args.phone_counts):
        # A batch-1 layer dependency cannot be parallelized by merely adding
        # more workers. More devices improve artifact capacity and throughput
        # for independent requests, but not this single-stream critical path.
        parallelism = min(phones, batch)
        phone_compute_ms = args.phone_ffn_ms * math.ceil(batch / parallelism)
        phone_step_ms = layers * (per_boundary_network_ms + phone_compute_ms)
        total_ms = non_ffn_ms + phone_step_ms
        scenarios.append(
            {
                "phones": phones,
                "single_stream_parallelism": 1 if batch == 1 else parallelism,
                "phone_weight_resident_mib_per_phone": mib(full_layer_weight_bytes / phones),
                "full_ffn_input_bytes_per_boundary": int(fp16_boundary_bytes * batch),
                "full_ffn_output_bytes_per_boundary": int(fp16_boundary_bytes * batch),
                "full_ffn_roundtrip_bytes_per_boundary": int(full_ffn_roundtrip_bytes * batch),
                "phone_boundary_network_ms": per_boundary_network_ms,
                "phone_ffn_compute_ms_per_boundary": phone_compute_ms,
                "phone_ffn_serial_ms": phone_step_ms,
                "non_ffn_center_ms": non_ffn_ms,
                "total_ms": total_ms,
                "tokens_per_second_single_stream": 1000.0 / max(total_ms, 1.0e-12),
                "phone_compute_is_critical": phone_compute_ms + per_boundary_network_ms > 0,
                "note": "For batch=1, layer dependencies remain serial; phone count does not divide latency.",
            }
        )

    return {
        "experiment": "full_ffn_phone_offload_simulation",
        "assumption": {
            "rows": rows,
            "hidden": hidden,
            "ffn": ffn,
            "layers": layers,
            "batch": batch,
            "steps": args.steps,
            "cpu_full_model_ms": args.cpu_full_model_ms,
            "ffn_share": args.ffn_share,
            "phone_ffn_ms_per_boundary": args.phone_ffn_ms,
            "network_gbps": args.network_gbps,
            "network_latency_ms": args.network_latency_ms,
            "weight_bits": args.weight_bits,
            "activation_bytes": args.activation_bytes,
            "note": "Analytical model, not a real Android benchmark.",
        },
        "ledger": {
            "fp16_activation_bytes_per_boundary": int(fp16_boundary_bytes),
            "full_ffn_roundtrip_bytes_per_boundary": int(full_ffn_roundtrip_bytes),
            "full_ffn_roundtrip_mib_per_boundary": mib(full_ffn_roundtrip_bytes),
            "all_boundaries_bytes_per_batch": int(all_boundaries_bytes),
            "all_boundaries_mib_per_batch": mib(all_boundaries_bytes),
            "compiled_ffn_weights_total_mib": mib(full_layer_weight_bytes),
            "old_full_weight_transfer_mib_for_steps": mib(direct_weight_bytes),
            "old_full_weight_transfer_ms": old_weight_transfer_ms,
            "center_non_ffn_ms": non_ffn_ms,
            "direct_cpu_reference_ms": args.cpu_full_model_ms,
        },
        "scenarios": scenarios,
        "interpretation": {
            "main_result": "Full FFN offload removes repeated runtime weight transfer; the link carries activation in/out instead.",
            "text_decode": "For small batch-1 text decode, activation traffic is tiny; phone compute and per-layer round trips dominate.",
            "video_forward": "For long-video packed sequences, every FFN boundary returns a large hidden tensor; layer-by-layer phone offload can become network-bound.",
            "recommended_topology": "Replicate or shard complete FFN layers across wired workers; use more phones for concurrent requests or storage, not as a claim of single-stream speedup.",
            "comparison_boundary": "This path is a full-FFN engineering baseline. It does not use the base/residual algorithm.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate complete FFN execution on wired phone/tablet workers")
    parser.add_argument("--rows", type=int, default=1, help="tokens/rows crossing each FFN boundary")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn", type=int, default=6144)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cpu-full-model-ms", type=float, default=23.3)
    parser.add_argument("--ffn-share", type=float, default=0.55)
    parser.add_argument("--phone-ffn-ms", type=float, default=2.0)
    parser.add_argument("--network-gbps", type=float, default=10.0)
    parser.add_argument("--network-latency-ms", type=float, default=0.35)
    parser.add_argument("--weight-bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--activation-bytes", type=int, choices=(2, 4), default=2)
    parser.add_argument("--phone-counts", default="1,2,4,8,16,32")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = simulate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
