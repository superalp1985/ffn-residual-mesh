from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mib(value: float) -> float:
    return value / 2.0**20


def ms_for_bytes(value: float, gbps: float) -> float:
    return value / max(gbps * 1.0e9, 1.0) * 1000.0


def ms_for_network_bytes(value: float, gigabits_per_second: float) -> float:
    return value / max(gigabits_per_second * 1.0e9 / 8.0, 1.0) * 1000.0


def table_bytes(input_dim: int, output_rows: int, block_size: int, states_per_value: int = 4) -> int:
    return (input_dim // block_size) * (states_per_value**block_size) * output_rows


def table_read_bytes(input_dim: int, output_rows: int, block_size: int, projections: int = 2) -> int:
    # One selected vector entry per input block and projection. The four radix
    # digits are already represented by the compiled artifact and are counted
    # in the state-table contract, not as four separate network transfers.
    return projections * (input_dim // block_size) * output_rows


def simulate(args: argparse.Namespace) -> dict[str, object]:
    h = args.hidden
    f = args.ffn
    layers = args.layers
    tile_rows = args.tile_rows
    elements = h * f
    table_total = 2 * table_bytes(h, f, args.block_size)
    table_read_total = table_read_bytes(h, f, args.block_size)
    tile_table_total = 2 * table_bytes(h, tile_rows, args.block_size)
    input_bytes = h + math.ceil(h / args.activation_group) * args.activation_scale_bytes
    base_output_bytes = 2 * f * args.base_bytes
    residual_h2d_bytes = 2 * (f * (h * args.residual_bits / 8.0 + math.ceil(h / args.residual_group) * args.alpha_bytes))
    gpu_copy_ms = ms_for_bytes(residual_h2d_bytes, args.pcie_gbps)
    gpu_compute_ms = args.gpu_compute_ms
    gpu_critical_ms = max(gpu_copy_ms, gpu_compute_ms)
    central_base_ms = ms_for_bytes(table_read_total, args.central_cpu_gbps) + args.central_overhead_ms

    scenarios: list[dict[str, object]] = []
    for phones in args.phone_counts:
        phones = max(1, phones)
        shard_table = table_total / phones
        shard_tile_table = tile_table_total / phones
        shard_read = table_read_total / phones
        l3_fit = shard_tile_table <= args.phone_l3_mib * 2.0**20
        cache_multiplier = args.l3_cache_multiplier if l3_fit else 1.0
        # Row-sharded workers run in parallel; efficiency loss covers fan-out,
        # uneven phone speeds and reduction bookkeeping.
        efficiency = max(0.1, 1.0 - args.parallel_loss * math.log2(phones))
        phone_compute_ms = ms_for_bytes(shard_read, args.phone_memory_gbps * cache_multiplier) / max(efficiency, 0.1)
        network_in_ms = args.network_latency_ms + ms_for_network_bytes(input_bytes, args.network_gbps)
        network_out_ms = args.network_latency_ms + ms_for_network_bytes(base_output_bytes, args.network_gbps)
        phone_ready_ms = network_in_ms + phone_compute_ms + network_out_ms
        layer_critical_ms = max(gpu_critical_ms, phone_ready_ms)
        total_ms = layer_critical_ms * layers
        scenarios.append({
            "phones": phones,
            "table_shard_mib_per_phone": mib(shard_table),
            "tile_table_working_set_mib_per_phone": mib(shard_tile_table),
            "active_table_read_mib_per_phone": mib(shard_read),
            "phone_l3_fit_for_tile": l3_fit,
            "cache_multiplier": cache_multiplier,
            "parallel_efficiency": efficiency,
            "phone_base_compute_ms": phone_compute_ms,
            "network_input_ms_per_layer": network_in_ms,
            "network_base_return_ms_per_layer": network_out_ms,
            "phone_base_ready_ms_per_layer": phone_ready_ms,
            "gpu_residual_critical_ms_per_layer": gpu_critical_ms,
            "layer_critical_ms": layer_critical_ms,
            "all_layers_critical_ms": total_ms,
            "idealized_decode_tokens_per_second": 1000.0 / max(total_ms, 1.0e-12),
            "network_base_bytes_per_token_all_layers_mib": mib(base_output_bytes * layers),
        })

    return {
        "experiment": "distributed_phone_ffn_base_cluster_simulation",
        "assumption": {
            "hidden": h,
            "ffn": f,
            "layers": layers,
            "block_size": args.block_size,
            "tile_rows": tile_rows,
            "phone_l3_mib": args.phone_l3_mib,
            "phone_memory_gbps": args.phone_memory_gbps,
            "network_gbps": args.network_gbps,
            "network_latency_ms": args.network_latency_ms,
            "parallel_model": "output-row sharding; phone workers run concurrently and return fp16 gate/up base rows",
            "note": "Analytical scheduler model, not a real Android benchmark; network_gbps is gigabits/s, phone_memory_gbps is decimal GB/s.",
        },
        "ledger": {
            "gate_up_compiled_table_total_mib": mib(table_total),
            "gate_up_runtime_table_read_mib_per_token": mib(table_read_total),
            "gate_up_tile_table_total_mib": mib(tile_table_total),
            "activation_descriptor_bytes_per_layer": int(input_bytes),
            "base_gate_up_return_bytes_per_layer": int(base_output_bytes),
            "base_gate_up_return_kib_per_layer": base_output_bytes / 1024.0,
            "residual_gate_up_h2d_mib_per_layer": mib(residual_h2d_bytes),
            "gpu_residual_copy_ms_per_layer": gpu_copy_ms,
            "gpu_full_chain_compute_ms_per_layer": gpu_compute_ms,
            "gpu_residual_critical_ms_per_layer": gpu_critical_ms,
            "central_cpu_base_ms_per_layer": central_base_ms,
            "central_cpu_gpu_critical_ms_per_layer": max(central_base_ms, gpu_critical_ms),
            "central_cpu_gpu_idealized_tokens_per_second": 1000.0 / max(layers * max(central_base_ms, gpu_critical_ms), 1.0e-12),
        },
        "scenarios": scenarios,
        "interpretation": {
            "what_cluster_solves": "aggregate CPU table bandwidth and per-worker cache pressure can scale with phone count",
            "what_cluster_does_not_solve": "batch-1 layer dependency, network latency, phone thermal throttling and down-policy bandwidth",
            "critical_path": "per layer is max(phone base ready, GPU residual critical) because the two branches can overlap",
            "strict_gpu_scope": "GPU still performs residual, merge, SwiGLU and down; phones perform the compiled main term",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a phone cluster as distributed FFN base workers")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--ffn", type=int, default=13824)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--block-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--tile-rows", type=int, default=2048)
    parser.add_argument("--phone-counts", default="1,2,4,8,16,32,64")
    parser.add_argument("--phone-l3-mib", type=float, default=8.0)
    parser.add_argument("--phone-memory-gbps", type=float, default=18.0)
    parser.add_argument("--l3-cache-multiplier", type=float, default=2.0)
    parser.add_argument("--parallel-loss", type=float, default=0.04)
    parser.add_argument("--network-gbps", type=float, default=1.0)
    parser.add_argument("--network-latency-ms", type=float, default=0.35)
    parser.add_argument("--pcie-gbps", type=float, default=12.0)
    parser.add_argument("--gpu-compute-ms", type=float, default=1.893)
    parser.add_argument("--central-cpu-gbps", type=float, default=6.3)
    parser.add_argument("--central-overhead-ms", type=float, default=0.2)
    parser.add_argument("--residual-bits", type=int, default=2)
    parser.add_argument("--residual-group", type=int, default=32)
    parser.add_argument("--alpha-bytes", type=int, default=4)
    parser.add_argument("--activation-group", type=int, default=32)
    parser.add_argument("--activation-scale-bytes", type=int, default=4)
    parser.add_argument("--base-bytes", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.phone_counts = [int(part) for part in args.phone_counts.split(",") if part.strip()]
    result = simulate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
