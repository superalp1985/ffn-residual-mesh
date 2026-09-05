from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mib(value: float) -> float:
    return value / (2.0**20)


def gib(value: float) -> float:
    return value / (2.0**30)


def ms_for_bytes(value: float, gbps: float) -> float:
    return value / (gbps * 1.0e9) * 1000.0


def load_json(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def projection_residual_bytes(rows: int, input_dim: int, bits: int, group_size: int, alpha_bytes: int) -> int:
    code_bytes = input_dim * bits / 8.0
    groups = math.ceil(input_dim / group_size)
    return int(math.ceil(rows * (code_bytes + groups * alpha_bytes)))


def build_budget(args: argparse.Namespace) -> dict[str, object]:
    hidden = args.hidden
    ffn = args.ffn
    layers = args.layers
    tile_rows = args.tile_rows
    buffers = args.buffers
    elements = hidden * ffn
    tile_count = math.ceil(ffn / tile_rows)

    gate_up_residual = 2 * projection_residual_bytes(
        ffn, hidden, args.residual_bits, args.group_size, args.alpha_bytes
    )
    base_gate_up = 2 * ffn * args.base_bytes
    gate_up_q4 = 2 * elements * args.q4_bytes_per_weight
    down_q4 = elements * args.q4_bytes_per_weight
    down_residual = projection_residual_bytes(
        hidden, ffn, args.residual_bits, args.group_size, args.alpha_bytes
    )
    resident_down = elements * args.down_bytes

    # Conservative full-chain working set for two projections and two slots.
    row_residual = gate_up_residual / (2 * ffn)
    packet_buffers = 2 * buffers * tile_rows * row_residual
    base_buffers = 2 * buffers * tile_rows * args.base_bytes
    residual_outputs = 2 * buffers * tile_rows * 4
    merged_hidden = buffers * tile_rows * 4
    down_partial = tile_count * hidden * 4
    down_output = hidden * 4
    activations = hidden * (args.activation_bytes + args.scale_bytes / args.group_size)
    workspace = packet_buffers + base_buffers + residual_outputs + merged_hidden + down_partial + down_output + activations

    # The current prototype uses an 8 FLOP proxy per residual element, 2 FLOPs
    # per down MAC, and eight scalar operations per SwiGLU element.
    residual_flops = 8 * elements
    down_flops = 2 * elements
    merge_flops = 8 * ffn
    full_flops = residual_flops + down_flops + merge_flops

    resident_h2d = gate_up_residual + base_gate_up
    stream_down_h2d = resident_h2d + down_q4
    split_down_h2d = resident_h2d + down_residual
    peak_with_down = resident_down + workspace + args.runtime_reserve_bytes
    peak_with_all_down = resident_down * layers + workspace + args.runtime_reserve_bytes

    result: dict[str, object] = {
        "experiment": "parameterized_27b_class_ffn_budget",
        "assumption": {
            "hidden": hidden,
            "ffn": ffn,
            "layers": layers,
            "tile_rows": tile_rows,
            "tile_count": tile_count,
            "residual_bits": args.residual_bits,
            "group_size": args.group_size,
            "alpha_bytes_per_group": args.alpha_bytes,
            "q4_bytes_per_weight": args.q4_bytes_per_weight,
            "note": "Sizing model only; dimensions are not claimed to be an exact named 27B checkpoint.",
        },
        "bytes": {
            "matrix_elements_per_projection": elements,
            "gate_up_residual_bytes_per_token": int(gate_up_residual),
            "gate_up_residual_mib_per_token": mib(gate_up_residual),
            "gate_up_base_bytes_per_token": int(base_gate_up),
            "gate_up_q4_bytes_if_streamed": int(gate_up_q4),
            "down_q4_bytes_if_streamed": int(down_q4),
            "down_residual_bytes_if_split": int(down_residual),
            "resident_down_fp16_bytes": int(resident_down),
            "resident_down_fp16_mib": mib(resident_down),
            "full_chain_workspace_bytes": int(math.ceil(workspace)),
            "full_chain_workspace_mib": mib(workspace),
            "peak_with_one_resident_down_and_reserve_bytes": int(math.ceil(peak_with_down)),
            "peak_with_one_resident_down_and_reserve_mib": mib(peak_with_down),
            "peak_with_all_down_layers_and_reserve_bytes": int(math.ceil(peak_with_all_down)),
            "peak_with_all_down_layers_and_reserve_mib": mib(peak_with_all_down),
            "runtime_reserve_mib": mib(args.runtime_reserve_bytes),
        },
        "arithmetic": {
            "residual_flop_proxy": int(residual_flops),
            "down_flop_proxy": int(down_flops),
            "merge_flop_proxy": int(merge_flops),
            "full_chain_flop_proxy": int(full_flops),
            "residual_flop_proxy_per_residual_byte": residual_flops / max(gate_up_residual, 1.0),
            "full_flop_proxy_per_residual_byte": full_flops / max(gate_up_residual, 1.0),
            "full_flop_proxy_per_resident_h2d_byte": full_flops / max(resident_h2d, 1.0),
            "full_flop_proxy_per_stream_down_h2d_byte": full_flops / max(stream_down_h2d, 1.0),
            "full_flop_proxy_per_split_down_h2d_byte": full_flops / max(split_down_h2d, 1.0),
        },
        "pcie_lower_bound": {
            "effective_gbps": args.pcie_gbps,
            "resident_down_h2d_bytes_per_token": int(resident_h2d),
            "resident_down_h2d_mib_per_token": mib(resident_h2d),
            "resident_down_copy_ms": ms_for_bytes(resident_h2d, args.pcie_gbps),
            "stream_down_h2d_bytes_per_token": int(stream_down_h2d),
            "stream_down_h2d_mib_per_token": mib(stream_down_h2d),
            "stream_down_copy_ms": ms_for_bytes(stream_down_h2d, args.pcie_gbps),
            "split_down_h2d_bytes_per_token": int(split_down_h2d),
            "split_down_h2d_mib_per_token": mib(split_down_h2d),
            "split_down_copy_ms": ms_for_bytes(split_down_h2d, args.pcie_gbps),
            "resident_down_all_layers_h2d_mib_per_token": mib(resident_h2d * layers),
            "stream_down_all_layers_h2d_mib_per_token": mib(stream_down_h2d * layers),
        },
        "decision": {
            "one_layer_down_resident_fits_vram": peak_with_down <= args.vram_bytes,
            "all_down_layers_resident_fits_vram": peak_with_all_down <= args.vram_bytes,
            "resident_down_is_required_for_bandwidth_goal": True,
            "guarantee_scope": "single-layer budget and measured bridge only; not end-to-end 27B token/s",
        },
    }

    measured = load_json(args.measured_json)
    if measured:
        result["measured_bridge"] = measured
        overlap = measured.get("double_buffer_overlap", {})
        if isinstance(overlap, dict):
            copy_ms = float(overlap.get("copy_active_ms", 0.0))
            compute_ms = float(overlap.get("compute_active_ms", 0.0))
            critical_ms = float(overlap.get("critical_ms", 0.0))
            measured_flops = 125878272.0
            peak_tflops = args.peak_fp32_tflops
            result["measured_bridge_metrics"] = {
                "copy_effective_gbps": (9437184.0 / max(copy_ms, 1.0e-12)) / 1.0e6,
                "compute_proxy_tflops": measured_flops / max(compute_ms, 1.0e-12) / 1.0e9,
                "compute_proxy_vs_fp32_peak": (measured_flops / max(compute_ms, 1.0e-12) / 1.0e9) / max(peak_tflops, 1.0e-12),
                "compute_active_share_of_critical": compute_ms / max(critical_ms, 1.0e-12),
                "copy_active_share_of_critical": copy_ms / max(critical_ms, 1.0e-12),
                "copy_compute_overlap_fraction": max(copy_ms + compute_ms - critical_ms, 0.0) / max(min(copy_ms, compute_ms), 1.0e-12),
                "gap_or_wait_ms": max(critical_ms - max(copy_ms, compute_ms), 0.0),
                "scaled_full_compute_ms_at_same_proxy": full_flops / measured_flops * compute_ms,
                "scaled_resident_copy_ms_at_same_pcie": ms_for_bytes(resident_h2d, args.pcie_gbps),
                "scaled_stream_down_copy_ms_at_same_pcie": ms_for_bytes(stream_down_h2d, args.pcie_gbps),
                "scaled_split_down_copy_ms_at_same_pcie": ms_for_bytes(split_down_h2d, args.pcie_gbps),
            }
            scaled_compute = full_flops / measured_flops * compute_ms
            resident_copy = ms_for_bytes(resident_h2d, args.pcie_gbps)
            stream_copy = ms_for_bytes(stream_down_h2d, args.pcie_gbps)
            split_copy = ms_for_bytes(split_down_h2d, args.pcie_gbps)
            result["idealized_ffn_only_projection"] = {
                "scope": "per-layer lower bound from max(copy, compute); excludes attention, CPU base production, launch gaps and fallback",
                "resident_down": {
                    "copy_ms_per_layer": resident_copy,
                    "compute_ms_per_layer": scaled_compute,
                    "critical_ms_per_layer": max(resident_copy, scaled_compute),
                    "critical_ms_all_layers": max(resident_copy, scaled_compute) * layers,
                    "ffn_only_tokens_per_second_upper_bound": 1000.0 / max(max(resident_copy, scaled_compute) * layers, 1.0e-12),
                },
                "stream_down_q4": {
                    "copy_ms_per_layer": stream_copy,
                    "compute_ms_per_layer": scaled_compute,
                    "critical_ms_per_layer": max(stream_copy, scaled_compute),
                    "critical_ms_all_layers": max(stream_copy, scaled_compute) * layers,
                    "ffn_only_tokens_per_second_upper_bound": 1000.0 / max(max(stream_copy, scaled_compute) * layers, 1.0e-12),
                },
                "split_down_residual": {
                    "copy_ms_per_layer": split_copy,
                    "compute_ms_per_layer": scaled_compute,
                    "critical_ms_per_layer": max(split_copy, scaled_compute),
                    "critical_ms_all_layers": max(split_copy, scaled_compute) * layers,
                    "ffn_only_tokens_per_second_upper_bound": 1000.0 / max(max(split_copy, scaled_compute) * layers, 1.0e-12),
                },
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 27B-class FFN VRAM/H2D/compute budget")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--ffn", type=int, default=13824)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--tile-rows", type=int, default=2048)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--residual-bits", type=int, choices=(1, 2, 3, 4, 8), default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--alpha-bytes", type=int, default=4)
    parser.add_argument("--base-bytes", type=int, default=4)
    parser.add_argument("--down-bytes", type=int, default=2)
    parser.add_argument("--activation-bytes", type=int, default=1)
    parser.add_argument("--scale-bytes", type=int, default=4)
    parser.add_argument("--q4-bytes-per-weight", type=float, default=0.5625)
    parser.add_argument("--pcie-gbps", type=float, default=12.0)
    parser.add_argument("--vram-mib", type=float, default=8188.0)
    parser.add_argument("--runtime-reserve-mib", type=float, default=1024.0)
    parser.add_argument("--peak-fp32-tflops", type=float, default=28.6)
    parser.add_argument("--measured-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.vram_bytes = args.vram_mib * 2.0**20
    args.runtime_reserve_bytes = args.runtime_reserve_mib * 2.0**20
    result = build_budget(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
