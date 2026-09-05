from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mib(value: float) -> float:
    return value / (2.0**20)


def ms_for_bytes(value: float, gbps: float, bits_per_byte: float = 8.0) -> float:
    if gbps <= 0:
        return float("inf")
    return value * bits_per_byte / (gbps * 1.0e9) * 1000.0


def align_frame_count(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)


def latent_shape(width: int, height: int, frame_count: int) -> tuple[int, int, int, int, int]:
    aligned = align_frame_count(frame_count)
    latent_t = 2 if aligned <= 5 else ((aligned - 5) // 17) * 5 + 2
    latent_h = height // 16
    latent_w = width // 16
    video_rows = latent_t * (latent_h // 2) * (latent_w // 2)
    audio_t = round(aligned / 24.0 * 40.0)
    audio_rows = audio_t * 2
    return aligned, latent_t, video_rows, audio_t, audio_rows


def projection_weight_bytes(
    rows: int,
    cols: int,
    residual_bits: int,
    group_size: int,
    alpha_bytes: int,
) -> int:
    groups = math.ceil(cols / group_size)
    return int(math.ceil(rows * (cols * residual_bits / 8.0 + groups * alpha_bytes)))


def parse_counts(value: str) -> list[int]:
    counts = []
    for item in value.split(","):
        item = item.strip()
        if item:
            counts.append(max(1, int(item)))
    if not counts:
        raise ValueError("phone-counts must contain at least one positive integer")
    return counts


def efficiency(phone_count: int, parallel_loss: float) -> float:
    return max(0.05, 1.0 - parallel_loss * max(0, math.log2(phone_count)))


def simulate(args: argparse.Namespace) -> dict[str, object]:
    aligned, latent_t, video_rows, audio_t, audio_rows = latent_shape(
        args.width, args.height, args.frames
    )
    sequence_rows = args.text_rows + video_rows + audio_rows
    hidden = args.hidden
    ffn = args.ffn
    blocks = args.blocks

    fc1_rows = 2 * ffn
    fc1_elements = hidden * fc1_rows
    fc2_elements = hidden * ffn
    fc1_residual_bytes = projection_weight_bytes(
        fc1_rows, hidden, args.residual_bits, args.residual_group, args.alpha_bytes
    )
    fc2_residual_bytes = projection_weight_bytes(
        hidden, ffn, args.residual_bits, args.residual_group, args.alpha_bytes
    )
    residual_bytes_per_block = fc1_residual_bytes
    if args.split_down:
        residual_bytes_per_block += fc2_residual_bytes

    if args.descriptor_mode == "values":
        descriptor_bytes_per_step = sequence_rows * hidden * args.activation_bits / 8.0
    else:
        groups = math.ceil(hidden / args.descriptor_group)
        descriptor_bytes_per_step = sequence_rows * groups * args.descriptor_bytes_per_group

    if args.phone_return == "gate_up_exact":
        phone_return_bytes_per_step = sequence_rows * fc1_rows * args.base_output_bytes
        return_contract = "exact gate/up pre-activation return; required before SwiGLU"
        exact = True
    elif args.phone_return == "hidden_approx":
        phone_return_bytes_per_step = sequence_rows * hidden * args.base_output_bytes
        return_contract = "approximate post-FFN hidden return; not algebraically exact"
        exact = False
    else:
        phone_return_bytes_per_step = 0.0
        return_contract = "no base output return; not a complete FFN path"
        exact = False

    # The compiled base table is a tunable artifact-size assumption. It is not
    # the original dense weight stream and must be validated separately.
    base_table_bytes_per_block = fc1_elements * args.base_table_bytes_per_weight
    gpu_residual_h2d_ms_per_block = ms_for_bytes(
        residual_bytes_per_block, args.pcie_gbps, bits_per_byte=1.0
    )
    gpu_residual_h2d_ms = blocks * gpu_residual_h2d_ms_per_block
    gpu_ffn_ms = args.full_step_ms * args.ffn_share
    non_ffn_ms = args.full_step_ms * (1.0 - args.ffn_share)
    gpu_branch_ms = max(gpu_residual_h2d_ms, gpu_ffn_ms)

    network_input_ms = ms_for_bytes(
        descriptor_bytes_per_step, args.network_gbps, bits_per_byte=8.0
    )
    network_return_ms = ms_for_bytes(
        phone_return_bytes_per_step, args.network_gbps, bits_per_byte=8.0
    )
    network_latency_ms = args.network_latency_ms

    scenarios = []
    for phone_count in parse_counts(args.phone_counts):
        eff = efficiency(phone_count, args.parallel_loss)
        phone_compute_ms = ms_for_bytes(
            base_table_bytes_per_block * blocks,
            args.phone_memory_gbps * phone_count * eff,
            bits_per_byte=1.0,
        )
        if args.broadcast_mode == "serial":
            input_ms = network_input_ms * phone_count
        else:
            input_ms = network_input_ms
        phone_ready_ms = input_ms + phone_compute_ms + network_return_ms + network_latency_ms
        real_step_ms = non_ffn_ms + max(gpu_branch_ms, phone_ready_ms)
        reuse_steps = max(0, args.steps - args.tea_real_steps)
        sampling_ms = (
            args.tea_real_steps * real_step_ms
            + reuse_steps * args.tea_reuse_step_ms
        )
        total_ms = args.init_ms + sampling_ms + args.decode_ms
        baseline_ms = args.init_ms + args.steps * args.full_step_ms + args.decode_ms
        cached_no_cluster_ms = (
            args.init_ms
            + args.tea_real_steps * args.full_step_ms
            + reuse_steps * args.tea_reuse_step_ms
            + args.decode_ms
        )
        residual_h2d_bytes_total = residual_bytes_per_block * blocks * args.tea_real_steps
        descriptor_bytes_total = descriptor_bytes_per_step * args.tea_real_steps
        return_bytes_total = phone_return_bytes_per_step * args.tea_real_steps
        scenarios.append(
            {
                "phones": phone_count,
                "parallel_efficiency": eff,
                "descriptor_bytes_per_real_step": int(math.ceil(descriptor_bytes_per_step)),
                "phone_return_bytes_per_real_step": int(math.ceil(phone_return_bytes_per_step)),
                "residual_h2d_bytes_per_real_step": int(residual_bytes_per_block * blocks),
                "phone_base_compute_ms": phone_compute_ms,
                "network_input_ms": input_ms,
                "network_return_ms": network_return_ms,
                "phone_base_ready_ms": phone_ready_ms,
                "gpu_residual_branch_ms": gpu_branch_ms,
                "real_step_critical_ms": real_step_ms,
                "sampling_ms": sampling_ms,
                "total_ms": total_ms,
                "total_seconds": total_ms / 1000.0,
                "speedup_vs_no_cache_baseline": baseline_ms / max(total_ms, 1.0e-9),
                "speedup_vs_teacache_no_cluster": cached_no_cluster_ms / max(total_ms, 1.0e-9),
                "network_bytes_total": int(math.ceil(descriptor_bytes_total + return_bytes_total)),
                "residual_h2d_bytes_total": int(residual_h2d_bytes_total),
                "phone_branch_is_critical": phone_ready_ms > gpu_branch_ms,
                "exact_contract_compatible": exact,
                "phone_branch_hidden_under_gpu_branch": phone_ready_ms <= gpu_branch_ms,
                "feasible_for_exact_contract": exact and phone_ready_ms <= gpu_branch_ms,
            }
        )

    return {
        "experiment": "comfyui_minimax_h3_phone_ffn_cluster_simulation",
        "assumption": {
            "model": "MiniMax H3 local safetensors layout",
            "width": args.width,
            "height": args.height,
            "requested_frames": args.frames,
            "aligned_frames": aligned,
            "latent_t": latent_t,
            "video_rows": video_rows,
            "audio_t": audio_t,
            "audio_rows": audio_rows,
            "text_rows": args.text_rows,
            "packed_sequence_rows": sequence_rows,
            "hidden": hidden,
            "ffn": ffn,
            "blocks": blocks,
            "fc1_shape": [fc1_rows, hidden],
            "fc2_shape": [hidden, ffn],
            "steps": args.steps,
            "tea_real_steps": args.tea_real_steps,
            "tea_reuse_steps": max(0, args.steps - args.tea_real_steps),
            "full_step_ms_reference": args.full_step_ms,
            "ffn_share": args.ffn_share,
            "network_gbps": args.network_gbps,
            "pcie_gbps": args.pcie_gbps,
            "phone_memory_gbps": args.phone_memory_gbps,
            "descriptor_mode": args.descriptor_mode,
            "phone_return": args.phone_return,
            "broadcast_mode": args.broadcast_mode,
            "note": "Analytical model, not a real phone or ComfyUI benchmark.",
        },
        "ledger": {
            "fc1_residual_bytes_per_block": fc1_residual_bytes,
            "fc2_residual_bytes_per_block": fc2_residual_bytes,
            "residual_bytes_per_block": residual_bytes_per_block,
            "residual_h2d_mib_per_real_step": mib(residual_bytes_per_block * blocks),
            "descriptor_bytes_per_real_step": int(math.ceil(descriptor_bytes_per_step)),
            "descriptor_mib_per_real_step": mib(descriptor_bytes_per_step),
            "phone_return_bytes_per_real_step": int(math.ceil(phone_return_bytes_per_step)),
            "phone_return_mib_per_real_step": mib(phone_return_bytes_per_step),
            "base_table_mib_per_block": mib(base_table_bytes_per_block),
            "gpu_residual_h2d_ms_per_real_step": gpu_residual_h2d_ms,
            "gpu_ffn_compute_ms": gpu_ffn_ms,
            "non_ffn_ms": non_ffn_ms,
            "return_contract": return_contract,
            "exact_contract": exact,
        },
        "scenarios": scenarios,
        "interpretation": {
            "why_teacache_matters": "TeaCache reduces the number of real DiT forwards; phone and residual traffic are charged only on real steps.",
            "exact_path": "Exact phone offload must return gate/up pre-activations before SwiGLU, so sequence length makes network traffic dominant unless the transport is very fast or the base output is further structured.",
            "approx_path": "Returning hidden after a phone-side approximation is smaller but changes the algebra and must be reported as approximate with fallback.",
            "comfyui_integration": "Use a custom node or model wrapper for orchestration; keep the stock ComfyUI queue, TeaCache wrapper, SageAttention path and GPU fallback intact.",
            "first_prototype": "Loopback workers with persistent sockets and row-sharded fc1 tables; validate packet framing and deadlines before Android deployment.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate MiniMax H3 ComfyUI FFN residual split with phone/tablet workers"
    )
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--text-rows", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--ffn", type=int, default=14336)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tea-real-steps", type=int, default=8)
    parser.add_argument("--tea-reuse-step-ms", type=float, default=50.0)
    parser.add_argument("--full-step-ms", type=float, default=12330.0)
    parser.add_argument("--ffn-share", type=float, default=0.55)
    parser.add_argument("--residual-bits", type=int, choices=(1, 2, 3, 4, 8), default=2)
    parser.add_argument("--residual-group", type=int, default=32)
    parser.add_argument("--alpha-bytes", type=int, default=4)
    parser.add_argument("--pcie-gbps", type=float, default=12.0)
    parser.add_argument("--network-gbps", type=float, default=1.0)
    parser.add_argument("--network-latency-ms", type=float, default=0.35)
    parser.add_argument("--phone-memory-gbps", type=float, default=18.0)
    parser.add_argument("--phone-counts", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--parallel-loss", type=float, default=0.04)
    parser.add_argument("--base-table-bytes-per-weight", type=float, default=0.0625)
    parser.add_argument("--descriptor-mode", choices=("values", "group_sums"), default="group_sums")
    parser.add_argument("--descriptor-group", type=int, default=32)
    parser.add_argument("--descriptor-bytes-per-group", type=float, default=2.0)
    parser.add_argument("--activation-bits", type=int, default=2)
    parser.add_argument("--phone-return", choices=("gate_up_exact", "hidden_approx", "none"), default="gate_up_exact")
    parser.add_argument("--base-output-bytes", type=int, default=2)
    parser.add_argument("--split-down", action="store_true")
    parser.add_argument("--broadcast-mode", choices=("parallel", "serial"), default="parallel")
    parser.add_argument("--init-ms", type=float, default=10000.0)
    parser.add_argument("--decode-ms", type=float, default=5000.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.tea_real_steps < 1 or args.tea_real_steps > args.steps:
        parser.error("tea-real-steps must be in [1, steps]")
    if not 0.0 < args.ffn_share < 1.0:
        parser.error("ffn-share must be between 0 and 1")
    result = simulate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
