from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch

from evaluate_polynomial_base_residual import load_layer
from scan_q4k_hierarchical_code_split import QK_K, SUBGROUPS, load_q4k_codes


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    value = torch.linalg.vector_norm(pred - target, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target, dim=1), min=1e-6
    )
    return float(value.mean())


def host_gpu_replay(
    x: torch.Tensor,
    codes,
    alpha,
    beta,
    residual_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact Q4_K split as host base + GPU residual terms.

    q = 2**residual_bits * q_hi + q_lo
    host = alpha * (2**residual_bits * q_hi) + beta
    gpu_residual = alpha * q_lo

    The implementation runs both paths on the selected device only to verify
    the identity. Placement and transfer costs are recorded separately.
    """
    if residual_bits not in (1, 2, 3, 4):
        raise ValueError("residual_bits must be in [1, 4]")
    rows, blocks, _ = codes.shape
    xg = x.view(len(x), blocks, SUBGROUPS, 32)
    cg = torch.from_numpy(codes.reshape(rows, blocks, SUBGROUPS, 32)).to(x.device)
    alpha_t = torch.from_numpy(alpha).to(x.device)
    beta_t = torch.from_numpy(beta).to(x.device)
    multiplier = 1 << residual_bits
    q_hi = torch.div(cg, multiplier, rounding_mode="floor").to(torch.float32)
    q_lo = (cg % multiplier).to(torch.float32)
    host_code = torch.einsum("nbsi,rbsi->nrbs", xg, q_hi)
    residual_code = torch.einsum("nbsi,rbsi->nrbs", xg, q_lo)
    group_sum = xg.sum(dim=-1).unsqueeze(1)
    host = (alpha_t.unsqueeze(0) * multiplier * host_code + beta_t.unsqueeze(0) * group_sum).sum(dim=(2, 3))
    residual = (alpha_t.unsqueeze(0) * residual_code).sum(dim=(2, 3))
    return host, residual, host + residual


def ledger(rows: int, blocks: int, residual_bits: int, output_bytes: int = 2, activation_bytes: int = 2) -> dict:
    values = rows * blocks * QK_K
    q4_code_bytes = values // 2
    q4_metadata_bytes = rows * blocks * 16
    residual_code_bytes = math.ceil(values * residual_bits / 8)
    # Pre-expanded alpha is one fp16 scale per 32-value subgroup. Beta stays
    # with the host base; it never needs to occupy GPU memory.
    alpha_fp16_bytes = rows * blocks * SUBGROUPS * 2
    gpu_residual_bytes = residual_code_bytes + alpha_fp16_bytes
    full_q4_bytes = q4_metadata_bytes + q4_code_bytes
    host_activation_bytes = 2048 * activation_bytes
    base_output_bytes_gate_up = 2 * rows * output_bytes
    # This function is called once per projection. Keep the per-projection
    # values explicit, then expose gate+up pair totals so the caller cannot
    # accidentally report one projection as the complete FFN gate/up path.
    pair = 2
    return {
        "scope": "single_projection",
        "gpu_residual_code_bytes": residual_code_bytes,
        "gpu_alpha_scale_fp16_bytes": alpha_fp16_bytes,
        "gpu_residual_package_bytes": gpu_residual_bytes,
        "gpu_residual_package_mib": gpu_residual_bytes / (1024 * 1024),
        "original_q4k_projection_bytes": full_q4_bytes,
        "gpu_resident_reduction_vs_single_q4_projection": 1.0 - gpu_residual_bytes / full_q4_bytes,
        "h2d_residual_bytes_per_projection_weight_page": gpu_residual_bytes,
        "h2d_reduction_vs_single_q4_projection": 1.0 - gpu_residual_bytes / full_q4_bytes,
        "optional_activation_to_cpu_bytes_per_token": host_activation_bytes,
        "base_gate_up_output_to_gpu_bytes_per_token": base_output_bytes_gate_up,
        "small_vector_exchange_bytes_per_token": host_activation_bytes + base_output_bytes_gate_up,
        "host_high_code_bits": 4 - residual_bits,
        "host_high_code_bytes_per_projection": math.ceil(values * (4 - residual_bits) / 8),
        "host_beta_metadata_bytes_per_projection": q4_metadata_bytes,
        "gate_up_pair": {
            "gpu_residual_code_bytes": pair * residual_code_bytes,
            "gpu_alpha_scale_fp16_bytes": pair * alpha_fp16_bytes,
            "gpu_residual_package_bytes": pair * gpu_residual_bytes,
            "gpu_residual_package_mib": pair * gpu_residual_bytes / (1024 * 1024),
            "original_q4k_gate_up_bytes": pair * full_q4_bytes,
            "gpu_resident_reduction_vs_gate_up_q4": 1.0 - (pair * gpu_residual_bytes) / (pair * full_q4_bytes),
            "h2d_residual_bytes_per_gate_up_weight_page": pair * gpu_residual_bytes,
            "h2d_reduction_vs_gate_up_q4": 1.0 - (pair * gpu_residual_bytes) / (pair * full_q4_bytes),
            "host_high_code_bytes": pair * math.ceil(values * (4 - residual_bits) / 8),
            "host_beta_metadata_bytes": pair * q4_metadata_bytes,
        },
        "note": "GPU package assumes alpha scales are pre-expanded as fp16; host base includes high-code and beta metadata.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact Q4_K split with host base and GPU residual placement")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--residual-bits", default="1,2,3,4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout_x_np, _, _, _, _ = load_layer(args.holdout_root, args.layer)
    holdout_x = torch.from_numpy(holdout_x_np).to(device)
    rows_out = []
    projections = {}
    for projection in ("gate", "up"):
        codes, alpha, beta, weight_np, q4_bytes = load_q4k_codes(args.model, args.layer, projection)
        weight = torch.from_numpy(weight_np).to(device)
        teacher = holdout_x @ weight.T
        rows, blocks, _ = codes.shape
        projections[projection] = {"shape": list(codes.shape), "q4k_bytes": q4_bytes}
        for residual_bits in [int(value) for value in args.residual_bits.split(",") if value.strip()]:
            host_base, gpu_residual, merged = host_gpu_replay(
                holdout_x, codes, alpha, beta, residual_bits
            )
            row = {
                "projection": projection,
                "residual_bits_gpu": residual_bits,
                "host_base_bits": 4 - residual_bits,
                "exact_merged_rel_l2": rel_l2(merged, teacher),
                "exact_merged_abs_max": float((merged - teacher).abs().max()),
                "host_base_norm_mean": float(torch.linalg.vector_norm(host_base, dim=1).mean()),
                "gpu_residual_norm_mean": float(torch.linalg.vector_norm(gpu_residual, dim=1).mean()),
                "artifact": ledger(rows, blocks, residual_bits),
                "arithmetic": {
                    "host_formula": f"alpha * {1 << residual_bits} * dot(x, q_hi) + beta * sum(x)",
                    "gpu_formula": "alpha * dot(x, q_lo)",
                    "merge_formula": "host_base + gpu_residual",
                    "additional_code_ops": "one shift-scale and add per code; no approximation",
                },
            }
            rows_out.append(row)

    result = {
        "experiment": "q4k_host_base_gpu_residual",
        "formula": "q = 2^r*q_hi + q_lo; host computes high/base term, GPU computes low residual, merge by addition",
        "scope": "exact Q4_K code split with intended host-base/GPU-residual placement; no kernel benchmark",
        "model": str(args.model),
        "layer": args.layer,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "projections": projections,
        "rows": rows_out,
        "interpretation": {
            "placement": "Bulky high-code base and beta correction remain in host RAM/CPU; only low-code residual plus alpha scales are staged to GPU.",
            "primary_metrics": "GPU resident residual bytes, H2D residual bytes, and small activation/base-output exchange; host RAM size is secondary.",
            "exactness": "Every row reconstructs the original Q4_K projection to floating-point accumulation error because q_hi/q_lo are lossless code partitions.",
            "merge": "Host gate/up base vectors must be sent to GPU before SiLU, unless the runtime instead sends the GPU residual vectors back to CPU for merge.",
            "caveat": "The ledger counts a full residual package transfer. A tiled implementation can lower peak VRAM further, but must retain scale alignment and overlap transfers with residual dot products.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
