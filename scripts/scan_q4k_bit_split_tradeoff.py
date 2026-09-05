from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch

from evaluate_polynomial_base_residual import load_layer
from scan_q4k_hierarchical_code_split import Q4_K_BLOCK_BYTES, QK_K, load_q4k_codes


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    value = torch.linalg.vector_norm(pred - target, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target, dim=1), min=1e-6
    )
    return float(value.mean())


def split_replay(
    x: torch.Tensor,
    codes: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    resident_bits: int,
) -> torch.Tensor:
    """Replay a Q4_K projection after exact high/low code splitting.

    q = 2**resident_bits * q_hi + q_lo. The two terms are evaluated
    separately and merged before returning the projection. No weight or
    activation approximation is introduced.
    """
    if resident_bits < 0 or resident_bits > 4:
        raise ValueError("resident_bits must be in [0, 4]")
    rows, blocks, _ = codes.shape
    xg = x.view(len(x), blocks, QK_K // 32, 32)
    cg = torch.from_numpy(codes.reshape(rows, blocks, QK_K // 32, 32)).to(x.device)
    alpha_t = torch.from_numpy(alpha).to(x.device)
    beta_t = torch.from_numpy(beta).to(x.device)
    base = 1 << (4 - resident_bits)
    q_hi = torch.div(cg, base, rounding_mode="floor").to(torch.float32)
    q_lo = (cg % base).to(torch.float32)
    # [batch, blocks, subgroup, 32] x [rows, blocks, subgroup, 32]
    s = xg.sum(dim=-1)
    hi_dot = torch.einsum("nbsi,rbsi->nrbs", xg, q_hi)
    lo_dot = torch.einsum("nbsi,rbsi->nrbs", xg, q_lo)
    # Each Q4_K subgroup has its own affine scale/min pair.
    # Broadcast the per-output-row/per-subgroup affine parameters over the
    # token dimension, then reduce all Q4_K blocks and subgroups.
    result = (
        alpha_t.unsqueeze(0) * (base * hi_dot + lo_dot)
        + beta_t.unsqueeze(0) * s.unsqueeze(1)
    ).sum(dim=(2, 3))
    return result


def byte_ledger(rows: int, blocks: int, resident_bits: int) -> dict:
    values = rows * blocks * QK_K
    q4_code_bytes = values // 2
    metadata_bytes = rows * blocks * 16
    resident_code_bytes = math.ceil(values * resident_bits / 8)
    dynamic_code_bytes = math.ceil(values * (4 - resident_bits) / 8)
    return {
        "q4k_metadata_scale_min_bytes": metadata_bytes,
        "resident_high_code_bytes": resident_code_bytes,
        "resident_total_bytes": metadata_bytes + resident_code_bytes,
        "dynamic_low_code_bytes_per_full_projection": dynamic_code_bytes,
        "original_q4k_bytes": metadata_bytes + q4_code_bytes,
        "gpu_resident_reduction_vs_q4k": 1.0 - (metadata_bytes + resident_code_bytes) / (metadata_bytes + q4_code_bytes),
        "dynamic_h2d_reduction_vs_q4k": 1.0 - dynamic_code_bytes / (metadata_bytes + q4_code_bytes),
        "dynamic_h2d_reduction_vs_code_only": 1.0 - dynamic_code_bytes / q4_code_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact Q4_K high/low code split for GPU residency/H2D tradeoffs")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--resident-bits", default="1,2,3,4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout_x_np, _, _, _, _ = load_layer(args.holdout_root, args.layer)
    holdout_x = torch.from_numpy(holdout_x_np).to(device)
    rows_out = []
    projections = {}
    # This probe uses the layer-input calibration tensor, so it applies to
    # gate/up (input width 2048). Down takes the SwiGLU hidden width and is
    # intentionally left for a separate post-merge probe.
    for projection in ("gate", "up"):
        codes, alpha, beta, weight_np, q4_bytes = load_q4k_codes(args.model, args.layer, projection)
        weight = torch.from_numpy(weight_np).to(device)
        rows, blocks, _ = codes.shape
        # Keep this validation small enough for a laptop GPU while still
        # covering all output rows and every Q4_K block in the projection.
        teacher = holdout_x @ weight.T
        pinfo = {"shape": list(codes.shape), "q4k_bytes": q4_bytes}
        projections[projection] = pinfo
        for resident_bits in [int(value) for value in args.resident_bits.split(",") if value.strip()]:
            pred = split_replay(holdout_x, codes, alpha, beta, resident_bits)
            ledger = byte_ledger(rows, blocks, resident_bits)
            row = {
                "projection": projection,
                "resident_bits": resident_bits,
                "dynamic_bits": 4 - resident_bits,
                "exact_identity_holdout_rel_l2": rel_l2(pred, teacher),
                "exact_identity_holdout_abs_max": float((pred - teacher).abs().max()),
                "artifact": ledger,
                "arithmetic": {
                    "merge": f"q = {1 << (4 - resident_bits)} * q_hi + q_lo",
                    "base_dot_products_per_output": 1,
                    "residual_dot_products_per_output": 1,
                    "additional_integer_ops_per_code": 1,
                    "metadata_resident": True,
                },
            }
            rows_out.append(row)

    result = {
        "experiment": "q4k_exact_high_low_code_split_tradeoff",
        "formula": "w = alpha*(2^dynamic_bits*q_hi + q_lo) + beta; evaluate high and low code dots separately, then add",
        "scope": "exact Q4_K code-domain split; GPU residency and H2D ledger; no kernel benchmark",
        "model": str(args.model),
        "layer": args.layer,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "projections": projections,
        "rows": rows_out,
        "interpretation": {
            "primary_metric": "GPU resident bytes and dynamic H2D bytes; host RAM is intentionally absent from the primary ledger.",
            "exactness": "All tested rows are exact up to floating-point accumulation because q_hi and q_lo reconstruct every original 4-bit code.",
            "tradeoff": "More resident high bits reduce H2D but consume more VRAM. Two resident bits / two dynamic bits is the symmetric midpoint.",
            "system_requirement": "Only the low-code stream should be transferred at runtime; scale/min metadata and high-code stream must remain GPU-resident or be generated on GPU.",
            "caveat": "The H2D figures assume a full projection is streamed once per token. Real gains depend on layer reuse, batching, overlap, and whether the baseline streams Q4_K weights at all.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
