from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_preexpanded_sparse_cp import load_weights
from evaluate_polynomial_base_residual import load_layer


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    value = torch.linalg.vector_norm(pred - target, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target, dim=1), min=1e-6
    )
    return float(value.mean())


def right_basis(weight: torch.Tensor, rank: int) -> torch.Tensor:
    # The eigenspace of W^T W is the right singular subspace. Computing the
    # Gram matrix avoids materializing a full 12k x 2k SVD for the joint case.
    gram = weight.T @ weight
    _, vectors = torch.linalg.eigh(gram)
    # Return descending singular-energy order so a prefix is the best rank-r
    # subspace when the caller reuses one max-rank factorization.
    return torch.flip(vectors[:, -rank:], dims=(1,))


def symmetric_quantize(matrix: torch.Tensor, bits: int, group: int = 32) -> tuple[torch.Tensor, int]:
    rows, cols = matrix.shape
    qmax = 1 if bits <= 2 else (1 << (bits - 1)) - 1
    groups = (cols + group - 1) // group
    padded = torch.zeros((rows, groups * group), device=matrix.device, dtype=matrix.dtype)
    padded[:, :cols] = matrix
    blocks = padded.view(rows, groups, group)
    scales = blocks.abs().amax(dim=2, keepdim=True) / max(qmax, 1)
    scales = scales.clamp_min(1e-8)
    quant = torch.round(blocks / scales).clamp(-qmax, qmax)
    decoded = (quant * scales).view(rows, groups * group)[:, :cols]
    scale_bytes = rows * groups * 2
    return decoded, scale_bytes


def projection_from_split(
    x: torch.Tensor,
    basis: torch.Tensor,
    coefficient: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    alpha = x @ basis
    residual_x = x - alpha @ basis.T
    return alpha @ coefficient.T + residual_x @ residual.T


def ff_result(g: torch.Tensor, u: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(g) * u) @ down.T


def evaluate_family(
    train_x: torch.Tensor,
    holdout_x: torch.Tensor,
    train_weight_teacher: torch.Tensor,
    holdout_weight_teacher: torch.Tensor,
    capture_holdout: torch.Tensor,
    train_g_teacher: torch.Tensor,
    holdout_g_teacher: torch.Tensor,
    train_u_teacher: torch.Tensor,
    holdout_u_teacher: torch.Tensor,
    wg: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
    basis_g: torch.Tensor,
    basis_u: torch.Tensor,
    family: str,
    rank: int,
) -> dict:
    coeff_g = wg @ basis_g
    coeff_u = wu @ basis_u
    residual_g = wg - coeff_g @ basis_g.T
    residual_u = wu - coeff_u @ basis_u.T
    train_g = projection_from_split(train_x, basis_g, coeff_g, residual_g)
    holdout_g = projection_from_split(holdout_x, basis_g, coeff_g, residual_g)
    train_up = projection_from_split(train_x, basis_u, coeff_u, residual_u)
    holdout_up = projection_from_split(holdout_x, basis_u, coeff_u, residual_u)
    full_train = ff_result(train_g, train_up, wd)
    full_holdout = ff_result(holdout_g, holdout_up, wd)
    rows = {
        "family": family,
        "rank": rank,
        "gate_rel_l2_vs_teacher": rel_l2(train_g, train_g_teacher),
        "gate_holdout_rel_l2_vs_teacher": rel_l2(holdout_g, holdout_g_teacher),
        "up_rel_l2_vs_teacher": rel_l2(train_up, train_u_teacher),
        "up_holdout_rel_l2_vs_teacher": rel_l2(holdout_up, holdout_u_teacher),
        "ffn_rel_l2_vs_weight_teacher": rel_l2(full_train, train_weight_teacher),
        "ffn_holdout_rel_l2_vs_weight_teacher": rel_l2(full_holdout, holdout_weight_teacher),
        "ffn_holdout_rel_l2_vs_capture": rel_l2(full_holdout, capture_holdout),
        "residual_frobenius_fraction": float(
            torch.sqrt(residual_g.square().sum() + residual_u.square().sum())
            / torch.sqrt(wg.square().sum() + wu.square().sum())
        ),
        # For a shared basis, basis_g and basis_u reference the same U. Count
        # the tensor once; the prior row-level statistic double-counted it.
        "base_static_fp16_bytes": int(
            ((basis_g.numel() if family == "separate_right_bases" else basis_g.numel())
             + (basis_u.numel() if family == "separate_right_bases" else 0)
             + coeff_g.numel() + coeff_u.numel()) * 2
        ),
        "base_activation_coefficients_fp16_bytes_per_token": int((basis_g.shape[1] + basis_u.shape[1]) * 2),
        "base_projection_mac_per_token": int(coeff_g.numel() + coeff_u.numel()),
        "dense_residual_mac_per_token": int(residual_g.numel() + residual_u.numel()),
        "residual": {},
    }
    for bits in (2, 3, 4, 8):
        qg, scales_g = symmetric_quantize(residual_g, bits)
        qu, scales_u = symmetric_quantize(residual_u, bits)
        train_gq = projection_from_split(train_x, basis_g, coeff_g, qg)
        holdout_gq = projection_from_split(holdout_x, basis_g, coeff_g, qg)
        train_uq = projection_from_split(train_x, basis_u, coeff_u, qu)
        holdout_uq = projection_from_split(holdout_x, basis_u, coeff_u, qu)
        train_yq = ff_result(train_gq, train_uq, wd)
        holdout_yq = ff_result(holdout_gq, holdout_uq, wd)
        rows["residual"][str(bits)] = {
            "bits": bits,
            "ffn_holdout_rel_l2_vs_weight_teacher": rel_l2(holdout_yq, holdout_weight_teacher),
            "ffn_holdout_rel_l2_vs_capture": rel_l2(holdout_yq, capture_holdout),
            "gate_holdout_rel_l2_vs_teacher": rel_l2(holdout_gq, holdout_g_teacher),
            "up_holdout_rel_l2_vs_teacher": rel_l2(holdout_uq, holdout_u_teacher),
            "residual_table_bytes": int((wg.numel() + wu.numel()) * bits / 8 + scales_g + scales_u),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Math scan for shared gate/up right bases and quantized residuals")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--ranks", default="8,16,32,64,128,256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    train_x_np, _, _, _, train_capture = load_layer(args.calibration_root, args.layer)
    holdout_x_np, _, _, _, holdout_capture = load_layer(args.holdout_root, args.layer)
    (wg_np, wg_q4_bytes), (wu_np, wu_q4_bytes), (wd_np, wd_q4_bytes) = load_weights(args.model, args.layer)
    train_x = torch.from_numpy(train_x_np).to(device)
    holdout_x = torch.from_numpy(holdout_x_np).to(device)
    capture_train = torch.from_numpy(train_capture).to(device)
    capture_holdout = torch.from_numpy(holdout_capture).to(device)
    wg, wu, wd = (torch.from_numpy(value).to(device) for value in (wg_np, wu_np, wd_np))
    train_g_teacher, train_u_teacher = train_x @ wg.T, train_x @ wu.T
    holdout_g_teacher, holdout_u_teacher = holdout_x @ wg.T, holdout_x @ wu.T
    train_weight_teacher = ff_result(train_g_teacher, train_u_teacher, wd)
    holdout_weight_teacher = ff_result(holdout_g_teacher, holdout_u_teacher, wd)

    hidden = wg.shape[1]
    max_rank = max(int(value) for value in args.ranks.split(",") if value.strip())
    joint = right_basis(torch.cat((wg, wu), dim=0), max_rank)
    separate_g = right_basis(wg, max_rank)
    separate_u = right_basis(wu, max_rank)
    rows = []
    for requested in [int(value) for value in args.ranks.split(",") if value.strip()]:
        rank = min(requested, hidden)
        rows.append(
            evaluate_family(
                train_x,
                holdout_x,
                train_weight_teacher,
                holdout_weight_teacher,
                capture_holdout,
                train_g_teacher,
                holdout_g_teacher,
                train_u_teacher,
                holdout_u_teacher,
                wg,
                wu,
                wd,
                joint[:, :rank],
                joint[:, :rank],
                "shared_right_basis",
                rank,
            )
        )
        rows.append(
            evaluate_family(
                train_x,
                holdout_x,
                train_weight_teacher,
                holdout_weight_teacher,
                capture_holdout,
                train_g_teacher,
                holdout_g_teacher,
                train_u_teacher,
                holdout_u_teacher,
                wg,
                wu,
                wd,
                separate_g[:, :rank],
                separate_u[:, :rank],
                "separate_right_bases",
                rank,
            )
        )

    base_shared_fp16 = hidden * max_rank * 2 + 2 * wg.shape[0] * max_rank * 2
    base_separate_fp16 = 2 * hidden * max_rank * 2 + 2 * wg.shape[0] * max_rank * 2
    result = {
        "experiment": "shared_gate_up_right_basis_math_scan",
        "formula": "Wp = (Wp U) U^T + Wp(I-UU^T); Wp*x = (WpU)(U^T x) + Wp(I-UU^T)x",
        "scope": "mathematical decomposition only; no PCIe, paging, or kernel benchmark",
        "layer": args.layer,
        "device": str(device),
        "dimensions": {"hidden": int(hidden), "ffn": int(wg.shape[0])},
        "original_q4_bytes": int(wg_q4_bytes + wu_q4_bytes + wd_q4_bytes),
        "weight_teacher_capture_holdout_rel_l2": rel_l2(holdout_weight_teacher, capture_holdout),
        "artifact_at_max_rank": {
            "shared_basis_and_coefficients_fp16_bytes": int(base_shared_fp16),
            "separate_bases_and_coefficients_fp16_bytes": int(base_separate_fp16),
            "shared_basis_saving_bytes": int(base_separate_fp16 - base_shared_fp16),
            "down_q4_bytes": int(wd_q4_bytes),
        },
        "rows": rows,
        "interpretation": {
            "shared_basis": "one input-coordinate basis reused by gate and up; saves basis storage but may leave a larger residual",
            "separate_basis": "two independently optimal input-coordinate bases; usually lower residual but higher base storage",
            "residual_quantization": "symmetric per-row 32-value groups; scale bytes included, kernel cost not included",
            "not_yet_proven": "a dense low-bit residual is not automatically a traffic win; it must later be structured, sparse, or low-rank",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
