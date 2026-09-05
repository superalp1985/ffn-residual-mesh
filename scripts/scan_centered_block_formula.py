from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_polynomial_base_residual import load_layer
from evaluate_preexpanded_sparse_cp import load_weights


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    value = torch.linalg.vector_norm(pred - target, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target, dim=1), min=1e-6
    )
    return float(value.mean())


def centered_projection(x: torch.Tensor, weight: torch.Tensor, block_size: int) -> tuple[torch.Tensor, dict]:
    if x.shape[1] % block_size:
        raise ValueError("block size must divide hidden size")
    blocks = x.shape[1] // block_size
    xb = x.view(len(x), blocks, block_size)
    a = xb.mean(dim=2)
    b = xb - a.unsqueeze(2)
    wb = weight.view(weight.shape[0], blocks, block_size)
    c = wb.mean(dim=2)
    d = wb - c.unsqueeze(2)
    # The block contribution is B*a*c (not a*c): each block contains B
    # repeated mean products before the centered cross terms cancel.
    base = block_size * (a @ c.T)
    residual = torch.einsum("nbi,fbi->nf", b, d)
    return base + residual, {"a": a, "b": b, "c": c, "d": d}


def centered_projection_compensated(
    x: torch.Tensor, weight: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, dict]:
    """Exact identity with float32 block sums retained as explicit corrections.

    This version is useful for diagnosing GPU reduction error: the mean form
    should be mathematically exact, while the compensated form exposes the
    two cross terms instead of relying on zero-sum cancellation.
    """
    if x.shape[1] % block_size:
        raise ValueError("block size must divide hidden size")
    blocks = x.shape[1] // block_size
    xb = x.view(len(x), blocks, block_size)
    a = xb.mean(dim=2)
    b = xb - a.unsqueeze(2)
    wb = weight.view(weight.shape[0], blocks, block_size)
    c = wb.mean(dim=2)
    d = wb - c.unsqueeze(2)
    sum_b = b.sum(dim=2)
    sum_d = d.sum(dim=2)
    base = block_size * (a @ c.T)
    cross_x = a @ sum_d.T
    cross_w = sum_b @ c.T
    residual = torch.einsum("nbi,fbi->nf", b, d)
    return base + cross_x + cross_w + residual, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "sum_b": sum_b,
        "sum_d": sum_d,
    }


def quantize_residual(d: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = (1 << (bits - 1)) - 1
    qmax = max(qmax, 1)
    scale = d.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(d / scale).clamp(-qmax, qmax)
    return q * scale, scale.squeeze(2)


def ff_result(g: torch.Tensor, u: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(g) * u) @ down.T


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan exact centered block base + residual FFN formula")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--bits", default="2,3,4,8")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    train_x_np, _, _, _, train_capture = load_layer(args.calibration_root, args.layer)
    holdout_x_np, _, _, _, holdout_capture = load_layer(args.holdout_root, args.layer)
    (wg_np, wg_q4_bytes), (wu_np, wu_q4_bytes), (wd_np, wd_q4_bytes) = load_weights(args.model, args.layer)
    train_x = torch.from_numpy(train_x_np).to(device)
    holdout_x = torch.from_numpy(holdout_x_np).to(device)
    capture_holdout = torch.from_numpy(holdout_capture).to(device)
    wg, wu, wd = (torch.from_numpy(value).to(device) for value in (wg_np, wu_np, wd_np))

    hidden = int(wg.shape[1])
    ffn = int(wg.shape[0])
    blocks = hidden // args.block_size
    train_g_teacher, train_u_teacher = train_x @ wg.T, train_x @ wu.T
    holdout_g_teacher, holdout_u_teacher = holdout_x @ wg.T, holdout_x @ wu.T
    train_weight_teacher = ff_result(train_g_teacher, train_u_teacher, wd)
    holdout_weight_teacher = ff_result(holdout_g_teacher, holdout_u_teacher, wd)

    train_g_exact, train_g_parts = centered_projection(train_x, wg, args.block_size)
    holdout_g_exact, holdout_g_parts = centered_projection(holdout_x, wg, args.block_size)
    train_u_exact, train_u_parts = centered_projection(train_x, wu, args.block_size)
    holdout_u_exact, holdout_u_parts = centered_projection(holdout_x, wu, args.block_size)
    train_g_comp, _ = centered_projection_compensated(train_x, wg, args.block_size)
    holdout_g_comp, _ = centered_projection_compensated(holdout_x, wg, args.block_size)
    train_u_comp, _ = centered_projection_compensated(train_x, wu, args.block_size)
    holdout_u_comp, _ = centered_projection_compensated(holdout_x, wu, args.block_size)
    exact_identity = {
        "gate_train_max_abs": float((train_g_exact - train_g_teacher).abs().max()),
        "gate_holdout_max_abs": float((holdout_g_exact - holdout_g_teacher).abs().max()),
        "up_train_max_abs": float((train_u_exact - train_u_teacher).abs().max()),
        "up_holdout_max_abs": float((holdout_u_exact - holdout_u_teacher).abs().max()),
        "compensated_gate_train_max_abs": float((train_g_comp - train_g_teacher).abs().max()),
        "compensated_gate_holdout_max_abs": float((holdout_g_comp - holdout_g_teacher).abs().max()),
        "compensated_up_train_max_abs": float((train_u_comp - train_u_teacher).abs().max()),
        "compensated_up_holdout_max_abs": float((holdout_u_comp - holdout_u_teacher).abs().max()),
    }

    rows = []
    for bits in [int(value) for value in args.bits.split(",") if value.strip()]:
        gate_dq, gate_scale = quantize_residual(train_g_parts["d"], bits)
        up_dq, up_scale = quantize_residual(train_u_parts["d"], bits)
        gate_c16 = train_g_parts["c"].to(torch.float16).to(torch.float32)
        up_c16 = train_u_parts["c"].to(torch.float16).to(torch.float32)

        def replay(x_parts: dict, c: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
            base = args.block_size * (x_parts["a"] @ c.T)
            residual = torch.einsum("nbi,fbi->nf", x_parts["b"], dq)
            return base + residual

        train_g = replay(train_g_parts, gate_c16, gate_dq)
        holdout_g = replay(holdout_g_parts, gate_c16, gate_dq)
        train_u = replay(train_u_parts, up_c16, up_dq)
        holdout_u = replay(holdout_u_parts, up_c16, up_dq)
        train_y = ff_result(train_g, train_u, wd)
        holdout_y = ff_result(holdout_g, holdout_u, wd)

        base_bytes = 2 * ffn * blocks * 2  # fp16 c_g/c_u
        residual_code_bytes = int(2 * ffn * hidden * bits / 8)
        residual_scale_bytes = 2 * ffn * blocks * 2
        total_gate_up_bytes = base_bytes + residual_code_bytes + residual_scale_bytes
        rows.append(
            {
                "bits": bits,
                "gate_holdout_rel_l2_vs_teacher": rel_l2(holdout_g, holdout_g_teacher),
                "up_holdout_rel_l2_vs_teacher": rel_l2(holdout_u, holdout_u_teacher),
                "ffn_holdout_rel_l2_vs_weight_teacher": rel_l2(holdout_y, holdout_weight_teacher),
                "ffn_holdout_rel_l2_vs_capture": rel_l2(holdout_y, capture_holdout),
                "residual_frobenius_fraction": float(
                    torch.sqrt(train_g_parts["d"].square().sum() + train_u_parts["d"].square().sum())
                    / torch.sqrt(wg.square().sum() + wu.square().sum())
                ),
                "residual_rmse_float": float(
                    torch.sqrt(
                        ((gate_dq - train_g_parts["d"]) ** 2).mean()
                        + ((up_dq - train_u_parts["d"]) ** 2).mean()
                    )
                ),
                "artifact": {
                    "gate_up_base_fp16_bytes": int(base_bytes),
                    "residual_code_bytes": residual_code_bytes,
                    "residual_scale_fp16_bytes": int(residual_scale_bytes),
                    "gate_up_total_bytes": int(total_gate_up_bytes),
                    "original_gate_up_q4_bytes": int(wg_q4_bytes + wu_q4_bytes),
                    "reduction_vs_original_gate_up": float(1.0 - total_gate_up_bytes / (wg_q4_bytes + wu_q4_bytes)),
                },
                "arithmetic": {
                    "base_scalar_macs_per_token_gate_up": int(2 * ffn * blocks),
                    "residual_macs_per_token_gate_up": int(2 * ffn * hidden),
                    "activation_block_means_shared_between_gate_up": True,
                },
            }
        )

    result = {
        "experiment": "centered_block_base_residual_formula",
        "formula": "x_i=a+b_i; w_i=c+d_i; dot= B*a*c + a*sum(d) + c*sum(b) + b*d; centered means make cross sums zero",
        "scope": "mathematical identity and residual quantization only; no transfer or kernel benchmark",
        "layer": args.layer,
        "block_size": args.block_size,
        "blocks": blocks,
        "dimensions": {"hidden": hidden, "ffn": ffn},
        "device": str(device),
        "exact_identity": exact_identity,
        "capture_holdout_teacher_rel_l2": rel_l2(holdout_weight_teacher, capture_holdout),
        "rows": rows,
        "interpretation": {
            "base": "one scalar per input block and output row; gate/up reuse the same activation block means",
            "residual": "within-block centered weight values; only b dot d is dense",
            "warning": "a low-bit dense residual is a storage candidate, not yet a proven runtime traffic win",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
