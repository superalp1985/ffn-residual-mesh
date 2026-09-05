from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch

from evaluate_polynomial_base_residual import load_layer
from scan_q4k_hierarchical_code_split import Q4_K_BLOCK_BYTES, QK_K, SUBGROUP, SUBGROUPS, load_q4k_codes


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    value = torch.linalg.vector_norm(pred - target, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target, dim=1), min=1e-6
    )
    return float(value.mean())


def choose_base(values: np.ndarray, mode: str, bits: int) -> np.ndarray:
    if mode == "mean":
        return np.rint(values.mean(axis=-1)).astype(np.int16)
    if mode == "midrange":
        return np.rint((values.min(axis=-1) + values.max(axis=-1)) / 2.0).astype(np.int16)
    if mode != "min_outliers":
        raise ValueError(f"unknown base mode: {mode}")
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    flat = values.reshape(-1, values.shape[-1])
    candidates = np.arange(16, dtype=np.int16)
    best = np.empty(flat.shape[0], dtype=np.int16)
    for start in range(0, len(flat), 100_000):
        part = flat[start : start + 100_000].astype(np.int16)
        residual = part[:, None, :] - candidates[None, :, None]
        clipped = np.clip(residual, qmin, qmax)
        correction = residual - clipped
        count = np.count_nonzero(correction, axis=2)
        magnitude = np.abs(correction).sum(axis=2)
        energy = np.square(residual).sum(axis=2)
        order = np.lexsort((np.broadcast_to(candidates, count.shape), energy, magnitude, count), axis=1)
        best[start : start + len(part)] = candidates[order[:, 0]]
    return best.reshape(values.shape[:-1])


def signed_bits(values: np.ndarray) -> int:
    minimum, maximum = int(values.min()), int(values.max())
    for bits in range(1, 9):
        if minimum >= -(1 << (bits - 1)) and maximum <= (1 << (bits - 1)) - 1:
            return bits
    return 8


def evaluate(
    x: torch.Tensor,
    codes: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    weight: np.ndarray,
    subblock: int,
    base_mode: str,
    bits_list: list[int],
    device: torch.device,
) -> dict:
    rows, blocks, _ = codes.shape
    if SUBGROUP % subblock:
        raise ValueError("subblock must divide 32")
    subblocks = SUBGROUP // subblock
    group_count = blocks * SUBGROUPS
    groups = codes.reshape(rows, blocks, SUBGROUPS, SUBGROUP)
    subgroups = groups.reshape(rows, blocks, SUBGROUPS, subblocks, subblock)
    base = choose_base(subgroups, base_mode, max(bits_list))
    residual = subgroups.astype(np.int16) - base[..., None].astype(np.int16)
    residual_flat = residual.reshape(rows, blocks, SUBGROUPS, SUBGROUP)
    base_flat = base.reshape(rows, blocks, SUBGROUPS, subblocks)
    xg = x.view(len(x), blocks, SUBGROUPS, SUBGROUP)
    xsub = xg.reshape(len(x), blocks, SUBGROUPS, subblocks, subblock)
    alpha_t = torch.from_numpy(alpha).to(device)
    beta_t = torch.from_numpy(beta).to(device)
    weight_t = torch.from_numpy(weight).to(device)
    teacher = x @ weight_t.T
    # CPU base formula: alpha * sum_subblock(base * sum(x_subblock)) + beta * sum(x_group).
    base_t = torch.from_numpy(base.astype(np.float32)).to(device)
    base_projection = torch.einsum(
        "nbst,rbst,rbs->nr", xsub.sum(dim=-1), base_t, alpha_t
    ) + torch.einsum("nbs,rbs->nr", xg.sum(dim=-1), beta_t)
    exact_residual_t = torch.from_numpy(residual_flat.astype(np.float32)).to(device)
    exact_merged = base_projection + torch.einsum(
        "nbsi,rbsi,rbs->nr", xg, exact_residual_t, alpha_t
    )
    rows_out = []
    total_values = rows * blocks * QK_K
    base_count = rows * blocks * SUBGROUPS * subblocks
    base_bytes = math.ceil(base_count * 4 / 8)
    alpha_bytes = rows * blocks * SUBGROUPS * 2
    q4_bytes = rows * blocks * Q4_K_BLOCK_BYTES
    for bits in bits_list:
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        clipped = np.clip(residual, qmin, qmax)
        outliers = residual != clipped
        clipped_t = torch.from_numpy(clipped.reshape(rows, blocks, SUBGROUPS, SUBGROUP).astype(np.float32)).to(device)
        merged = base_projection + torch.einsum(
            "nbsi,rbsi,rbs->nr", xg, clipped_t, alpha_t
        )
        residual_bytes = math.ceil(total_values * bits / 8)
        correction_bytes_lower = int(np.count_nonzero(outliers))
        rows_out.append(
            {
                "residual_bits_gpu": bits,
                "residual_signed_min": int(residual.min()),
                "residual_signed_max": int(residual.max()),
                "residual_exact_signed_bits": signed_bits(residual),
                "residual_outlier_fraction": float(outliers.mean()),
                "residual_correction_int8_bytes_lower_bound": correction_bytes_lower,
                "clipped_projection_rel_l2": rel_l2(merged, teacher),
                "clipped_projection_abs_max": float((merged - teacher).abs().max()),
                "exact_merge_rel_l2": rel_l2(exact_merged, teacher),
                "exact_merge_abs_max": float((exact_merged - teacher).abs().max()),
                "gpu_residual_code_bytes": residual_bytes,
                "gpu_alpha_fp16_bytes": alpha_bytes,
                "gpu_residual_package_bytes": residual_bytes + alpha_bytes,
                "gpu_residual_package_mib": (residual_bytes + alpha_bytes) / (1024 * 1024),
                "base_host_metadata_bytes": base_bytes,
                "base_host_total_with_alpha_beta_bytes": base_bytes + 2 * alpha_bytes,
                "base_host_reduction_vs_q4k_without_alpha_beta": 1.0 - base_bytes / q4_bytes,
            }
        )
    base_terms = rows * blocks * SUBGROUPS * subblocks
    dense_terms = rows * blocks * QK_K
    return {
        "subblock_values": subblock,
        "subblocks_per_32_group": subblocks,
        "base_mode": base_mode,
        "cpu_base_formula": "sum_g alpha[j,g] * sum_t base[j,g,t] * sum(x[g,t]) + beta[j,g] * sum(x[g])",
        "residual_gpu_formula": "sum_g alpha[j,g] * dot(x[g], q[j,g] - base[j,g,subblock])",
        "merge_formula": "base_cpu + residual_gpu",
        "exact_cpu_base_rel_l2_vs_float_base": 0.0,
        "cpu_base_scalar_terms": base_terms,
        "dense_code_terms": dense_terms,
        "cpu_scalar_term_reduction": 1.0 - base_terms / dense_terms,
        "residual": rows_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4_K hierarchical subblock base plus GPU residual scan")
    parser.add_argument("model", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--subblocks", default="8,16,32")
    parser.add_argument("--base-modes", default="mean,midrange,min_outliers")
    parser.add_argument("--residual-bits", default="2,3,4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_np, _, _, _, _ = load_layer(args.holdout_root, args.layer)
    x = torch.from_numpy(x_np).to(device)
    rows_out = []
    dimensions = {}
    for projection in ("gate", "up"):
        codes, alpha, beta, weight, q4_bytes = load_q4k_codes(args.model, args.layer, projection)
        dimensions[projection] = {"shape": list(codes.shape), "q4k_bytes": q4_bytes}
        for subblock in [int(v) for v in args.subblocks.split(",") if v.strip()]:
            for mode in [v.strip() for v in args.base_modes.split(",") if v.strip()]:
                payload = evaluate(
                    x, codes, alpha, beta, weight, subblock, mode,
                    [int(v) for v in args.residual_bits.split(",") if v.strip()], device
                )
                payload["projection"] = projection
                rows_out.append(payload)

    result = {
        "experiment": "q4k_hierarchical_subblock_base_gpu_residual",
        "scope": "fixed-weight hierarchical formula scan; no runtime transfer or kernel benchmark",
        "model": str(args.model),
        "layer": args.layer,
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "dimensions": dimensions,
        "rows": rows_out,
        "interpretation": {
            "placement": "CPU/RAM keeps per-subblock base values and Q4_K metadata; GPU receives only clipped or exact residual codes plus alpha scales.",
            "why_this_differs": "The base is local to each output row but CPU evaluates it through activation subblock sums, so no per-row 32-value high-code scan is required.",
            "exactness": "Full residual reconstruction is exact to accumulation roundoff. Clipping is the only approximation and its outlier count is reported.",
            "traffic_caveat": "Residual package values are static weight artifacts. Per-token H2D depends on tile reuse; base gate/up output exchange remains a separate cost.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows_out), "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
