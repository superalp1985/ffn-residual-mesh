from __future__ import annotations

import argparse
import json
import math
import platform
import time
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


def signed_bits_for_exact(values: np.ndarray) -> int:
    minimum, maximum = int(values.min()), int(values.max())
    for bits in range(1, 9):
        if minimum >= -(1 << (bits - 1)) and maximum <= (1 << (bits - 1)) - 1:
            return bits
    return 8


def pack_bytes(values: int, bits: int) -> int:
    return math.ceil(values * bits / 8)


def fit_group_dictionary(
    groups: np.ndarray,
    dictionary_size: int,
    iterations: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit shared 32-position integer patterns for one projection.

    ``groups`` has shape [rows, input_groups, 32]. Each input group gets a
    separate small dictionary shared across all output rows. At runtime CPU
    evaluates K prototype dot products for the group, then reads a compact
    selector per output row. This is a fixed-weight formula table, not an
    activation lookup.
    """
    rows, input_groups, width = groups.shape
    if width != SUBGROUP:
        raise ValueError(f"expected {SUBGROUP} values per group")
    dictionary = np.empty((input_groups, dictionary_size, width), dtype=np.int8)
    selectors = np.empty((input_groups, rows), dtype=np.uint8)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for group_index in range(input_groups):
        data = torch.from_numpy(groups[:, group_index, :].astype(np.float32, copy=False)).to(device)
        # Deterministic random initial rows avoid the duplicate-centre bias of
        # a fixed evenly spaced slice for quantized code patterns.
        initial = torch.randperm(rows, generator=generator, device=device)[:dictionary_size]
        centers = data[initial].clone()
        for _ in range(iterations):
            distance = (data[:, None, :] - centers[None, :, :]).square().sum(dim=2)
            labels = distance.argmin(dim=1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, data)
            counts = torch.bincount(labels, minlength=dictionary_size).to(data.dtype).unsqueeze(1)
            centers = torch.where(counts > 0, sums / counts.clamp_min(1.0), centers)

        # The base must be integer Q4 code-domain data so q = base + residual
        # has an exact integer merge path when the full residual is retained.
        centers = torch.round(centers).clamp_(0, 15)
        labels = (data[:, None, :] - centers[None, :, :]).square().sum(dim=2).argmin(dim=1)
        dictionary[group_index] = centers.to(torch.int8).cpu().numpy()
        selectors[group_index] = labels.to(torch.uint8).cpu().numpy()
    return dictionary, selectors


def reconstruct_base(dictionary: np.ndarray, selectors: np.ndarray) -> np.ndarray:
    # [input_groups, rows, 32] -> [rows, input_groups, 32]
    return np.stack(
        [dictionary[group_index, selectors[group_index]] for group_index in range(len(dictionary))],
        axis=1,
    )


def cpu_base_projection(
    x: np.ndarray,
    dictionary: np.ndarray,
    selectors: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Evaluate the dictionary base without scanning per-row 32-value codes."""
    tokens, hidden = x.shape
    rows, blocks, subgroups = alpha.shape
    if hidden != blocks * QK_K:
        raise ValueError("input dimensions do not match Q4_K geometry")
    group_count = blocks * subgroups
    xg = x.reshape(tokens, group_count, SUBGROUP)
    alpha_flat = alpha.reshape(rows, group_count)
    beta_flat = beta.reshape(rows, group_count)
    result = np.zeros((tokens, rows), dtype=np.float32)
    sums = xg.sum(axis=2)
    for group_index in range(group_count):
        prototype_dot = xg[:, group_index] @ dictionary[group_index].astype(np.float32).T
        selected = prototype_dot[:, selectors[group_index]]
        result += selected * alpha_flat[:, group_index][None, :]
        result += sums[:, group_index, None] * beta_flat[:, group_index][None, :]
    return result


def timed_cpu_base(
    x: np.ndarray,
    dictionary: np.ndarray,
    selectors: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    samples: int,
    repeats: int,
) -> dict:
    values = x[: min(samples, len(x))].astype(np.float32, copy=False)
    cpu_base_projection(values[:1], dictionary, selectors, alpha, beta)
    timings = []
    for _ in range(repeats):
        for item in values:
            start = time.perf_counter_ns()
            cpu_base_projection(item[None, :], dictionary, selectors, alpha, beta)
            timings.append((time.perf_counter_ns() - start) / 1000.0)
    return {
        "tokens": len(timings),
        "median_us": float(np.median(timings)),
        "p95_us": float(np.percentile(timings, 95)),
        "mean_us": float(np.mean(timings)),
    }


def evaluate_projection(
    x: torch.Tensor,
    x_cpu: np.ndarray,
    codes: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    weight: np.ndarray,
    dictionary_size: int,
    iterations: int,
    residual_bits: list[int],
    device: torch.device,
    timing_samples: int,
    timing_repeats: int,
    seed: int,
) -> dict:
    rows, blocks, _ = codes.shape
    input_groups = blocks * SUBGROUPS
    groups = codes.reshape(rows, input_groups, SUBGROUP)
    dictionary, selectors = fit_group_dictionary(
        groups, dictionary_size, iterations, device, seed
    )
    base_codes = reconstruct_base(dictionary, selectors)
    residual = groups.astype(np.int16) - base_codes.astype(np.int16)
    weight_t = torch.from_numpy(weight).to(device)
    teacher = x @ weight_t.T
    alpha_t = torch.from_numpy(alpha.reshape(rows, input_groups)).to(device)
    beta_t = torch.from_numpy(beta.reshape(rows, input_groups)).to(device)
    base_t = torch.from_numpy(base_codes.astype(np.float32)).to(device)
    xg = x.view(len(x), input_groups, SUBGROUP)
    base_projection = (
        torch.einsum("ngi,rgi,rg->nr", xg, base_t, alpha_t)
        + torch.einsum("ng,rg->nr", xg.sum(dim=2), beta_t)
    )
    cpu_base = cpu_base_projection(x_cpu, dictionary, selectors, alpha, beta)
    cpu_base_error = rel_l2(torch.from_numpy(cpu_base).to(device), base_projection)
    rows_out = []
    total_values = rows * input_groups * SUBGROUP
    selector_bits = max(1, math.ceil(math.log2(dictionary_size)))
    dictionary_bytes = pack_bytes(input_groups * dictionary_size * SUBGROUP, 4)
    selector_bytes = pack_bytes(rows * input_groups, selector_bits)
    host_alpha_beta_bytes = rows * input_groups * 2 * 2
    for bits in residual_bits:
        minimum, maximum = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        clipped = np.clip(residual, minimum, maximum)
        residual_t = torch.from_numpy(clipped.astype(np.float32)).to(device)
        gpu_residual = torch.einsum("ngi,rgi,rg->nr", xg, residual_t, alpha_t)
        merged = base_projection + gpu_residual
        full_residual = torch.from_numpy(residual.astype(np.float32)).to(device)
        exact_merged = base_projection + torch.einsum("ngi,rgi,rg->nr", xg, full_residual, alpha_t)
        outliers = residual != clipped
        exact_bits = signed_bits_for_exact(residual)
        residual_bytes = pack_bytes(total_values, bits)
        alpha_gpu_bytes = rows * input_groups * 2
        rows_out.append(
            {
                "residual_bits_gpu": bits,
                "residual_signed_min": int(residual.min()),
                "residual_signed_max": int(residual.max()),
                "residual_exact_signed_bits": exact_bits,
                "residual_outlier_fraction": float(outliers.mean()),
                "exact_merge_rel_l2": rel_l2(exact_merged, teacher),
                "exact_merge_abs_max": float((exact_merged - teacher).abs().max()),
                "clipped_merge_rel_l2": rel_l2(merged, teacher),
                "clipped_merge_abs_max": float((merged - teacher).abs().max()),
                "gpu_residual_package_bytes": residual_bytes + alpha_gpu_bytes,
                "gpu_residual_package_mib": (residual_bytes + alpha_gpu_bytes) / (1024 * 1024),
                "gpu_residual_code_bytes": residual_bytes,
                "gpu_alpha_fp16_bytes": alpha_gpu_bytes,
            }
        )
    original_q4_bytes = rows * blocks * Q4_K_BLOCK_BYTES
    base_code_terms = input_groups * dictionary_size * SUBGROUP
    base_row_aggregate_terms = rows * input_groups
    return {
        "dictionary_size": dictionary_size,
        "input_groups": input_groups,
        "base_cpu_formula": "sum_g alpha[j,g] * dot(x_g, prototype[g, selector[j,g]]) + beta[j,g] * sum(x_g)",
        "residual_gpu_formula": "sum_g alpha[j,g] * dot(x_g, q[j,g] - prototype[g, selector[j,g]])",
        "merge_formula": "base_cpu + residual_gpu",
        "cpu_base_replay_rel_l2": cpu_base_error,
        "cpu_base_timing": timed_cpu_base(
            x_cpu, dictionary, selectors, alpha, beta, timing_samples, timing_repeats
        ),
        "base_artifact_host_bytes": {
            "prototype_q4_bytes": dictionary_bytes,
            "selector_bits": selector_bits,
            "selector_bytes": selector_bytes,
            "alpha_beta_fp16_bytes": host_alpha_beta_bytes,
            "total_bytes": dictionary_bytes + selector_bytes + host_alpha_beta_bytes,
            "original_q4k_projection_bytes": original_q4_bytes,
            "reduction_vs_q4k": 1.0 - (dictionary_bytes + selector_bytes + host_alpha_beta_bytes) / original_q4_bytes,
        },
        "cpu_base_work_proxy": {
            "prototype_dot_code_terms": base_code_terms,
            "row_selector_aggregate_terms": base_row_aggregate_terms,
            "dense_code_terms_replaced": rows * input_groups * SUBGROUP,
            "prototype_dot_reduction_vs_dense": 1.0 - base_code_terms / (rows * input_groups * SUBGROUP),
            "note": "Selector aggregation still scans alpha/beta plus selectors, but does not scan a 32-value high-code vector per row/group.",
        },
        "residual": rows_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a shared-pattern CPU base plus GPU code residual for Q4_K gate/up"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--dictionary-sizes", default="4,8,16")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--residual-bits", default="2,3,4")
    parser.add_argument("--timing-samples", type=int, default=8)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_np, _, _, _, _ = load_layer(args.holdout_root, args.layer)
    x = torch.from_numpy(x_np).to(device)
    residual_bits = [int(v) for v in args.residual_bits.split(",") if v.strip()]
    result_rows = []
    dimensions = {}
    for projection_index, projection in enumerate(("gate", "up")):
        codes, alpha, beta, weight, q4_bytes = load_q4k_codes(args.model, args.layer, projection)
        dimensions[projection] = {"shape": list(codes.shape), "q4k_bytes": q4_bytes}
        for dictionary_size in [int(v) for v in args.dictionary_sizes.split(",") if v.strip()]:
            payload = evaluate_projection(
                x,
                x_np,
                codes,
                alpha,
                beta,
                weight,
                dictionary_size,
                args.iterations,
                residual_bits,
                device,
                args.timing_samples,
                args.timing_repeats,
                args.seed + projection_index * 10_000 + dictionary_size,
            )
            payload["projection"] = projection
            result_rows.append(payload)

    result = {
        "experiment": "q4k_shared_pattern_cpu_base_gpu_residual",
        "scope": "fixed-weight formula-table scan; CPU base replay and mathematical residual verification, no PCIe/kernel benchmark",
        "model": str(args.model),
        "layer": args.layer,
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "dimensions": dimensions,
        "rows": result_rows,
        "interpretation": {
            "what_is_new": "Unlike a high/low bit split, the CPU base has no per-row 32-value high-code stream. It evaluates a few shared fixed patterns, then consumes selector and Q4_K scale/offset metadata.",
            "exactness": "Keeping the full signed residual makes q = prototype + residual exact. Any residual clipping is the only declared approximation in this experiment.",
            "acceptance": "A useful candidate requires a low residual bit width with acceptable projection error and a CPU base that is faster than a dense code scan; a compact base alone is insufficient.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(result_rows), "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
