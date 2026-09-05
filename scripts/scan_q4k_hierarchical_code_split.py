from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import Q4_K, dequantize

from evaluate_polynomial_base_residual import load_layer


QK_K = 256
Q4_K_BLOCK_BYTES = 144
SUBGROUP = 32
SUBGROUPS = QK_K // SUBGROUP


def load_q4k_codes(
    model: Path, layer: int, projection: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    reader = GGUFReader(str(model))
    tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_{projection}.weight")
    if int(tensor.tensor_type) != 12:
        raise ValueError(f"expected Q4_K tensor type 12, got {tensor.tensor_type}")
    rows, packed_row_bytes = (int(v) for v in tensor.data.shape)
    if packed_row_bytes % Q4_K_BLOCK_BYTES:
        raise ValueError("unexpected Q4_K row size")
    blocks_per_row = packed_row_bytes // Q4_K_BLOCK_BYTES
    raw = tensor.data.reshape(rows, blocks_per_row, Q4_K_BLOCK_BYTES)
    scales = raw[:, :, 4:16].reshape(-1, 12)
    sc, minimum = Q4_K.get_scale_min(scales)
    d = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, blocks_per_row, 1)
    dmin = raw[:, :, 2:4].copy().view(np.float16).astype(np.float32).reshape(rows, blocks_per_row, 1)
    alpha = d * sc.reshape(rows, blocks_per_row, SUBGROUPS).astype(np.float32)
    beta = -dmin * minimum.reshape(rows, blocks_per_row, SUBGROUPS).astype(np.float32)
    qs = raw[:, :, 16:].reshape(rows, blocks_per_row, 4, 32)
    lo = qs & 0x0F
    hi = qs >> 4
    # Q4_K stores four 32-value chunks, each with low/high nibbles.
    codes = np.stack([lo, hi], axis=3).reshape(rows, blocks_per_row, QK_K)
    # Dequantized values are used only for activation-weighted holdout error.
    weight = dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)
    return codes, alpha, beta, weight, int(tensor.n_bytes)


def load_q4k_raw(model: Path, layer: int, projection: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return decoded Q4_K code/scales in the exact block ordering."""
    return load_q4k_codes(model, layer, projection)


def choose_base(q: np.ndarray, mode: str, bits: int | None = None) -> int:
    if mode == "mean":
        return int(np.rint(float(q.mean())))
    if mode == "midrange":
        return int(np.rint((int(q.min()) + int(q.max())) / 2.0))
    if mode == "min_outliers":
        if bits is None:
            raise ValueError("min_outliers requires residual bit width")
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        best = None
        for candidate in range(16):
            residual = q.astype(np.int16) - candidate
            clipped = np.clip(residual, qmin, qmax)
            correction = residual - clipped
            score = (
                int(np.count_nonzero(correction)),
                int(np.abs(correction).sum()),
                int(np.square(residual).sum()),
            )
            if best is None or score < best[0]:
                best = (score, candidate)
        assert best is not None
        return best[1]
    raise ValueError(f"unknown base mode: {mode}")


def choose_bases(groups: np.ndarray, mode: str, bits: int) -> np.ndarray:
    """Vectorized base selection for [rows, blocks, subgroups, 32] code groups."""
    if mode == "mean":
        return np.rint(groups.mean(axis=-1)).astype(np.int16)
    if mode == "midrange":
        return np.rint((groups.min(axis=-1) + groups.max(axis=-1)) / 2.0).astype(np.int16)
    if mode != "min_outliers":
        raise ValueError(f"unknown base mode: {mode}")

    flat = groups.reshape(-1, SUBGROUP)
    candidates = np.arange(16, dtype=np.int16)
    result = np.empty(flat.shape[0], dtype=np.int16)
    # Keep the temporary residual tensor bounded; this is the same exhaustive
    # 16-base search as choose_base(), but without millions of Python loops.
    chunk = 100_000
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    for start in range(0, flat.shape[0], chunk):
        values = flat[start : start + chunk].astype(np.int16)
        residual = values[:, None, :] - candidates[None, :, None]
        clipped = np.clip(residual, qmin, qmax)
        correction = residual - clipped
        count = np.count_nonzero(correction, axis=2)
        magnitude = np.abs(correction).sum(axis=2)
        energy = np.square(residual).sum(axis=2)
        # np.lexsort makes the tie-break deterministic: count, magnitude,
        # residual energy, then smallest candidate.
        order = np.lexsort((np.broadcast_to(candidates, count.shape), energy, magnitude, count), axis=1)
        result[start : start + len(values)] = candidates[order[:, 0]]
    return result.reshape(groups.shape[:-1])


def signed_bits_for_exact(residual: np.ndarray) -> int:
    minimum = int(residual.min())
    maximum = int(residual.max())
    for bits in range(1, 9):
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        if minimum >= qmin and maximum <= qmax:
            return bits
    return 8


def clip_residual(residual: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    clipped = np.clip(residual, qmin, qmax).astype(np.int16)
    outlier = residual.astype(np.int16) - clipped
    return clipped, outlier


def weighted_projection_error(
    x: torch.Tensor,
    weight: torch.Tensor,
    codes: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    base: np.ndarray,
    bits: int,
) -> tuple[float, float, float]:
    rows, blocks, _ = codes.shape
    code_groups = codes.reshape(rows, blocks, SUBGROUPS, SUBGROUP)
    residual = code_groups.astype(np.int16) - base[..., None].astype(np.int16)
    clipped, _ = clip_residual(residual, bits)
    approx_codes = (base[..., None].astype(np.int16) + clipped).astype(np.uint8)
    # Q4_K scales/mins are unchanged; replacing nibble codes gives a direct
    # code-domain approximation without introducing a second quantizer.
    approx_t = torch.from_numpy(approx_codes.astype(np.float32)).to(x.device)
    w_t = weight.to(x.device)
    qa_groups = approx_t
    alpha_t = torch.from_numpy(alpha).to(x.device).unsqueeze(-1)
    beta_t = torch.from_numpy(beta).to(x.device).unsqueeze(-1)
    approx_w = alpha_t * qa_groups + beta_t
    approx_w = approx_w.view_as(w_t)
    teacher = x @ w_t.T
    pred = x @ approx_w.T
    per = torch.linalg.vector_norm(pred - teacher, dim=1) / torch.clamp(torch.linalg.vector_norm(teacher, dim=1), min=1e-6)
    return float(per.mean()), float(per.quantile(0.95)), float((pred - teacher).abs().max())


def summarize_scheme(codes: np.ndarray, bits: int, base_mode: str, base: np.ndarray | None = None) -> dict:
    rows, blocks, _ = codes.shape
    groups = codes.reshape(rows, blocks, SUBGROUPS, SUBGROUP)
    if base is None:
        base = choose_bases(groups, base_mode, bits)
    residual = groups.astype(np.int16) - base[..., None].astype(np.int16)
    clipped, outlier = clip_residual(residual, bits)
    exact_bits = np.vectorize(signed_bits_for_exact, signature="(n)->()")(
        residual.reshape(-1, SUBGROUP)
    ).reshape(rows, blocks, SUBGROUPS)
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    outlier_count = int(np.count_nonzero(outlier))
    total_values = int(np.prod(residual.shape))
    # Base is one signed byte per 32-value subgroup. Residuals are packed at
    # the requested bit width; optional outliers use int8 correction values.
    base_bytes = math.ceil(rows * blocks * SUBGROUPS / 2)
    residual_bytes = math.ceil(total_values * bits / 8)
    outlier_bytes = outlier_count  # index omitted in this lower-bound ledger
    outlier_bitmap_bytes = math.ceil(total_values / 8) if outlier_count else 0
    q4_bytes = rows * blocks * Q4_K_BLOCK_BYTES
    header_bytes = rows * blocks * 16
    approximate_bytes = header_bytes + base_bytes + residual_bytes
    exact_bitmap_bytes = approximate_bytes + outlier_bitmap_bytes + outlier_bytes
    return {
        "base_mode": base_mode,
        "bits": bits,
        "qmin": qmin,
        "qmax": qmax,
        "groups": int(rows * blocks * SUBGROUPS),
        "exact_signed_bits_p50": float(np.percentile(exact_bits, 50)),
        "exact_signed_bits_p90": float(np.percentile(exact_bits, 90)),
        "exact_signed_bits_p99": float(np.percentile(exact_bits, 99)),
        "groups_exact_at_bits": int(np.count_nonzero(exact_bits <= bits)),
        "values_total": total_values,
        "outlier_values": outlier_count,
        "outlier_fraction": outlier_count / total_values,
        "artifact_lower_bound_bytes": {
            "q4k_scale_min_bytes": header_bytes,
            "base_packed4_bytes": base_bytes,
            "residual_packed_bytes": residual_bytes,
            "outlier_correction_int8_bytes": outlier_bytes,
            "approximate_total_bytes": approximate_bytes,
            "exact_total_without_indices_bytes": approximate_bytes + outlier_bytes,
            "exact_total_bitmap_indices_bytes": exact_bitmap_bytes,
            "original_q4k_bytes": q4_bytes,
            "approximate_reduction_vs_q4k": 1.0 - approximate_bytes / q4_bytes,
            "exact_bitmap_reduction_vs_q4k": 1.0 - exact_bitmap_bytes / q4_bytes,
            "dynamic_residual_only_bytes_if_header_base_resident": residual_bytes,
            "dynamic_residual_reduction_vs_q4k": 1.0 - residual_bytes / q4_bytes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Q4_K 32-value shared-base plus signed residual schemes")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--base-modes", default="mean,midrange,min_outliers")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout_x_np, _, _, _, _ = load_layer(args.holdout_root, args.layer)
    holdout_x = torch.from_numpy(holdout_x_np).to(device)
    projections = {}
    rows = []
    for projection in ("gate", "up"):
        codes, alpha, beta, weight_np, q4_bytes = load_q4k_codes(args.model, args.layer, projection)
        weight = torch.from_numpy(weight_np).to(device)
        projections[projection] = {
            "shape": list(codes.shape),
            "q4k_bytes": q4_bytes,
            "code_min": int(codes.min()),
            "code_max": int(codes.max()),
            "code_mean": float(codes.mean()),
        }
        groups = codes.reshape(codes.shape[0], codes.shape[1], SUBGROUPS, SUBGROUP)
        for mode in [value.strip() for value in args.base_modes.split(",") if value.strip()]:
            for bits in [int(value) for value in args.bits.split(",") if value.strip()]:
                base = choose_bases(groups, mode, bits)
                summary = summarize_scheme(codes, bits, mode, base)
                holdout_mean, holdout_p95, holdout_max = weighted_projection_error(
                    holdout_x, weight, codes, alpha, beta, base, bits
                )
                summary.update(
                    {
                        "projection": projection,
                        "holdout_rel_l2_mean": holdout_mean,
                        "holdout_rel_l2_p95": holdout_p95,
                        "holdout_abs_max": holdout_max,
                    }
                )
                rows.append(summary)

        # Cross-output sharing probe: a shared base per (Q4_K block, 32-value
        # subgroup) would be useful only if most rows agree on it. Record the
        # modal agreement to distinguish true structure from per-row storage.
        base_mean = np.rint(groups.mean(axis=-1)).astype(np.int16)
        agreement = []
        for block in range(base_mean.shape[1]):
            for subgroup in range(base_mean.shape[2]):
                counts = np.bincount(base_mean[:, block, subgroup], minlength=16)
                agreement.append(float(counts.max() / base_mean.shape[0]))
        projections[projection]["cross_output_mean_base_mode_agreement"] = {
            "mean": float(np.mean(agreement)),
            "p50": float(np.percentile(agreement, 50)),
            "p90": float(np.percentile(agreement, 90)),
            "all_rows_same_fraction": float(np.mean(np.asarray(agreement) == 1.0)),
        }

    result = {
        "experiment": "q4k_hierarchical_code_split",
        "formula": "q=b+r; sum(z*q)=b*sum(z)+sum(z*r), with Q4_K scale/min retained per 256-value block",
        "scope": "fixed-weight code-domain decomposition; no transfer or kernel benchmark",
        "model": str(args.model),
        "layer": args.layer,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "dimensions": projections,
        "rows": rows,
        "interpretation": {
            "exactness": "If residual outliers are retained, q reconstruction is exact; clipped rows are approximation only.",
            "traffic": "The byte ledger is a static lower bound. Runtime benefit requires base terms to be generated from aggregate sums rather than transferring full outputs.",
            "outlier_index_caveat": "Outlier correction bytes exclude index metadata, so reported reductions are optimistic.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
