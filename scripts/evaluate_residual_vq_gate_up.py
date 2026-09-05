from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_compiled_crosschannel_swiglu import profile_h2d_bytes
from evaluate_preexpanded_sparse_cp import load_weights
from evaluate_polynomial_base_residual import load_layer


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target, dim=1) / torch.clamp(torch.linalg.vector_norm(target, dim=1), min=1e-6)


def assign(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    distances = x.square().sum(1, keepdim=True) - 2.0 * (x @ centers.T) + centers.square().sum(1)
    return distances.argmin(dim=1)


def fit_kmeans(x: torch.Tensor, codes: int, seed: int, iterations: int = 50) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    centers = x[torch.randperm(len(x), device="cuda", generator=generator)[:codes]].clone()
    for _ in range(iterations):
        labels = assign(x, centers)
        next_centers = torch.zeros_like(centers)
        next_centers.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=codes).to(x.dtype).unsqueeze(1)
        replacement = x[torch.randint(len(x), (codes,), device="cuda", generator=generator)]
        next_centers = torch.where(counts > 0, next_centers / counts.clamp_min(1.0), replacement)
        if torch.max(torch.abs(next_centers - centers)) < 1e-5:
            centers = next_centers
            break
        centers = next_centers
    return centers


def profile_down_merge(g: torch.Tensor, u: torch.Tensor, wd: torch.Tensor, repeats: int) -> dict[str, float]:
    for _ in range(20):
        (torch.nn.functional.silu(g) * u) @ wd.T
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            (torch.nn.functional.silu(g) * u) @ wd.T
            stop.record()
            stop.synchronize()
            times.append(float(start.elapsed_time(stop)))
    return {
        "median_us_per_token": float(np.median(times) * 1000 / len(g)),
        "p95_us_per_token": float(np.percentile(times, 95) * 1000 / len(g)),
        "scope": "PyTorch fp32 down merge only; production requires a Q4 down kernel",
    }


def routed_metrics(residual: torch.Tensor, error: torch.Tensor, calibration_residual: torch.Tensor, quantile: float) -> dict[str, float]:
    threshold = torch.quantile(calibration_residual, quantile)
    accept = residual <= threshold
    accepted_error = error[accept]
    return {
        "residual_quantile": quantile,
        "threshold": float(threshold),
        "approx_fraction": float(accept.float().mean()),
        "fallback_fraction": float((~accept).float().mean()),
        "accepted_rel_l2": float(accepted_error.mean()) if len(accepted_error) else 0.0,
        "accepted_rel_l2_p95": float(torch.quantile(accepted_error, 0.95)) if len(accepted_error) else 0.0,
        "hybrid_rel_l2": float(accepted_error.sum() / len(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual-VQ preexpanded gate/up tables with exact GPU SwiGLU and down")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--codes", type=int, default=8)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--profile-repeats", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    calibration_x_np, _, _, _, calibration_capture_y_np = load_layer(args.calibration_root, args.layer)
    holdout_x_np, _, _, _, holdout_capture_y_np = load_layer(args.holdout_root, args.layer)
    calibration_x = torch.from_numpy(calibration_x_np.astype(np.float32, copy=False)).cuda()
    holdout_x = torch.from_numpy(holdout_x_np.astype(np.float32, copy=False)).cuda()
    calibration_capture_y = torch.from_numpy(calibration_capture_y_np.astype(np.float32, copy=False)).cuda()
    holdout_capture_y = torch.from_numpy(holdout_capture_y_np.astype(np.float32, copy=False)).cuda()
    (wg_np, wg_q4_bytes), (wu_np, wu_q4_bytes), (wd_np, wd_q4_bytes) = load_weights(args.model, args.layer)
    wg, wu, wd = torch.from_numpy(wg_np).cuda(), torch.from_numpy(wu_np).cuda(), torch.from_numpy(wd_np).cuda()
    hidden, ffn = calibration_x.shape[1], wg.shape[0]
    if hidden % args.block_size:
        raise SystemExit(f"block-size {args.block_size} must divide hidden {hidden}")
    blocks = hidden // args.block_size

    codebooks: list[list[torch.Tensor]] = []
    gate_tables: list[list[torch.Tensor]] = []
    up_tables: list[list[torch.Tensor]] = []
    start = time.perf_counter()
    # This loop is the cold-start compiler. It scans only original Wg/Wu here,
    # then emits table contributions that runtime can add without reading them.
    for block in range(blocks):
        span = slice(block * args.block_size, (block + 1) * args.block_size)
        residual = calibration_x[:, span].clone()
        block_codebooks = []
        block_gate_tables = []
        block_up_tables = []
        for level in range(args.levels):
            centers = fit_kmeans(residual, args.codes, 50000 + block * 100 + level, args.kmeans_iters)
            labels = assign(residual, centers)
            residual -= centers[labels]
            block_codebooks.append(centers)
            block_gate_tables.append(centers @ wg[:, span].T)
            block_up_tables.append(centers @ wu[:, span].T)
        codebooks.append(block_codebooks)
        gate_tables.append(block_gate_tables)
        up_tables.append(block_up_tables)
    cold_compile_seconds = time.perf_counter() - start

    def encode(x: torch.Tensor, fp16_tables: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = torch.zeros((len(x), ffn), device="cuda")
        u = torch.zeros_like(g)
        residual_energy = torch.zeros(len(x), device="cuda")
        for block in range(blocks):
            span = slice(block * args.block_size, (block + 1) * args.block_size)
            residual = x[:, span].clone()
            for level in range(args.levels):
                labels = assign(residual, codebooks[block][level])
                residual -= codebooks[block][level][labels]
                gate = gate_tables[block][level]
                up = up_tables[block][level]
                if fp16_tables:
                    gate, up = gate.to(torch.float16).to(torch.float32), up.to(torch.float16).to(torch.float32)
                g += gate[labels]
                u += up[labels]
            residual_energy += residual.square().sum(dim=1)
        residual_energy = torch.sqrt(residual_energy) / torch.clamp(torch.linalg.vector_norm(x, dim=1), min=1e-6)
        return g, u, residual_energy

    calibration_g, calibration_u, calibration_residual = encode(calibration_x, fp16_tables=True)
    holdout_g, holdout_u, holdout_residual = encode(holdout_x, fp16_tables=True)
    with torch.no_grad():
        calibration_prediction = (torch.nn.functional.silu(calibration_g) * calibration_u) @ wd.T
        holdout_prediction = (torch.nn.functional.silu(holdout_g) * holdout_u) @ wd.T
        weight_teacher_holdout = (torch.nn.functional.silu(holdout_x @ wg.T) * (holdout_x @ wu.T)) @ wd.T
    holdout_error_teacher = rel_l2(holdout_prediction, weight_teacher_holdout)
    holdout_error_capture = rel_l2(holdout_prediction, holdout_capture_y)
    table_elements = blocks * args.levels * args.codes * ffn
    codebook_elements = blocks * args.levels * args.codes * args.block_size
    cpu_table_bytes = table_elements * 2 * 2
    cpu_codebook_bytes = codebook_elements * 2
    down_gpu_q4_bytes = wd_q4_bytes
    d2h_input_bytes = hidden * 2
    h2d_gate_up_bytes = 2 * ffn * 2
    dynamic_total = d2h_input_bytes + h2d_gate_up_bytes
    cpu_table_read_bytes = blocks * args.levels * 2 * ffn * 2
    gpu_down_mac = hidden * ffn
    result = {
        "experiment": "residual_vq_preexpanded_gate_up",
        "formula": "x_hat=sum_(block b, level l) codebook[b,l,code(x_residual_b)]; g_hat=Wg*x_hat; u_hat=Wu*x_hat; y=Wd*(SiLU(g_hat)*u_hat)",
        "runtime_contract": {
            "original_ffn_weight_reads": 0,
            "original_ffn_weight_use": "cold-start table generation and exact fallback only",
            "cpu": "residual-VQ encode each input block; add preexpanded gate/up contribution vectors",
            "gpu": "receive compact g_hat/u_hat, run unmodified SiLU and down projection with resident Wd",
            "dynamic_bytes_if_cpu_routes": {"d2h_ffn_input_fp16": d2h_input_bytes, "h2d_gate_up_fp16": h2d_gate_up_bytes, "total": dynamic_total},
        },
        "cold_start": {
            "seconds": cold_compile_seconds,
            "original_weight_scans": "all Wg/Wu columns while generating contribution tables; no runtime scans",
            "blocks": blocks,
            "levels": args.levels,
            "codes_per_level": args.codes,
        },
        "layer": args.layer,
        "dimensions": {"hidden": hidden, "ffn": ffn},
        "accuracy": {
            "calibration_rel_l2_vs_captured_ffn": float(rel_l2(calibration_prediction, calibration_capture_y).mean()),
            "holdout_rel_l2_vs_weight_teacher": float(holdout_error_teacher.mean()),
            "holdout_rel_l2_p95_vs_weight_teacher": float(torch.quantile(holdout_error_teacher, 0.95)),
            "holdout_rel_l2_vs_captured_ffn": float(holdout_error_capture.mean()),
            "holdout_rel_l2_p95_vs_captured_ffn": float(torch.quantile(holdout_error_capture, 0.95)),
            "weight_teacher_rel_l2_vs_captured_holdout": float(rel_l2(weight_teacher_holdout, holdout_capture_y).mean()),
            "routing": [
                routed_metrics(holdout_residual, holdout_error_capture, calibration_residual, quantile)
                for quantile in (0.90, 0.95, 0.99)
            ],
        },
        "artifact": {
            "cpu_gate_up_tables_fp16_bytes": cpu_table_bytes,
            "cpu_codebooks_fp16_bytes": cpu_codebook_bytes,
            "gpu_down_weight_q4_bytes": down_gpu_q4_bytes,
            "total_bytes": cpu_table_bytes + cpu_codebook_bytes + down_gpu_q4_bytes,
            "original_full_ffn_q4_bytes": wg_q4_bytes + wu_q4_bytes + wd_q4_bytes,
            "gpu_weight_bytes_removed_vs_full_ffn": wg_q4_bytes + wu_q4_bytes,
        },
        "work_and_wait": {
            "cpu_table_read_bytes_per_token": cpu_table_read_bytes,
            "cpu_projection_weight_reads_removed_per_token_q4": wg_q4_bytes + wu_q4_bytes,
            "gpu_down_mac_per_token": gpu_down_mac,
            "gpu_mac_per_h2d_gate_up_byte": gpu_down_mac / h2d_gate_up_bytes,
            "gpu_mac_per_total_cpu_routed_dynamic_byte": gpu_down_mac / dynamic_total,
            "pinned_h2d_gate_up_profile": profile_h2d_bytes(2 * ffn, args.profile_repeats),
            "gpu_down_merge_decode_batch1": profile_down_merge(holdout_g[:1], holdout_u[:1], wd, args.profile_repeats),
            "gpu_down_merge_throughput_batch": profile_down_merge(holdout_g[: min(64, len(holdout_g))], holdout_u[: min(64, len(holdout_u))], wd, args.profile_repeats),
            "scope": "copy and down merge are measured separately; full pipeline wait requires llama.cpp callback integration",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
