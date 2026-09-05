from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_compiled_crosschannel_swiglu import artifact_bytes, profile_h2d_bytes, profile_runtime, rel_l2, runtime_mac
from evaluate_preexpanded_sparse_cp import load_weights
from evaluate_polynomial_base_residual import load_layer


class CompiledCrossChannelSwiGLU(nn.Module):
    """Compiled runtime artifact. It has no Wg/Wu/Wd references."""

    def __init__(self, hidden: int, rank: int, full_linear: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden) if full_linear else None
        self.gate = nn.Linear(hidden, rank)
        self.up = nn.Linear(hidden, rank)
        self.down = nn.Linear(rank, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x) if self.linear is not None else 0.0
        return base + self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x)) + self.bias


def teacher_forward(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, batch_size: int) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for begin in range(0, len(x), batch_size):
            chunk = x[begin : begin + batch_size]
            rows.append((torch.nn.functional.silu(chunk @ wg.T) * (chunk @ wu.T)) @ wd.T)
    return torch.cat(rows)


def make_cold_start_inputs(x: torch.Tensor, count: int, noise_scale: float, seed: int) -> torch.Tensor:
    """Stay near measured layer states while creating extra cold-start teacher samples."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    first = torch.randint(len(x), (count,), device="cuda", generator=generator)
    second = torch.randint(len(x), (count,), device="cuda", generator=generator)
    mix = torch.rand((count, 1), device="cuda", generator=generator)
    std = torch.clamp(x.std(dim=0, keepdim=True), min=1e-4)
    jitter = torch.randn((count, x.shape[1]), device="cuda", generator=generator) * std * noise_scale
    return mix * x[first] + (1.0 - mix) * x[second] + jitter


def fit_rank(
    rank: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    holdout_x: torch.Tensor,
    holdout_teacher_y: torch.Tensor,
    holdout_capture_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    full_linear: bool,
) -> tuple[CompiledCrossChannelSwiGLU, dict[str, float]]:
    model = CompiledCrossChannelSwiGLU(train_x.shape[1], rank, full_linear).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    best_state: dict[str, torch.Tensor] | None = None
    best_valid = float("inf")
    stale = 0
    generator = torch.Generator(device="cuda")
    generator.manual_seed(7000 + rank + int(full_linear))
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), device="cuda", generator=generator)
        for begin in range(0, len(train_x), batch_size):
            selected = order[begin : begin + batch_size]
            prediction = model(train_x[selected])
            target = train_y[selected]
            loss = torch.mean((prediction - target) ** 2) / torch.clamp(torch.mean(target**2), min=1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            valid_error = float(rel_l2(model(valid_x), valid_y).mean())
        if valid_error < best_valid:
            best_valid = valid_error
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 40:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_error = rel_l2(model(train_x), train_y)
        valid_error = rel_l2(model(valid_x), valid_y)
        holdout_teacher_error = rel_l2(model(holdout_x), holdout_teacher_y)
        holdout_capture_error = rel_l2(model(holdout_x), holdout_capture_y)
    return model, {
        "epochs_completed": epoch + 1,
        "cold_compile_fit_seconds": time.perf_counter() - start,
        "train_rel_l2_vs_weight_teacher": float(train_error.mean()),
        "validation_rel_l2_vs_weight_teacher": float(valid_error.mean()),
        "holdout_rel_l2_vs_weight_teacher": float(holdout_teacher_error.mean()),
        "holdout_rel_l2_p95_vs_weight_teacher": float(torch.quantile(holdout_teacher_error, 0.95)),
        "holdout_rel_l2_vs_captured_ffn": float(holdout_capture_error.mean()),
        "holdout_rel_l2_p95_vs_captured_ffn": float(torch.quantile(holdout_capture_error, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start distill a cross-channel SwiGLU artifact directly from FFN weights")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--ranks", default="256")
    parser.add_argument("--full-linear", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=2048)
    parser.add_argument("--noise-scale", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
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
    (wg_np, wg_q4_bytes), (wu_np, wu_q4_bytes), (wd_np, wd_q4_bytes) = load_weights(args.model, args.layer)
    wg = torch.from_numpy(wg_np).cuda()
    wu = torch.from_numpy(wu_np).cuda()
    wd = torch.from_numpy(wd_np).cuda()
    calibration_x_raw = torch.from_numpy(calibration_x_np.astype(np.float32, copy=False)).cuda()
    holdout_x_raw = torch.from_numpy(holdout_x_np.astype(np.float32, copy=False)).cuda()
    holdout_capture_y = torch.from_numpy(holdout_capture_y_np.astype(np.float32, copy=False)).cuda()

    split = int(len(calibration_x_raw) * 0.8)
    seed_x_raw = calibration_x_raw[:split]
    valid_x_raw = calibration_x_raw[split:]
    compile_start = time.perf_counter()
    synthetic_x_raw = make_cold_start_inputs(seed_x_raw, args.synthetic_samples, args.noise_scale, 2468)
    train_x_raw = torch.cat((seed_x_raw, synthetic_x_raw))
    # The only normal-path scans of original FFN weights happen here at cold start.
    train_teacher_y = teacher_forward(train_x_raw, wg, wu, wd, args.batch_size)
    valid_teacher_y = teacher_forward(valid_x_raw, wg, wu, wd, args.batch_size)
    holdout_teacher_y = teacher_forward(holdout_x_raw, wg, wu, wd, args.batch_size)
    teacher_compile_seconds = time.perf_counter() - compile_start

    x_mu = seed_x_raw.mean(dim=0, keepdim=True)
    x_sigma = torch.clamp(seed_x_raw.std(dim=0, keepdim=True), min=1e-3)
    train_x = (train_x_raw - x_mu) / x_sigma
    valid_x = (valid_x_raw - x_mu) / x_sigma
    holdout_x = (holdout_x_raw - x_mu) / x_sigma
    hidden = train_x.shape[1]
    rows = []
    for rank in [int(item) for item in args.ranks.split(",") if item.strip()]:
        model, metrics = fit_rank(
            rank,
            train_x,
            train_teacher_y,
            valid_x,
            valid_teacher_y,
            holdout_x,
            holdout_teacher_y,
            holdout_capture_y,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.full_linear,
        )
        artifact = artifact_bytes(hidden, rank, args.full_linear)
        mac = runtime_mac(hidden, rank, args.full_linear)
        metrics["gpu_runtime_profile_decode_batch1"] = profile_runtime(model, holdout_x[:1], args.profile_repeats)
        metrics["artifact_fp16_bytes"] = artifact
        metrics["artifact_ratio_vs_original_q4"] = artifact / (wg_q4_bytes + wu_q4_bytes + wd_q4_bytes)
        metrics["runtime_mac_per_token"] = mac
        metrics["runtime_mac_per_host_input_byte"] = mac / (hidden * 2)
        metrics["runtime_original_ffn_weight_reads"] = 0
        rows.append({"rank": rank, **metrics})
        del model
        torch.cuda.empty_cache()

    device = torch.cuda.get_device_properties(0)
    result = {
        "experiment": "weight_distilled_crosschannel_swiglu",
        "formula": (
            "x'=normalize(x); y=Lx' + C[SiLU(A^T x') * (B^T x')] + b"
            if args.full_linear
            else "x'=normalize(x); y=C[SiLU(A^T x') * (B^T x')] + b"
        ),
        "runtime_contract": {
            "original_ffn_weight_reads": 0,
            "original_ffn_weight_use": "cold-start compilation and exact fallback only",
            "runtime_dynamic_input_bytes_if_x_is_on_gpu": 0,
            "runtime_dynamic_input_bytes_if_x_crosses_host_device": int(hidden * 2),
            "artifact_residency": "GPU-resident after cold start",
            "primary_metrics": ["dynamic bytes", "GPU MAC per dynamic byte", "GPU copy/sync wait"],
        },
        "cold_start": {
            "original_weight_scans": 3,
            "teacher_generation_seconds": teacher_compile_seconds,
            "real_seed_samples": int(len(seed_x_raw)),
            "synthetic_samples": args.synthetic_samples,
            "synthetic_distribution": "convex pairs of real layer inputs plus gaussian jitter",
            "noise_scale": args.noise_scale,
        },
        "layer": args.layer,
        "full_linear": args.full_linear,
        "dimensions": {"hidden": int(hidden), "ffn": int(wg.shape[0])},
        "original_q4_ffn_bytes": int(wg_q4_bytes + wu_q4_bytes + wd_q4_bytes),
        "weight_teacher_rel_l2_vs_captured_holdout": float(rel_l2(holdout_teacher_y, holdout_capture_y).mean()),
        "pinned_h2d_input_profile": profile_h2d_bytes(hidden, args.profile_repeats),
        "device": {"name": device.name, "total_memory_bytes": int(device.total_memory)},
        "rows": rows,
        "caveat": "Cold-start synthesis samples input neighborhoods but not full model trajectories. Holdout is required, and model-level evaluation remains future work.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
