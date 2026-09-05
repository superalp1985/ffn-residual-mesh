from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_polynomial_base_residual import load_layer


class CompiledCrossChannelSwiGLU(nn.Module):
    """Runtime artifact: no references to the original FFN Wg/Wu/Wd tensors."""

    def __init__(self, hidden: int, rank: int, full_linear: bool) -> None:
        super().__init__()
        self.full_linear = full_linear
        if full_linear:
            self.linear = nn.Linear(hidden, hidden)
        else:
            self.linear = None
        self.gate = nn.Linear(hidden, rank)
        self.up = nn.Linear(hidden, rank)
        self.down = nn.Linear(rank, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))
        base = self.linear(x) if self.linear is not None else 0.0
        return base + residual + self.bias


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target, dim=1) / torch.clamp(torch.linalg.vector_norm(target, dim=1), min=1e-6)


def artifact_bytes(hidden: int, rank: int, full_linear: bool) -> int:
    # fp16: input mean/scale, optional L+b, compact gate/up/down+bias.
    elements = 2 * hidden
    if full_linear:
        elements += hidden * hidden + hidden
    elements += 3 * hidden * rank + 2 * rank + hidden
    return elements * 2


def runtime_mac(hidden: int, rank: int, full_linear: bool) -> int:
    mac = 3 * hidden * rank
    if full_linear:
        mac += hidden * hidden
    return mac


def run_rank(
    rank: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    holdout_x: torch.Tensor,
    holdout_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    full_linear: bool,
) -> tuple[CompiledCrossChannelSwiGLU, dict[str, float]]:
    hidden = train_x.shape[1]
    model = CompiledCrossChannelSwiGLU(hidden, rank, full_linear).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    best_state = None
    best_valid = float("inf")
    stale = 0
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1000 + rank)
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=generator, device="cuda")
        for begin in range(0, train_x.shape[0], batch_size):
            indices = order[begin : begin + batch_size]
            pred = model(train_x[indices])
            # Relative scale stabilizes fitting across output dimensions.
            loss = torch.mean((pred - train_y[indices]) ** 2) / torch.mean(train_y[indices] ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            valid_error = float(rel_l2(model(valid_x), valid_y).mean().item())
        if valid_error < best_valid:
            best_valid = valid_error
            stale = 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 80:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_error = rel_l2(model(train_x), train_y)
        valid_error = rel_l2(model(valid_x), valid_y)
        holdout_error = rel_l2(model(holdout_x), holdout_y)
    elapsed_s = time.perf_counter() - start
    return model, {
        "epochs_completed": epoch + 1,
        "cold_compile_train_seconds": elapsed_s,
        "train_rel_l2": float(train_error.mean().item()),
        "validation_rel_l2": float(valid_error.mean().item()),
        "holdout_rel_l2": float(holdout_error.mean().item()),
        "holdout_rel_l2_p95": float(torch.quantile(holdout_error, 0.95).item()),
    }


def profile_runtime(model: nn.Module, x: torch.Tensor, repeats: int) -> dict[str, float]:
    for _ in range(20):
        model(x)
    torch.cuda.synchronize()
    elapsed = []
    with torch.no_grad():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            stop.record()
            stop.synchronize()
            elapsed.append(float(start.elapsed_time(stop)))
    return {
        "median_ms_per_batch": float(np.median(elapsed)),
        "p95_ms_per_batch": float(np.percentile(elapsed, 95)),
        "median_us_per_token": float(np.median(elapsed) * 1000 / x.shape[0]),
    }


def profile_h2d_bytes(hidden: int, repeats: int) -> dict[str, float]:
    """Measure only a pinned fp16 input packet transfer, not model scheduling."""
    source = torch.empty((1, hidden), dtype=torch.float16, pin_memory=True)
    destination = torch.empty((1, hidden), dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    for _ in range(20):
        with torch.cuda.stream(stream):
            destination.copy_(source, non_blocking=True)
    stream.synchronize()
    elapsed_us = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            destination.copy_(source, non_blocking=True)
            stop.record(stream)
        stop.synchronize()
        elapsed_us.append(float(start.elapsed_time(stop) * 1000))
    return {
        "bytes": int(hidden * 2),
        "median_us": float(np.median(elapsed_us)),
        "p95_us": float(np.percentile(elapsed_us, 95)),
        "scope": "pinned host-to-device copy only; excludes producer and model synchronization",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start compile a cross-channel SwiGLU surrogate without runtime Wg/Wu/Wd")
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--ranks", default="128,256")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--full-linear", action="store_true")
    parser.add_argument("--profile-batch", type=int, default=64)
    parser.add_argument("--profile-repeats", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this cold-start compiler experiment")
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_x_np, _, _, _, train_y_np = load_layer(args.calibration_root, args.layer)
    holdout_x_np, _, _, _, holdout_y_np = load_layer(args.holdout_root, args.layer)
    x_mu = train_x_np.mean(axis=0, keepdims=True).astype(np.float32)
    x_sigma = np.maximum(train_x_np.std(axis=0, keepdims=True).astype(np.float32), 1e-3)
    train_x_np = (train_x_np - x_mu) / x_sigma
    holdout_x_np = (holdout_x_np - x_mu) / x_sigma
    split = int(len(train_x_np) * 0.8)
    train_x = torch.from_numpy(train_x_np[:split]).cuda()
    train_y = torch.from_numpy(train_y_np[:split].astype(np.float32, copy=False)).cuda()
    valid_x = torch.from_numpy(train_x_np[split:]).cuda()
    valid_y = torch.from_numpy(train_y_np[split:].astype(np.float32, copy=False)).cuda()
    holdout_x = torch.from_numpy(holdout_x_np).cuda()
    holdout_y = torch.from_numpy(holdout_y_np.astype(np.float32, copy=False)).cuda()
    hidden = train_x.shape[1]
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    rows = []
    for rank in ranks:
        model, metrics = run_rank(
            rank,
            train_x,
            train_y,
            valid_x,
            valid_y,
            holdout_x,
            holdout_y,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.full_linear,
        )
        batch = min(args.profile_batch, holdout_x.shape[0])
        metrics["gpu_runtime_profile_decode_batch1"] = profile_runtime(model, holdout_x[:1], args.profile_repeats)
        metrics["gpu_runtime_profile_throughput_batch"] = profile_runtime(model, holdout_x[:batch], args.profile_repeats)
        artifact = artifact_bytes(hidden, rank, args.full_linear)
        mac = runtime_mac(hidden, rank, args.full_linear)
        # Artifact is resident after cold-start. The only dynamic input is x, normally already on GPU from the prior layer.
        metrics["artifact_fp16_bytes"] = artifact
        metrics["runtime_mac_per_token"] = mac
        # This is the user-facing dynamic-byte ratio. Artifact bytes are GPU-resident
        # after cold start and must not be mixed into a host/device transport metric.
        metrics["runtime_mac_per_host_input_byte"] = mac / (hidden * 2)
        metrics["runtime_mac_per_artifact_byte"] = mac / artifact
        metrics["runtime_original_ffn_weight_reads"] = 0
        rows.append({"rank": rank, **metrics})
        del model
        torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(0)
    result = {
        "experiment": "cold_compiled_crosschannel_swiglu",
        "formula": (
            "x'=normalize(x); y=Lx' + C[SiLU(A^T x') * (B^T x')] + b"
            if args.full_linear
            else "x'=normalize(x); y=C[SiLU(A^T x') * (B^T x')] + b"
        ),
        "runtime_contract": {
            "original_ffn_weight_reads": 0,
            "original_ffn_weight_use": "not used by this captured-data capacity probe; a production compiler may use them only at cold start and exact fallback",
            "runtime_dynamic_input_bytes_if_x_is_on_gpu": 0,
            "runtime_dynamic_input_bytes_if_x_crosses_host_device": int(hidden * 2),
            "artifact_residency": "GPU-resident after cold start",
            "primary_metrics": ["dynamic bytes", "GPU MAC per dynamic byte", "GPU copy/sync wait"],
        },
        "layer": args.layer,
        "train_samples": int(train_x.shape[0]),
        "validation_samples": int(valid_x.shape[0]),
        "holdout_samples": int(holdout_x.shape[0]),
        "hidden": int(hidden),
        "full_linear": args.full_linear,
        "normalization_fp16_bytes": int(x_mu.size * 2 + x_sigma.size * 2),
        "pinned_h2d_input_profile": profile_h2d_bytes(hidden, args.profile_repeats),
        "device": {"name": properties.name, "total_memory_bytes": int(properties.total_memory)},
        "rows": rows,
        "caveat": (
            "This is a captured-data-fitted surrogate capacity probe, not yet a compiler distilled directly from Wg/Wu/Wd. "
            "It does not export a production CUDA kernel, does not measure full-model scheduling, and requires larger "
            "calibration coverage before model-quality claims."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
