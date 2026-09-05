from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_compiled_crosschannel_swiglu import profile_h2d_bytes, rel_l2
from evaluate_preexpanded_sparse_cp import load_weights
from evaluate_polynomial_base_residual import load_layer


class LocalFormula(nn.Module):
    """One table entry: base output plus local linear and nonlinear residuals."""

    def __init__(self, hidden: int, rank: int, base: torch.Tensor) -> None:
        super().__init__()
        self.rank = rank
        self.base = nn.Parameter(base.clone())
        # These three projections are CPU/RAM-side factors in the intended runtime.
        self.linear_in = nn.Linear(hidden, rank, bias=False)
        self.gate = nn.Linear(hidden, rank)
        self.up = nn.Linear(hidden, rank)
        # These two matrices and base are GPU-resident in the intended runtime.
        self.linear_out = nn.Linear(rank, hidden, bias=False)
        self.down = nn.Linear(rank, hidden, bias=False)

    def cpu_features(self, delta: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.linear_in(delta), self.gate(delta), self.up(delta)), dim=1)

    def merge(self, packed: torch.Tensor) -> torch.Tensor:
        p, g, u = torch.split(packed, self.rank, dim=1)
        return self.base + self.linear_out(p) + self.down(torch.nn.functional.silu(g) * u)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.merge(self.cpu_features(delta))


def teacher_forward(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    with torch.no_grad():
        for begin in range(0, len(x), batch_size):
            chunk = x[begin : begin + batch_size]
            parts.append((torch.nn.functional.silu(chunk @ wg.T) * (chunk @ wu.T)) @ wd.T)
    return torch.cat(parts)


def assign_clusters(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    # Squared Euclidean distance without materializing a [tokens, K, hidden] tensor.
    distance = x.square().sum(dim=1, keepdim=True) - 2.0 * (x @ centers.T) + centers.square().sum(dim=1)
    return distance.argmin(dim=1)


def fit_router_kmeans(x: torch.Tensor, clusters: int, seed: int, iterations: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-K GPU Lloyd iteration used only during cold-start compilation."""
    clusters = min(clusters, len(x))
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    centers = x[torch.randperm(len(x), device="cuda", generator=generator)[:clusters]].clone()
    for _ in range(iterations):
        code = assign_clusters(x, centers)
        updated = torch.zeros_like(centers)
        updated.index_add_(0, code, x)
        counts = torch.bincount(code, minlength=clusters).to(x.dtype).unsqueeze(1)
        replacement = x[torch.randint(len(x), (clusters,), device="cuda", generator=generator)]
        next_centers = torch.where(counts > 0, updated / counts.clamp_min(1.0), replacement)
        if torch.max(torch.abs(next_centers - centers)) < 1e-5:
            centers = next_centers
            break
        centers = next_centers
    return centers, assign_clusters(x, centers)


def synthesize_local_inputs(members: torch.Tensor, count: int, noise_scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    indexes = torch.randint(len(members), (count,), device="cuda", generator=generator)
    local_std = torch.clamp(members.std(dim=0, keepdim=True), min=1e-4)
    noise = torch.randn((count, members.shape[1]), device="cuda", generator=generator) * local_std * noise_scale
    return members[indexes] + noise


def profile_merge(model: LocalFormula, packed: torch.Tensor, repeats: int) -> dict[str, float]:
    for _ in range(20):
        model.merge(packed)
    torch.cuda.synchronize()
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            model.merge(packed)
            stop.record()
            stop.synchronize()
            timings.append(float(start.elapsed_time(stop)))
    return {
        "median_us_per_token": float(np.median(timings) * 1000 / len(packed)),
        "p95_us_per_token": float(np.percentile(timings, 95) * 1000 / len(packed)),
    }


def train_entry(
    delta: torch.Tensor,
    target: torch.Tensor,
    rank: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[LocalFormula, dict[str, float]]:
    split = max(1, int(len(delta) * 0.85))
    train_delta, valid_delta = delta[:split], delta[split:]
    train_target, valid_target = target[:split], target[split:]
    base = train_target.mean(dim=0)
    model = LocalFormula(delta.shape[1], rank, base).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_valid = float("inf")
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_delta), device="cuda", generator=generator)
        for begin in range(0, len(train_delta), batch_size):
            rows = order[begin : begin + batch_size]
            pred = model(train_delta[rows])
            loss = torch.mean((pred - train_target[rows]) ** 2) / torch.clamp(torch.mean(train_target[rows] ** 2), min=1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            valid_error = float(rel_l2(model(valid_delta), valid_target).mean()) if len(valid_delta) else 0.0
        if valid_error < best_valid:
            best_valid = valid_error
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_error = rel_l2(model(train_delta), train_target)
        valid_error = rel_l2(model(valid_delta), valid_target) if len(valid_delta) else torch.zeros(1, device="cuda")
    return model, {
        "epochs_completed": epoch + 1,
        "train_rel_l2_vs_weight_teacher": float(train_error.mean()),
        "validation_rel_l2_vs_weight_teacher": float(valid_error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile per-state FFN formula tables directly from Wg/Wu/Wd")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("holdout_root", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--synthetic-per-cluster", type=int, default=512)
    parser.add_argument("--noise-scale", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
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
    calibration_x_np, _, _, _, _ = load_layer(args.calibration_root, args.layer)
    holdout_x_np, _, _, _, holdout_capture_y_np = load_layer(args.holdout_root, args.layer)
    calibration_x = torch.from_numpy(calibration_x_np.astype(np.float32, copy=False)).cuda()
    holdout_x = torch.from_numpy(holdout_x_np.astype(np.float32, copy=False)).cuda()
    holdout_capture_y = torch.from_numpy(holdout_capture_y_np.astype(np.float32, copy=False)).cuda()
    input_mean = calibration_x.mean(dim=0, keepdim=True)
    input_scale = torch.clamp(calibration_x.std(dim=0, keepdim=True), min=1e-3)
    router_centers, calibration_code = fit_router_kmeans((calibration_x - input_mean) / input_scale, args.clusters, 3456)
    holdout_code = assign_clusters((holdout_x - input_mean) / input_scale, router_centers)
    centers = router_centers * input_scale + input_mean
    (wg_np, wg_q4_bytes), (wu_np, wu_q4_bytes), (wd_np, wd_q4_bytes) = load_weights(args.model, args.layer)
    wg, wu, wd = torch.from_numpy(wg_np).cuda(), torch.from_numpy(wu_np).cuda(), torch.from_numpy(wd_np).cuda()

    start = time.perf_counter()
    models: list[LocalFormula] = []
    entry_rows = []
    for cluster in range(len(router_centers)):
        members = calibration_x[calibration_code == cluster]
        synthetic = synthesize_local_inputs(members, args.synthetic_per_cluster, args.noise_scale, 9000 + cluster)
        compile_x = torch.cat((members, synthetic))
        # This is the cold-start-only original-weight scan for this table entry.
        compile_y = teacher_forward(compile_x, wg, wu, wd, args.batch_size)
        delta = (compile_x - centers[cluster]) / input_scale
        model, train_metrics = train_entry(
            delta, compile_y, args.rank, args.epochs, args.batch_size, args.lr, args.weight_decay, 10000 + cluster
        )
        models.append(model)
        entry_rows.append({"cluster": cluster, "real_seed_samples": int(len(members)), "compiled_samples": int(len(compile_x)), **train_metrics})
    cold_compile_seconds = time.perf_counter() - start
    holdout_teacher_y = teacher_forward(holdout_x, wg, wu, wd, args.batch_size)

    prediction = torch.empty_like(holdout_teacher_y)
    profiles = []
    with torch.no_grad():
        for cluster, model in enumerate(models):
            index = torch.nonzero(holdout_code == cluster, as_tuple=False).squeeze(1)
            if len(index) == 0:
                continue
            delta = (holdout_x[index] - centers[cluster]) / input_scale
            prediction[index] = model(delta)
            packed = model.cpu_features(delta[:1]).detach()
            profiles.append({
                "cluster": cluster,
                "merge_decode_batch1": profile_merge(model, packed, args.profile_repeats),
                "merge_throughput_batch": profile_merge(model, model.cpu_features(delta[: min(len(delta), 32)]).detach(), args.profile_repeats),
            })

    hidden = calibration_x.shape[1]
    k = len(models)
    rank = args.rank
    cpu_factor_bytes = k * 3 * hidden * rank * 2 + k * 2 * rank * 2
    gpu_factor_bytes = k * (2 * hidden * rank * 2 + hidden * 2)
    router_bytes = k * hidden * 2
    h2d_feature_bytes = 3 * rank * 2 + 2  # fp16 p/g/u plus uint16 state code
    gpu_merge_mac = 2 * hidden * rank
    dynamic_if_cpu_routes = hidden * 2 + h2d_feature_bytes
    original_bytes = wg_q4_bytes + wu_q4_bytes + wd_q4_bytes
    result = {
        "experiment": "state_table_local_nonlinear_formula",
        "formula": "entry[k]: delta=(x-center[k])/sigma; y=base[k]+U[k](V[k]^T delta)+C[k](SiLU(A[k]^T delta+a[k])*(B[k]^T delta+b[k]))",
        "runtime_contract": {
            "original_ffn_weight_reads": 0,
            "original_ffn_weight_use": "cold-start compilation and exact fallback only",
            "cpu": "route k and compute p=V^Tdelta, g=A^Tdelta+a, u=B^Tdelta+b from the selected table entry",
            "gpu": "lookup resident base/U/C by k, then merge compact p/g/u",
            "runtime_dynamic_bytes_if_cpu_routes": {"d2h_ffn_input_fp16": hidden * 2, "h2d_packed_features": h2d_feature_bytes, "total": dynamic_if_cpu_routes},
            "runtime_dynamic_bytes_if_router_is_gpu_resident": {"h2d": 0, "d2h": 0},
        },
        "cold_start": {
            "original_weight_scans": "one teacher evaluation per cold-start synthetic batch; no runtime scans",
            "seconds": cold_compile_seconds,
            "synthetic_distribution": "real-state cluster members plus local gaussian jitter",
            "synthetic_per_cluster": args.synthetic_per_cluster,
            "noise_scale": args.noise_scale,
        },
        "layer": args.layer,
        "clusters": k,
        "rank": rank,
        "entries": entry_rows,
        "holdout": {
            "rel_l2_vs_weight_teacher": float(rel_l2(prediction, holdout_teacher_y).mean()),
            "rel_l2_p95_vs_weight_teacher": float(torch.quantile(rel_l2(prediction, holdout_teacher_y), 0.95)),
            "rel_l2_vs_captured_ffn": float(rel_l2(prediction, holdout_capture_y).mean()),
            "rel_l2_p95_vs_captured_ffn": float(torch.quantile(rel_l2(prediction, holdout_capture_y), 0.95)),
            "tokens_per_entry": {str(cluster): int((holdout_code == cluster).sum()) for cluster in range(k)},
        },
        "artifact_fp16_bytes": {
            "cpu_projection_factors": cpu_factor_bytes,
            "gpu_merge_factors_and_bases": gpu_factor_bytes,
            "router_centers": router_bytes,
            "total": cpu_factor_bytes + gpu_factor_bytes + router_bytes,
            "ratio_vs_original_q4_ffn": (cpu_factor_bytes + gpu_factor_bytes + router_bytes) / original_bytes,
        },
        "work_and_wait": {
            "cpu_projection_mac_per_token": 3 * hidden * rank,
            "gpu_merge_mac_per_token": gpu_merge_mac,
            "gpu_mac_per_h2d_feature_byte": gpu_merge_mac / h2d_feature_bytes,
            "gpu_mac_per_total_cpu_routed_dynamic_byte": gpu_merge_mac / dynamic_if_cpu_routes,
            "pinned_h2d_feature_profile": profile_h2d_bytes(3 * rank + 1, args.profile_repeats),
            "gpu_merge_profiles": profiles,
            "scope": "copy and merge are measured separately; full graph overlap/synchronization is not yet measured",
        },
        "weight_teacher_rel_l2_vs_captured_holdout": float(rel_l2(holdout_teacher_y, holdout_capture_y).mean()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
