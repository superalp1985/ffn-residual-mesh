from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from gguf import GGUFReader
from gguf.quants import dequantize

from benchmark_exact_radix_split_pipeline import (
    direct_group_dots,
    evaluate_radix_table,
    load_q4_projection,
    projection_from_group_dots,
    quantize_groupwise_q8,
)


def load_down_weight(model: Path, layer: int) -> np.ndarray:
    """Cold-start load only the resident down projection needed by this bridge."""
    reader = GGUFReader(str(model))
    tensor = next(item for item in reader.tensors if item.name == f"blk.{layer}.ffn_down.weight")
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)


@triton.jit
def residual_kernel(
    packed_ptr,
    alpha_ptr,
    z_ptr,
    scale_ptr,
    partial_ptr,
    packed_row_bytes: tl.constexpr,
    alpha_row_floats: tl.constexpr,
    alpha_offset_floats: tl.constexpr,
    groups: tl.constexpr,
    chunks: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * block_n + tl.arange(0, block_n)
    mask = offsets < chunks * block_n
    packed_value = tl.load(packed_ptr + row * packed_row_bytes + offsets // 4, mask=mask, other=0).to(tl.uint32)
    q = (packed_value >> ((offsets & 3) * 2)) & 3
    group = offsets // 32
    alpha = tl.load(
        alpha_ptr + row * alpha_row_floats + alpha_offset_floats + group,
        mask=mask,
        other=0.0,
    )
    z = tl.load(z_ptr + offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + group, mask=mask, other=0.0)
    contribution = q.to(tl.float32) * alpha * z * scale
    tl.store(partial_ptr + row * chunks + chunk, tl.sum(tl.where(mask, contribution, 0.0), axis=0))


@triton.jit
def swiglu_merge_kernel(gate_base, up_base, gate_residual, up_residual, output, hidden: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < hidden
    gate = tl.load(gate_base + offsets, mask=mask, other=0.0) + tl.load(gate_residual + offsets, mask=mask, other=0.0)
    up = tl.load(up_base + offsets, mask=mask, other=0.0) + tl.load(up_residual + offsets, mask=mask, other=0.0)
    silu = gate * tl.sigmoid(gate)
    tl.store(output + offsets, silu * up, mask=mask)


def load_base(table_dir: Path, projection: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((table_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = int(manifest["projections"][projection]["rows"])
    groups = int(manifest["projections"][projection]["groups"])
    blocks = int(manifest["projections"][projection]["blocks"])
    state_count = int(manifest["projections"][projection]["state_count"])
    block_size = int(manifest["block_size"])
    table = np.fromfile(table_dir / f"{projection}.table.u8.bin", dtype=np.uint8).reshape(blocks, state_count, rows)
    high_sum = np.fromfile(table_dir / f"{projection}.high_sum.i16.bin", dtype="<i2").reshape(rows, groups)
    alpha = np.fromfile(table_dir / f"{projection}.alpha.f32.bin", dtype="<f4").reshape(rows, groups)
    beta = np.fromfile(table_dir / f"{projection}.beta.f32.bin", dtype="<f4").reshape(rows, groups)
    states = np.fromfile(table_dir / "states.u16.bin", dtype="<u2").reshape(4, blocks)
    z = np.fromfile(table_dir / "activation.z.i8.bin", dtype="i1").reshape(1, -1)
    scales = np.fromfile(table_dir / "activation.scale.f32.bin", dtype="<f4").reshape(1, -1)
    high_dot = evaluate_radix_table(table, high_sum, states, 32 // block_size)
    base = projection_from_group_dots(high_dot, z, scales, alpha, beta, code_multiplier=4)
    return base.astype(np.float32, copy=False), z.reshape(-1), scales.reshape(-1)


def pack_payload(artifact_dir: Path, projection: str, rows: int, groups: int) -> np.ndarray:
    codes = np.fromfile(artifact_dir / f"{projection}.qlo2.rowpacked.bin", dtype=np.uint8).reshape(rows, -1)
    alpha = np.fromfile(artifact_dir / f"{projection}.alpha.f32.bin", dtype="<f4").reshape(rows, groups)
    return np.concatenate((codes, alpha.view(np.uint8)), axis=1).copy()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    table_manifest = json.loads((args.table_artifact / "manifest.json").read_text(encoding="utf-8"))
    rows = int(table_manifest["projections"]["gate"]["rows"])
    hidden = int(table_manifest["projections"]["gate"]["hidden"])
    groups = int(table_manifest["projections"]["gate"]["groups"])
    if rows != 6144 or hidden != 2048 or groups != 64:
        raise ValueError("the first prototype expects layer-23 Qwen3.5 dimensions")

    gate_base, z, scales = load_base(args.table_artifact, "gate")
    up_base, _, _ = load_base(args.table_artifact, "up")
    gate_payload = pack_payload(args.residual_artifact, "gate", rows, groups)
    up_payload = pack_payload(args.residual_artifact, "up", rows, groups)
    host_gate = torch.from_numpy(gate_payload).pin_memory()
    host_up = torch.from_numpy(up_payload).pin_memory()
    device_gate = torch.empty_like(host_gate, device="cuda")
    device_up = torch.empty_like(host_up, device="cuda")
    device_z = torch.from_numpy(np.ascontiguousarray(z)).cuda()
    device_scales = torch.from_numpy(np.ascontiguousarray(scales)).cuda()
    device_gate_base = torch.from_numpy(gate_base).cuda()
    device_up_base = torch.from_numpy(up_base).cuda()
    chunks = hidden // 256
    device_gate_residual = torch.empty(rows, dtype=torch.float32, device="cuda")
    device_up_residual = torch.empty(rows, dtype=torch.float32, device="cuda")
    device_gate_partial = torch.empty((rows, chunks), dtype=torch.float32, device="cuda")
    device_up_partial = torch.empty((rows, chunks), dtype=torch.float32, device="cuda")
    device_hidden = torch.empty(rows, dtype=torch.float32, device="cuda")
    down = load_down_weight(args.model, args.layer)
    device_down = torch.from_numpy(np.ascontiguousarray(down.astype(np.float16, copy=False))).cuda()
    device_down_output = torch.empty((1, hidden), dtype=torch.float16, device="cuda")
    grid = (rows, chunks)
    merge_grid = ((rows + 255) // 256,)

    # Keep code and alpha contiguous in one packet. The float view retains the
    # full packet row stride, so the kernel receives that stride explicitly.
    code_bytes = hidden // 4
    packet_row_floats = (code_bytes + groups * 4) // 4
    def launch_residual(packet: torch.Tensor, partial: torch.Tensor, output: torch.Tensor) -> None:
        residual_kernel[grid](
            packet,
            packet.view(torch.float32),
            device_z,
            device_scales,
            partial,
            packed_row_bytes=code_bytes + groups * 4,
            alpha_row_floats=packet_row_floats,
            alpha_offset_floats=code_bytes // 4,
            groups=groups,
            chunks=chunks,
            block_n=256,
            num_warps=4,
        )
        torch.sum(partial, dim=1, out=output)

    # Warm up Triton and cuBLAS kernels before collecting samples.
    for _ in range(3):
        device_gate.copy_(host_gate)
        device_up.copy_(host_up)
        launch_residual(device_gate, device_gate_partial, device_gate_residual)
        launch_residual(device_up, device_up_partial, device_up_residual)
        swiglu_merge_kernel[merge_grid](device_gate_base, device_up_base, device_gate_residual, device_up_residual, device_hidden, hidden=rows)
        torch.mm(device_hidden.half().unsqueeze(0), device_down.t(), out=device_down_output)
    torch.cuda.synchronize()

    def measure(full: bool) -> dict[str, float]:
        copy_samples: list[float] = []
        total_samples: list[float] = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            copy_stop = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            device_gate.copy_(host_gate, non_blocking=True)
            device_up.copy_(host_up, non_blocking=True)
            copy_stop.record()
            launch_residual(device_gate, device_gate_partial, device_gate_residual)
            launch_residual(device_up, device_up_partial, device_up_residual)
            if full:
                swiglu_merge_kernel[merge_grid](device_gate_base, device_up_base, device_gate_residual, device_up_residual, device_hidden, hidden=rows)
                torch.mm(device_hidden.half().unsqueeze(0), device_down.t(), out=device_down_output)
            stop.record()
            stop.synchronize()
            copy_samples.append(float(start.elapsed_time(copy_stop)))
            total_samples.append(float(start.elapsed_time(stop)))
        total = statistics.median(total_samples)
        copy = statistics.median(copy_samples)
        return {"copy_ms": copy, "compute_ms": total - copy, "critical_ms": total}

    residual_only = measure(False)
    full_path = measure(True)
    down_flops = 2 * rows * hidden
    residual_flops = 2 * rows * hidden * 4
    merge_flops = rows * 8
    full_flops = residual_flops + merge_flops + down_flops
    payload_bytes = int(gate_payload.nbytes + up_payload.nbytes)

    # Correctness check after a full launch.
    torch.cuda.synchronize()
    gate_residual = device_gate_residual.detach().cpu().numpy().copy()
    up_residual = device_up_residual.detach().cpu().numpy().copy()
    x_q = z.astype(np.float32) * scales.repeat(32)
    gate_codes, gate_alpha, gate_beta, _ = load_q4_projection(args.model, args.layer, "gate")
    up_codes, up_alpha, up_beta, _ = load_q4_projection(args.model, args.layer, "up")
    gate_ref = projection_from_group_dots(direct_group_dots(gate_codes & 3, z), z.reshape(1, -1), scales.reshape(1, -1), gate_alpha, np.zeros_like(gate_alpha), 1)
    up_ref = projection_from_group_dots(direct_group_dots(up_codes & 3, z), z.reshape(1, -1), scales.reshape(1, -1), up_alpha, np.zeros_like(up_alpha), 1)
    gate_err = float(np.linalg.norm(gate_residual - gate_ref) / max(np.linalg.norm(gate_ref), 1e-12))
    up_err = float(np.linalg.norm(up_residual - up_ref) / max(np.linalg.norm(up_ref), 1e-12))
    gate_merged = gate_base + gate_residual
    up_merged = up_base + up_residual
    gate_full_ref = projection_from_group_dots(
        direct_group_dots(gate_codes, z),
        z.reshape(1, -1),
        scales.reshape(1, -1),
        gate_alpha,
        gate_beta,
        1,
    )
    up_full_ref = projection_from_group_dots(
        direct_group_dots(up_codes, z),
        z.reshape(1, -1),
        scales.reshape(1, -1),
        up_alpha,
        up_beta,
        1,
    )
    gate_merged_err = float(np.linalg.norm(gate_merged - gate_full_ref) / max(np.linalg.norm(gate_full_ref), 1e-12))
    up_merged_err = float(np.linalg.norm(up_merged - up_full_ref) / max(np.linalg.norm(up_full_ref), 1e-12))
    swiglu_ref = (gate_full_ref / (1.0 + np.exp(-gate_full_ref))) * up_full_ref
    hidden_gpu = device_hidden.detach().cpu().numpy().copy()
    down_gpu = device_down_output.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    down_ref = swiglu_ref @ down.T
    swiglu_err = float(np.linalg.norm(hidden_gpu - swiglu_ref) / max(np.linalg.norm(swiglu_ref), 1e-12))
    down_err = float(np.linalg.norm(down_gpu - down_ref) / max(np.linalg.norm(down_ref), 1e-12))
    result = {
        "experiment": "gpu_full_ffn_compute_density_bridge",
        "layer": args.layer,
        "dimensions": {"hidden": hidden, "ffn": rows},
        "down_weight": {"format": "resident fp16 dequantized Q4_K", "bytes": int(device_down.numel() * 2)},
        "h2d": {"residual_payload_bytes": payload_bytes, "residual_payload_mib": payload_bytes / 2**20},
        "flop_proxy": {
            "residual_only": residual_flops,
            "full_gate_swiglu_down": full_flops,
            "residual_flops_per_h2d_byte": residual_flops / payload_bytes,
            "full_flops_per_h2d_byte": full_flops / payload_bytes,
        },
        "residual_only": residual_only,
        "full_gate_swiglu_down": full_path,
        "density_gain_compute_time": full_path["compute_ms"] / max(residual_only["compute_ms"], 1e-12),
        "correctness": {
            "gate_residual_rel_l2": gate_err,
            "up_residual_rel_l2": up_err,
            "gate_residual_finite": bool(np.isfinite(gate_residual).all()),
            "up_residual_finite": bool(np.isfinite(up_residual).all()),
            "gate_residual_max_abs": float(np.nanmax(np.abs(gate_residual))),
            "up_residual_max_abs": float(np.nanmax(np.abs(up_residual))),
            "gate_merged_rel_l2": gate_merged_err,
            "up_merged_rel_l2": up_merged_err,
            "swiglu_rel_l2": swiglu_err,
            "down_output_rel_l2_fp16": down_err,
        },
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark residual-only versus full gate/SwiGLU/down GPU work")
    parser.add_argument("model", type=Path)
    parser.add_argument("table_artifact", type=Path)
    parser.add_argument("residual_artifact", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
