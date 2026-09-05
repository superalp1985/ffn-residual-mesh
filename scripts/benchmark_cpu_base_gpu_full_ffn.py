from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from benchmark_exact_radix_split_pipeline import (
    direct_group_dots,
    load_q4_projection,
    projection_from_group_dots,
)
from benchmark_gpu_full_ffn_density import load_base, load_down_weight, pack_payload


def load_radix_projection(table_dir: Path, projection: str) -> dict[str, object]:
    manifest = json.loads((table_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["projections"][projection]
    rows = int(entry["rows"])
    groups = int(entry["groups"])
    blocks = int(entry["blocks"])
    state_count = int(entry["state_count"])
    table = np.fromfile(table_dir / f"{projection}.table.u8.bin", dtype=np.uint8).reshape(blocks, state_count, rows)
    high_sum = np.fromfile(table_dir / f"{projection}.high_sum.i16.bin", dtype="<i2").reshape(rows, groups)
    alpha = np.fromfile(table_dir / f"{projection}.alpha.f32.bin", dtype="<f4").reshape(rows, groups)
    beta = np.fromfile(table_dir / f"{projection}.beta.f32.bin", dtype="<f4").reshape(rows, groups)
    states = np.fromfile(table_dir / "states.u16.bin", dtype="<u2").reshape(4, blocks)
    z = np.fromfile(table_dir / "activation.z.i8.bin", dtype="i1")
    scales = np.fromfile(table_dir / "activation.scale.f32.bin", dtype="<f4")
    block_size = int(manifest["block_size"])
    z_sum = z.reshape(groups, 32).astype(np.int32).sum(axis=1)
    return {
        "table": table,
        "high_sum": high_sum,
        "alpha": alpha,
        "beta": beta,
        "states": states,
        "scales": scales,
        "z_sum": z_sum,
        "blocks_per_group": 32 // block_size,
        "groups": groups,
        "rows": rows,
    }


def evaluate_base_tile(artifact: dict[str, object], begin: int, end: int) -> np.ndarray:
    table = artifact["table"]
    high_sum = artifact["high_sum"]
    alpha = artifact["alpha"]
    beta = artifact["beta"]
    states = artifact["states"]
    scales = artifact["scales"]
    z_sum = artifact["z_sum"]
    blocks_per_group = int(artifact["blocks_per_group"])
    groups = int(artifact["groups"])
    rows = end - begin
    group_dot = np.zeros((groups, rows), dtype=np.int32)
    for digit in range(4):
        radix = 1 << (2 * digit)
        for block in range(table.shape[0]):
            group = block // blocks_per_group
            state = int(states[digit, block])
            group_dot[group] += radix * table[block, state, begin:end].astype(np.int32, copy=False)
    group_dot -= 128 * high_sum[begin:end].T.astype(np.int32, copy=False)
    return np.sum(
        scales[None, :] * (alpha[begin:end] * (4 * group_dot.T) + beta[begin:end] * z_sum[None, :]),
        axis=1,
        dtype=np.float32,
    )


@triton.jit
def residual_direct_kernel(
    packed_ptr,
    alpha_ptr,
    z_ptr,
    scale_ptr,
    output_ptr,
    packed_row_bytes: tl.constexpr,
    alpha_row_floats: tl.constexpr,
    alpha_offset_floats: tl.constexpr,
    hidden: tl.constexpr,
    groups: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, hidden, block_n):
        offsets = start + tl.arange(0, block_n)
        group = offsets // 32
        packed_value = tl.load(packed_ptr + row * packed_row_bytes + offsets // 4).to(tl.uint32)
        q = (packed_value >> ((offsets & 3) * 2)) & 3
        alpha = tl.load(alpha_ptr + row * alpha_row_floats + alpha_offset_floats + group)
        z = tl.load(z_ptr + offsets).to(tl.float32)
        scale = tl.load(scale_ptr + group)
        acc += tl.sum(q.to(tl.float32) * alpha * z * scale, axis=0)
    tl.store(output_ptr + row, acc)


@triton.jit
def tile_swiglu_kernel(
    gate_base_ptr,
    up_base_ptr,
    gate_residual_ptr,
    up_residual_ptr,
    hidden_ptr,
    gate_merged_ptr,
    up_merged_ptr,
    tile_rows: tl.constexpr,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < tile_rows
    gate = tl.load(gate_base_ptr + offsets, mask=mask, other=0.0)
    gate += tl.load(gate_residual_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_base_ptr + offsets, mask=mask, other=0.0)
    up += tl.load(up_residual_ptr + offsets, mask=mask, other=0.0)
    tl.store(gate_merged_ptr + offsets, gate, mask=mask)
    tl.store(up_merged_ptr + offsets, up, mask=mask)
    tl.store(hidden_ptr + offsets, gate * tl.sigmoid(gate) * up, mask=mask)


@triton.jit
def tile_swiglu_down_kernel(
    gate_base_ptr,
    up_base_ptr,
    gate_residual_ptr,
    up_residual_ptr,
    down_ptr,
    partial_ptr,
    tile_index,
    tile_rows: tl.constexpr,
    ffn_dim: tl.constexpr,
    out_dim: tl.constexpr,
    block_m: tl.constexpr,
):
    out_index = tl.program_id(0)
    if out_index >= out_dim:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, tile_rows, block_m):
        row_offsets = start + tl.arange(0, block_m)
        row_mask = row_offsets < tile_rows
        gate = tl.load(gate_base_ptr + row_offsets, mask=row_mask, other=0.0)
        gate += tl.load(gate_residual_ptr + row_offsets, mask=row_mask, other=0.0)
        up = tl.load(up_base_ptr + row_offsets, mask=row_mask, other=0.0)
        up += tl.load(up_residual_ptr + row_offsets, mask=row_mask, other=0.0)
        hidden = gate * tl.sigmoid(gate) * up
        weights = tl.load(
            down_ptr + out_index * ffn_dim + tile_index * tile_rows + row_offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(weights * hidden, axis=0)
    tl.store(partial_ptr + tile_index * out_dim + out_index, acc)


def make_payload_tensor(payload: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(payload)).pin_memory()


def make_base_tensor(base: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(base.astype(np.float32, copy=False))).pin_memory()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = json.loads((args.table_artifact / "manifest.json").read_text(encoding="utf-8"))
    rows = int(manifest["projections"]["gate"]["rows"])
    hidden = int(manifest["projections"]["gate"]["hidden"])
    groups = int(manifest["projections"]["gate"]["groups"])
    if (rows, hidden, groups) != (6144, 2048, 64):
        raise ValueError("the first prototype expects layer-23 Qwen3.5 dimensions")
    if rows % args.tile_rows:
        raise ValueError("rows must be divisible by tile_rows")

    gate_base, z, scales = load_base(args.table_artifact, "gate")
    up_base, _, _ = load_base(args.table_artifact, "up")
    gate_table = load_radix_projection(args.table_artifact, "gate") if args.base_mode == "table" else None
    up_table = load_radix_projection(args.table_artifact, "up") if args.base_mode == "table" else None
    gate_payload = pack_payload(args.residual_artifact, "gate", rows, groups)
    up_payload = pack_payload(args.residual_artifact, "up", rows, groups)
    down = load_down_weight(args.model, args.layer)
    if down.shape != (hidden, rows):
        raise ValueError(f"unexpected down shape {down.shape}")

    tile_count = rows // args.tile_rows
    code_bytes = hidden // 4
    packet_row_bytes = code_bytes + groups * 4
    packet_row_floats = packet_row_bytes // 4
    host_gate = make_payload_tensor(gate_payload)
    host_up = make_payload_tensor(up_payload)
    host_gate_base = make_base_tensor(gate_base) if args.base_mode == "preexpanded" else None
    host_up_base = make_base_tensor(up_base) if args.base_mode == "preexpanded" else None
    host_gate_base_slots = [torch.empty(args.tile_rows, dtype=torch.float32, pin_memory=True) for _ in range(2)]
    host_up_base_slots = [torch.empty(args.tile_rows, dtype=torch.float32, pin_memory=True) for _ in range(2)]
    device_z = torch.from_numpy(np.ascontiguousarray(z)).cuda()
    device_scales = torch.from_numpy(np.ascontiguousarray(scales)).cuda()
    device_down = torch.from_numpy(np.ascontiguousarray(down.astype(np.float16, copy=False))).cuda()

    packet_gate = [torch.empty((args.tile_rows, packet_row_bytes), dtype=torch.uint8, device="cuda") for _ in range(2)]
    packet_up = [torch.empty((args.tile_rows, packet_row_bytes), dtype=torch.uint8, device="cuda") for _ in range(2)]
    base_gate = [torch.empty(args.tile_rows, dtype=torch.float32, device="cuda") for _ in range(2)]
    base_up = [torch.empty(args.tile_rows, dtype=torch.float32, device="cuda") for _ in range(2)]
    residual_gate = [torch.empty(args.tile_rows, dtype=torch.float32, device="cuda") for _ in range(2)]
    residual_up = [torch.empty(args.tile_rows, dtype=torch.float32, device="cuda") for _ in range(2)]
    down_partial = torch.empty((tile_count, hidden), dtype=torch.float32, device="cuda")
    hidden_tiles = torch.empty((tile_count, args.tile_rows), dtype=torch.float32, device="cuda")
    gate_merged_tiles = torch.empty((tile_count, args.tile_rows), dtype=torch.float32, device="cuda")
    up_merged_tiles = torch.empty((tile_count, args.tile_rows), dtype=torch.float32, device="cuda")
    gate_residual_tiles = torch.empty((tile_count, args.tile_rows), dtype=torch.float32, device="cuda")
    up_residual_tiles = torch.empty((tile_count, args.tile_rows), dtype=torch.float32, device="cuda")
    down_output = torch.empty(hidden, dtype=torch.float32, device="cuda")
    down_grid = (hidden,)
    cpu_pool = (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.cpu_threads)
        if args.base_mode == "table"
        else None
    )

    def launch_residual(packet: torch.Tensor, output: torch.Tensor) -> None:
        residual_direct_kernel[(args.tile_rows,)](
            packet,
            packet.view(torch.float32),
            device_z,
            device_scales,
            output,
            packed_row_bytes=packet_row_bytes,
            alpha_row_floats=packet_row_floats,
            alpha_offset_floats=code_bytes // 4,
            hidden=hidden,
            groups=groups,
            block_n=256,
            num_warps=4,
        )

    def enqueue_tile(slot: int, tile: int, compute_stream: torch.cuda.Stream, collect_debug: bool) -> None:
        with torch.cuda.stream(compute_stream):
            launch_residual(packet_gate[slot], residual_gate[slot])
            launch_residual(packet_up[slot], residual_up[slot])
            if collect_debug:
                gate_residual_tiles[tile].copy_(residual_gate[slot])
                up_residual_tiles[tile].copy_(residual_up[slot])
                tile_swiglu_kernel[((args.tile_rows + 255) // 256,)](
                    base_gate[slot],
                    base_up[slot],
                    residual_gate[slot],
                    residual_up[slot],
                    hidden_tiles[tile],
                    gate_merged_tiles[tile],
                    up_merged_tiles[tile],
                    tile_rows=args.tile_rows,
                )
            tile_swiglu_down_kernel[down_grid](
                base_gate[slot],
                base_up[slot],
                residual_gate[slot],
                residual_up[slot],
                device_down,
                down_partial,
                tile,
                tile_rows=args.tile_rows,
                ffn_dim=rows,
                out_dim=hidden,
                block_m=256,
                num_warps=4,
            )

    def measure(overlap: bool, collect_debug: bool = False) -> dict[str, float]:
        copy_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        ready = [torch.cuda.Event() for _ in range(tile_count)]
        done = [torch.cuda.Event() for _ in range(tile_count)]
        copy_begin = [torch.cuda.Event(enable_timing=True) for _ in range(tile_count)]
        copy_end = [torch.cuda.Event(enable_timing=True) for _ in range(tile_count)]
        compute_begin = [torch.cuda.Event(enable_timing=True) for _ in range(tile_count)]
        compute_end = [torch.cuda.Event(enable_timing=True) for _ in range(tile_count)]
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        cpu_start = time.perf_counter()
        base_futures = None
        if args.base_mode == "table":
            assert cpu_pool is not None and gate_table is not None and up_table is not None
            base_futures = [
                (
                    cpu_pool.submit(evaluate_base_tile, gate_table, tile * args.tile_rows, (tile + 1) * args.tile_rows),
                    cpu_pool.submit(evaluate_base_tile, up_table, tile * args.tile_rows, (tile + 1) * args.tile_rows),
                )
                for tile in range(tile_count)
            ]
        start.record(copy_stream)
        for tile in range(tile_count):
            slot = tile & 1 if overlap else 0
            offset = tile * args.tile_rows
            if args.base_mode == "table":
                assert base_futures is not None
                gate_tile = base_futures[tile][0].result()
                up_tile = base_futures[tile][1].result()
                np.copyto(host_gate_base_slots[slot].numpy(), gate_tile)
                np.copyto(host_up_base_slots[slot].numpy(), up_tile)
                gate_base_src = host_gate_base_slots[slot]
                up_base_src = host_up_base_slots[slot]
            else:
                assert host_gate_base is not None and host_up_base is not None
                gate_base_src = host_gate_base[offset : offset + args.tile_rows]
                up_base_src = host_up_base[offset : offset + args.tile_rows]
            with torch.cuda.stream(copy_stream):
                previous_tile = tile - 2 if overlap else tile - 1
                if previous_tile >= 0:
                    copy_stream.wait_event(done[previous_tile])
                copy_begin[tile].record(copy_stream)
                packet_gate[slot].copy_(host_gate[offset : offset + args.tile_rows], non_blocking=True)
                packet_up[slot].copy_(host_up[offset : offset + args.tile_rows], non_blocking=True)
                base_gate[slot].copy_(gate_base_src, non_blocking=True)
                base_up[slot].copy_(up_base_src, non_blocking=True)
                copy_end[tile].record(copy_stream)
                ready[tile].record(copy_stream)
            compute_stream.wait_event(ready[tile])
            compute_begin[tile].record(compute_stream)
            enqueue_tile(slot, tile, compute_stream, collect_debug)
            compute_end[tile].record(compute_stream)
            done[tile].record(compute_stream)
        stop.record(compute_stream)
        stop.synchronize()
        copy_ms = float(sum(begin.elapsed_time(end) for begin, end in zip(copy_begin, copy_end)))
        compute_ms = float(sum(begin.elapsed_time(end) for begin, end in zip(compute_begin, compute_end)))
        critical_ms = float(start.elapsed_time(stop))
        copy_stream.synchronize()
        compute_stream.synchronize()
        torch.sum(down_partial, dim=0, out=down_output)
        torch.cuda.synchronize()
        result = {
            "copy_active_ms": copy_ms,
            "compute_active_ms": compute_ms,
            "critical_ms": critical_ms,
        }
        if args.base_mode == "table":
            result["cpu_base_production_wall_ms"] = (time.perf_counter() - cpu_start) * 1000.0
        return result

    # Warm up compilation and execute one full chain before measuring.
    for _ in range(2):
        measure(True)

    overlap_samples = [measure(True) for _ in range(args.repeats)]
    serial_samples = [measure(False) for _ in range(max(3, args.repeats // 2))]
    overlap = {key: float(statistics.median([sample[key] for sample in overlap_samples])) for key in overlap_samples[0]}
    serial = {key: float(statistics.median([sample[key] for sample in serial_samples])) for key in serial_samples[0]}
    measure(True, collect_debug=True)
    if cpu_pool is not None:
        cpu_pool.shutdown(wait=True)

    # Validate the final tile-independent output against the original Q4 path.
    gate_codes, gate_alpha, gate_beta, _ = load_q4_projection(args.model, args.layer, "gate")
    up_codes, up_alpha, up_beta, _ = load_q4_projection(args.model, args.layer, "up")
    gate_ref = projection_from_group_dots(
        direct_group_dots(gate_codes, z), z.reshape(1, -1), scales.reshape(1, -1), gate_alpha, gate_beta, 1
    )
    up_ref = projection_from_group_dots(
        direct_group_dots(up_codes, z), z.reshape(1, -1), scales.reshape(1, -1), up_alpha, up_beta, 1
    )
    hidden_ref = (gate_ref / (1.0 + np.exp(-gate_ref))) * up_ref
    output_ref = hidden_ref @ down.T
    output_gpu = down_output.detach().cpu().numpy().copy()
    partial_gpu = down_partial.detach().cpu().numpy().copy()
    hidden_tiles_gpu = hidden_tiles.detach().cpu().numpy().copy().reshape(-1)
    gate_merged_gpu = gate_merged_tiles.detach().cpu().numpy().copy().reshape(-1)
    up_merged_gpu = up_merged_tiles.detach().cpu().numpy().copy().reshape(-1)
    gate_residual_gpu = gate_residual_tiles.detach().cpu().numpy().copy().reshape(-1)
    up_residual_gpu = up_residual_tiles.detach().cpu().numpy().copy().reshape(-1)
    output_rel_l2 = float(np.linalg.norm(output_gpu - output_ref) / max(np.linalg.norm(output_ref), 1e-12))
    tile_hidden_ref = hidden_ref[: args.tile_rows]
    tile_output_ref = tile_hidden_ref @ down[:, : args.tile_rows].T
    tile_partial_rel_l2 = float(
        np.linalg.norm(partial_gpu[0] - tile_output_ref) / max(np.linalg.norm(tile_output_ref), 1e-12)
    )
    hidden_tile_rel_l2 = float(
        np.linalg.norm(hidden_tiles_gpu[: args.tile_rows] - tile_hidden_ref) / max(np.linalg.norm(tile_hidden_ref), 1e-12)
    )
    gate_tile_rel_l2 = float(
        np.linalg.norm(gate_merged_gpu[: args.tile_rows] - gate_ref[: args.tile_rows])
        / max(np.linalg.norm(gate_ref[: args.tile_rows]), 1e-12)
    )
    up_tile_rel_l2 = float(
        np.linalg.norm(up_merged_gpu[: args.tile_rows] - up_ref[: args.tile_rows])
        / max(np.linalg.norm(up_ref[: args.tile_rows]), 1e-12)
    )
    gate_residual_tile_rel_l2 = float(
        np.linalg.norm(gate_residual_gpu[: args.tile_rows] - (gate_ref[: args.tile_rows] - gate_base[: args.tile_rows]))
        / max(np.linalg.norm(gate_ref[: args.tile_rows] - gate_base[: args.tile_rows]), 1e-12)
    )
    up_residual_tile_rel_l2 = float(
        np.linalg.norm(up_residual_gpu[: args.tile_rows] - (up_ref[: args.tile_rows] - up_base[: args.tile_rows]))
        / max(np.linalg.norm(up_ref[: args.tile_rows] - up_base[: args.tile_rows]), 1e-12)
    )
    tile_partial_errors = []
    tile_hidden_errors = []
    for tile in range(tile_count):
        begin = tile * args.tile_rows
        end = begin + args.tile_rows
        tile_hidden = hidden_ref[begin:end]
        tile_output = tile_hidden @ down[:, begin:end].T
        tile_partial_errors.append(
            float(np.linalg.norm(partial_gpu[tile] - tile_output) / max(np.linalg.norm(tile_output), 1e-12))
        )
        tile_hidden_errors.append(
            float(np.linalg.norm(hidden_tiles_gpu[begin:end] - tile_hidden) / max(np.linalg.norm(tile_hidden), 1e-12))
        )
    payload_bytes = int(gate_payload.nbytes + up_payload.nbytes)
    base_bytes = int(gate_base.nbytes + up_base.nbytes)
    return {
        "experiment": "cpu_base_gpu_residual_full_ffn",
        "base_mode": args.base_mode,
        "cpu_threads": args.cpu_threads,
        "layer": args.layer,
        "dimensions": {"hidden": hidden, "ffn": rows, "tile_rows": args.tile_rows, "tile_count": tile_count},
        "h2d": {
            "residual_payload_bytes": payload_bytes,
            "residual_payload_mib": payload_bytes / 2**20,
            "cpu_base_total_bytes": base_bytes,
            "cpu_base_per_tile_bytes": base_bytes // tile_count,
            "total_payload_bytes": payload_bytes + base_bytes,
        },
        "serial": serial,
        "double_buffer_overlap": overlap,
        "overlap_speedup": serial["critical_ms"] / max(overlap["critical_ms"], 1e-12),
        "correctness": {
            "full_down_output_rel_l2_fp32_accum": output_rel_l2,
            "first_tile_down_partial_rel_l2": tile_partial_rel_l2,
            "gpu_output_norm": float(np.linalg.norm(output_gpu)),
            "reference_output_norm": float(np.linalg.norm(output_ref)),
            "first_tile_gpu_norm": float(np.linalg.norm(partial_gpu[0])),
            "first_tile_reference_norm": float(np.linalg.norm(tile_output_ref)),
            "first_tile_swiglu_rel_l2": hidden_tile_rel_l2,
            "first_tile_gate_merged_rel_l2": gate_tile_rel_l2,
            "first_tile_up_merged_rel_l2": up_tile_rel_l2,
            "first_tile_gate_residual_rel_l2": gate_residual_tile_rel_l2,
            "first_tile_up_residual_rel_l2": up_residual_tile_rel_l2,
            "tile_down_partial_rel_l2": tile_partial_errors,
            "tile_swiglu_rel_l2": tile_hidden_errors,
        },
        "down_weight": {"format": "resident fp16 dequantized Q4_K", "bytes": int(device_down.numel() * 2)},
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU radix base plus GPU residual/SwiGLU/down pipeline")
    parser.add_argument("model", type=Path)
    parser.add_argument("table_artifact", type=Path)
    parser.add_argument("residual_artifact", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--tile-rows", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-mode", choices=("preexpanded", "table"), default="preexpanded")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
