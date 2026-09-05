from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def pack_2bit_rows(values: np.ndarray) -> np.ndarray:
    codes = np.asarray(values, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] % 4:
        raise ValueError("2-bit rows must have a width divisible by four")
    if np.any(codes > 3):
        raise ValueError("2-bit codes must be in [0, 3]")
    return (
        codes[:, 0::4]
        | (codes[:, 1::4] << 2)
        | (codes[:, 2::4] << 4)
        | (codes[:, 3::4] << 6)
    ).astype(np.uint8, copy=False)


def unpack_2bit_rows(packed: np.ndarray, width: int) -> np.ndarray:
    data = np.asarray(packed, dtype=np.uint8)
    if data.ndim != 2 or width != data.shape[1] * 4:
        raise ValueError("packed row width does not match requested output width")
    out = np.empty((data.shape[0], width), dtype=np.uint8)
    for position in range(4):
        out[:, position::4] = (data >> (2 * position)) & 3
    return out


def quantize_groupwise_q8(x: np.ndarray, group_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % group_size:
        raise ValueError("x must have shape [tokens, hidden] with complete groups")
    grouped = values.reshape(values.shape[0], -1, group_size)
    scale = np.max(np.abs(grouped), axis=2) / np.float32(127.0)
    scale = np.maximum(scale, np.float32(1e-12))
    codes = np.clip(np.rint(grouped / scale[:, :, None]), -128, 127).astype(np.int8)
    return codes.reshape(values.shape), scale.astype(np.float32, copy=False)


def compile_radix_table(q_hi: np.ndarray, block_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Compile exact base-4 partial sums for q_hi shaped [rows, groups, group_size]."""
    q = np.asarray(q_hi, dtype=np.uint8)
    if q.ndim != 3:
        raise ValueError("q_hi must have shape [rows, groups, group_size]")
    rows, groups, group_size = q.shape
    if block_size not in (2, 4, 8) or group_size % block_size:
        raise ValueError("block_size must be one of 2, 4, or 8 and divide group_size")
    blocks_per_group = group_size // block_size
    blocks = q.reshape(rows, groups * blocks_per_group, block_size).transpose(1, 0, 2).astype(np.uint16)
    states = np.arange(4**block_size, dtype=np.uint32)
    digits = np.stack([(states // (4**position)) & 3 for position in range(block_size)], axis=1)
    table = np.einsum("brc,sc->bsr", blocks, digits, optimize=True).astype(np.uint8)
    high_sum = q.astype(np.int32).sum(axis=2)
    return table, high_sum


def encode_signed_base4_states(z: np.ndarray, block_size: int = 4) -> np.ndarray:
    """Encode int8 values as four unsigned base-4 digit-state streams."""
    values = np.asarray(z, dtype=np.int8).reshape(-1)
    if block_size not in (2, 4, 8) or values.size % block_size:
        raise ValueError("block_size must be one of 2, 4, or 8 and divide the input width")
    unsigned = values.astype(np.int16) + 128
    blocks = unsigned.reshape(-1, block_size)
    digit_states = []
    positional = (4 ** np.arange(block_size, dtype=np.int32))[None, :]
    for radix_position in range(4):
        digits = (blocks // (4**radix_position)) & 3
        digit_states.append(np.sum(digits * positional, axis=1, dtype=np.int32))
    return np.stack(digit_states).astype(np.int64, copy=False)


def evaluate_radix_table(
    table: np.ndarray,
    high_sum: np.ndarray,
    states: np.ndarray,
    blocks_per_group: int,
) -> np.ndarray:
    """Reconstruct exact signed-int8 group dots from a compiled base-4 table."""
    lookup = np.asarray(table, dtype=np.uint8)
    state_codes = np.asarray(states, dtype=np.int64)
    if state_codes.shape[0] != 4 or lookup.shape[0] != state_codes.shape[1]:
        raise ValueError("state streams do not match the compiled table")
    state_count = lookup.shape[1]
    block_size = 0
    while 4**block_size < state_count:
        block_size += 1
    if 4**block_size != state_count:
        raise ValueError("table state count must be a power of four")
    rows, groups = high_sum.shape
    if lookup.shape[0] != groups * blocks_per_group or lookup.shape[2] != rows:
        raise ValueError("table geometry does not match high_sum")
    block_index = np.arange(lookup.shape[0], dtype=np.int64)
    group_dot = np.zeros((groups, rows), dtype=np.int32)
    for radix_position in range(4):
        selected = lookup[block_index, state_codes[radix_position]].astype(np.int32)
        partial = selected.reshape(groups, blocks_per_group, rows).sum(axis=1, dtype=np.int32)
        group_dot += partial * (4**radix_position)
    group_dot -= 128 * high_sum.T
    return group_dot.T


def projection_from_group_dots(
    group_dots: np.ndarray,
    z: np.ndarray,
    scales: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    code_multiplier: int = 1,
) -> np.ndarray:
    z_groups = np.asarray(z, dtype=np.int8).reshape(-1, alpha.shape[1], 32)
    if len(z_groups) != 1:
        raise ValueError("the decode prototype currently evaluates one token")
    z_sum = z_groups[0].astype(np.int32).sum(axis=1)
    return np.sum(
        scales[0][None, :]
        * (alpha * (code_multiplier * group_dots) + beta * z_sum[None, :]),
        axis=1,
        dtype=np.float32,
    )


def benchmark_cpu_table(
    table: np.ndarray,
    high_sum: np.ndarray,
    states: np.ndarray,
    blocks_per_group: int,
    z: np.ndarray,
    scales: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, dict[str, float]]:
    output = None
    for _ in range(warmup):
        dots = evaluate_radix_table(table, high_sum, states, blocks_per_group)
        output = projection_from_group_dots(dots, z, scales, alpha, beta, code_multiplier=4)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        dots = evaluate_radix_table(table, high_sum, states, blocks_per_group)
        output = projection_from_group_dots(dots, z, scales, alpha, beta, code_multiplier=4)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    assert output is not None
    return output, {
        "median_ms": float(statistics.median(timings)),
        "min_ms": float(min(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def direct_group_dots(codes: np.ndarray, z: np.ndarray) -> np.ndarray:
    grouped_z = np.asarray(z, dtype=np.int8).reshape(codes.shape[1], codes.shape[2])
    return np.einsum("rgi,gi->rg", codes.astype(np.int32), grouped_z.astype(np.int32), optimize=True)


def benchmark_projection_cpu(
    codes: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    z: np.ndarray,
    scales: np.ndarray,
    warmup: int,
    repeats: int,
    block_size: int = 4,
) -> tuple[dict, np.ndarray, np.ndarray]:
    q_hi = codes >> 2
    q_lo = codes & 3
    states = encode_signed_base4_states(z.reshape(-1), block_size=block_size)
    compile_start = time.perf_counter()
    table, high_sum = compile_radix_table(q_hi, block_size=block_size)
    compile_seconds = time.perf_counter() - compile_start
    base, cpu_profile = benchmark_cpu_table(
        table,
        high_sum,
        states,
        blocks_per_group=32 // block_size,
        z=z,
        scales=scales,
        alpha=alpha,
        beta=beta,
        warmup=warmup,
        repeats=repeats,
    )
    high_direct = direct_group_dots(q_hi, z)
    low_direct = direct_group_dots(q_lo, z)
    direct_base = projection_from_group_dots(high_direct, z, scales, alpha, beta, code_multiplier=4)
    residual = projection_from_group_dots(
        low_direct,
        z,
        scales,
        alpha,
        np.zeros_like(beta),
        code_multiplier=1,
    )
    direct_full = projection_from_group_dots(
        4 * high_direct + low_direct,
        z,
        scales,
        alpha,
        beta,
        code_multiplier=1,
    )
    merged = base + residual
    profile = {
        "cold_compile_seconds": compile_seconds,
        "table_bytes": int(table.nbytes),
        "table_mib": table.nbytes / 2**20,
        "runtime_selected_table_bytes_per_token": int(4 * table.shape[0] * table.shape[2]),
        "runtime_selected_table_mib_per_token": 4 * table.shape[0] * table.shape[2] / 2**20,
        "cpu_runtime": cpu_profile,
        "integer_high_dot_exact": bool(np.array_equal(evaluate_radix_table(table, high_sum, states, 32 // block_size), high_direct)),
        "base_max_abs_error": float(np.max(np.abs(base - direct_base))),
        "merged_max_abs_error": float(np.max(np.abs(merged - direct_full))),
        "merged_rel_l2": float(np.linalg.norm(merged - direct_full) / max(np.linalg.norm(direct_full), 1e-12)),
    }
    del table
    gc.collect()
    return profile, merged, residual


def gpu_residual_benchmark(
    packed: np.ndarray,
    alpha: np.ndarray,
    z: np.ndarray,
    scales: np.ndarray,
    tile_rows: int,
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, dict]:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def residual_kernel(
        packed_ptr,
        alpha_ptr,
        z_ptr,
        scale_ptr,
        partial_ptr,
        packed_row_bytes: tl.constexpr,
        groups: tl.constexpr,
        chunks: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        offsets = chunk * block_n + tl.arange(0, block_n)
        packed_value = tl.load(packed_ptr + row * packed_row_bytes + offsets // 4).to(tl.uint32)
        shift = (offsets & 3) * 2
        q = (packed_value >> shift) & 3
        group = offsets // 32
        a = tl.load(alpha_ptr + row * groups + group)
        activation = tl.load(z_ptr + offsets).to(tl.float32)
        activation_scale = tl.load(scale_ptr + group)
        contribution = q.to(tl.float32) * a * activation * activation_scale
        tl.store(partial_ptr + row * chunks + chunk, tl.sum(contribution, axis=0))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows, packed_width = packed.shape
    hidden = packed_width * 4
    groups = hidden // 32
    chunks = hidden // 256
    if rows % tile_rows:
        raise ValueError("rows must be divisible by tile_rows")
    host_packed = torch.from_numpy(np.ascontiguousarray(packed)).pin_memory()
    host_alpha = torch.from_numpy(np.ascontiguousarray(alpha.astype(np.float32, copy=False))).pin_memory()
    z_device = torch.from_numpy(np.ascontiguousarray(z.reshape(-1))).cuda()
    scale_device = torch.from_numpy(np.ascontiguousarray(scales.reshape(-1))).cuda()
    packed_buffers = [torch.empty((tile_rows, packed_width), dtype=torch.uint8, device="cuda") for _ in range(2)]
    alpha_buffers = [torch.empty((tile_rows, groups), dtype=torch.float32, device="cuda") for _ in range(2)]
    partial_buffers = [torch.empty((tile_rows, chunks), dtype=torch.float32, device="cuda") for _ in range(2)]
    output_buffers = [torch.empty(tile_rows, dtype=torch.float32, device="cuda") for _ in range(2)]

    def launch(buffer: int) -> None:
        residual_kernel[(tile_rows, chunks)](
            packed_buffers[buffer],
            alpha_buffers[buffer],
            z_device,
            scale_device,
            partial_buffers[buffer],
            packed_row_bytes=packed_width,
            groups=groups,
            chunks=chunks,
            block_n=256,
            num_warps=4,
        )
        torch.sum(partial_buffers[buffer], dim=1, out=output_buffers[buffer])

    # Compile and verify one tile before timing.
    packed_buffers[0].copy_(host_packed[:tile_rows], non_blocking=False)
    alpha_buffers[0].copy_(host_alpha[:tile_rows], non_blocking=False)
    launch(0)
    torch.cuda.synchronize()
    first_output = output_buffers[0].cpu().numpy().copy()

    tile_count = rows // tile_rows

    def measure_copy_only() -> float:
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for tile in range(tile_count):
                begin = tile * tile_rows
                packed_buffers[0].copy_(host_packed[begin : begin + tile_rows], non_blocking=True)
                alpha_buffers[0].copy_(host_alpha[begin : begin + tile_rows], non_blocking=True)
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        return float(statistics.median(samples))

    def measure_serial() -> float:
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for tile in range(tile_count):
                begin = tile * tile_rows
                packed_buffers[0].copy_(host_packed[begin : begin + tile_rows], non_blocking=True)
                alpha_buffers[0].copy_(host_alpha[begin : begin + tile_rows], non_blocking=True)
                launch(0)
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        return float(statistics.median(samples))

    def measure_overlap() -> float:
        copy_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        ready = [torch.cuda.Event() for _ in range(2)]
        done = [torch.cuda.Event() for _ in range(2)]
        # Seed reusable-buffer completion events.
        for event in done:
            event.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for tile in range(tile_count):
                buffer = tile & 1
                begin = tile * tile_rows
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_event(done[buffer])
                    packed_buffers[buffer].copy_(host_packed[begin : begin + tile_rows], non_blocking=True)
                    alpha_buffers[buffer].copy_(host_alpha[begin : begin + tile_rows], non_blocking=True)
                    ready[buffer].record(copy_stream)
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(ready[buffer])
                    launch(buffer)
                    done[buffer].record(compute_stream)
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        return float(statistics.median(samples))

    for _ in range(warmup):
        measure_serial()
    copy_ms = measure_copy_only()
    serial_ms = measure_serial()
    overlap_ms = measure_overlap()
    payload_bytes = int(packed.nbytes + alpha.nbytes)
    return first_output, {
        "tile_rows": tile_rows,
        "tile_count": tile_count,
        "payload_bytes": payload_bytes,
        "payload_mib": payload_bytes / 2**20,
        "copy_only_median_ms": copy_ms,
        "serial_copy_compute_median_ms": serial_ms,
        "double_buffer_overlap_median_ms": overlap_ms,
        "overlap_speedup_vs_serial": serial_ms / overlap_ms,
        "serial_minus_overlap_ms": serial_ms - overlap_ms,
        "effective_payload_gbps_overlap": payload_bytes / max(overlap_ms / 1000.0, 1e-12) / 1e9,
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
    }


def load_q4_projection(model: Path, layer: int, projection: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from scan_q4k_hierarchical_code_split import load_q4k_codes

    codes, alpha, beta, _, source_bytes = load_q4k_codes(model, layer, projection)
    rows, blocks, _ = codes.shape
    return (
        codes.reshape(rows, blocks * 8, 32),
        alpha.reshape(rows, blocks * 8),
        beta.reshape(rows, blocks * 8),
        source_bytes,
    )


def load_probe_input(path: Path, hidden: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != hidden:
        raise ValueError(f"expected {hidden} float32 values, found {values.size}")
    return values.reshape(1, hidden)


def baseline_q4_copy_ms(total_bytes: int, repeats: int) -> float:
    import torch

    host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    device = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        device.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return float(statistics.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact radix main table plus exact 2-bit GPU residual feasibility benchmark")
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--tile-rows", type=int, default=256)
    parser.add_argument("--table-block-size", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    projection_data = {name: load_q4_projection(args.model, args.layer, name) for name in ("gate", "up")}
    hidden = projection_data["gate"][0].shape[1] * 32
    x = load_probe_input(args.input, hidden)
    z, scales = quantize_groupwise_q8(x, group_size=32)
    x_quant = z.reshape(1, -1, 32).astype(np.float32) * scales[:, :, None]
    quantization_rel_l2 = float(np.linalg.norm(x_quant.reshape(x.shape) - x) / max(np.linalg.norm(x), 1e-12))

    projections = {}
    original_q4_bytes = 0
    for name, (codes, alpha, beta, source_bytes) in projection_data.items():
        original_q4_bytes += source_bytes
        cpu_profile, merged, residual_cpu = benchmark_projection_cpu(
            codes,
            alpha,
            beta,
            z,
            scales,
            args.warmup,
            args.repeats,
            block_size=args.table_block_size,
        )
        packed = pack_2bit_rows((codes & 3).reshape(codes.shape[0], -1))
        gpu_first, gpu_profile = gpu_residual_benchmark(
            packed, alpha, z, scales, args.tile_rows, args.warmup, args.repeats
        )
        tile_error = gpu_first - residual_cpu[: args.tile_rows]
        gpu_profile["first_tile_max_abs_error_vs_cpu"] = float(np.max(np.abs(tile_error)))
        gpu_profile["first_tile_rel_l2_vs_cpu"] = float(
            np.linalg.norm(tile_error) / max(np.linalg.norm(residual_cpu[: args.tile_rows]), 1e-12)
        )
        projections[name] = {
            "shape": list(codes.shape),
            "source_q4_bytes": source_bytes,
            "cpu_main": cpu_profile,
            "gpu_exact_residual": gpu_profile,
            "merged_output_norm": float(np.linalg.norm(merged)),
        }

    q4_copy_ms = baseline_q4_copy_ms(original_q4_bytes, max(args.repeats, 10))
    cpu_gate_up_ms = sum(row["cpu_main"]["cpu_runtime"]["median_ms"] for row in projections.values())
    gpu_gate_up_serial_ms = sum(row["gpu_exact_residual"]["serial_copy_compute_median_ms"] for row in projections.values())
    gpu_gate_up_overlap_ms = sum(row["gpu_exact_residual"]["double_buffer_overlap_median_ms"] for row in projections.values())
    optimistic_parallel_ms = max(cpu_gate_up_ms, gpu_gate_up_overlap_ms)
    result = {
        "experiment": "exact_radix_main_plus_exact_2bit_gpu_residual",
        "date": "2026-09-04",
        "platform": platform.platform(),
        "layer": args.layer,
        "input": str(args.input),
        "dimensions": {"hidden": hidden, "ffn": projection_data["gate"][0].shape[0]},
        "activation": {
            "representation": "groupwise signed int8, 32 values per scale",
            "quantization_rel_l2_vs_fp32_capture": quantization_rel_l2,
            "scope": "the radix table is exact for these int8 codes; activation quantization is reported separately",
        },
        "formula": {
            "weight": "q = 4*q_hi + q_lo, both q_hi and q_lo retained exactly",
            "activation": "z = -128 + sum_{d=0..3} 4^d*z_d, z_d in [0,3]",
            "main": "cold-start table[block4,state] returns exact uint8 partial dot for q_hi; runtime uses four digit lookups",
            "residual": "GPU unpacks exact q_lo from 2-bit rows and computes scaled group dot",
        },
        "projections": projections,
        "gate_up_summary": {
            "original_q4_bytes": original_q4_bytes,
            "baseline_original_q4_h2d_copy_only_median_ms": q4_copy_ms,
            "cpu_main_median_ms": cpu_gate_up_ms,
            "gpu_residual_serial_median_ms": gpu_gate_up_serial_ms,
            "gpu_residual_double_buffer_median_ms": gpu_gate_up_overlap_ms,
            "optimistic_cpu_gpu_parallel_critical_path_ms": optimistic_parallel_ms,
            "note": "the optimistic critical path assumes CPU gate/up main evaluation overlaps the complete GPU residual pipeline; integration overhead is excluded",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["gate_up_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
