from __future__ import annotations

import time

import numpy as np
import torch
import triton
import triton.language as tl

from resident_residual_format import ResidentArtifact


@triton.jit
def _direct_q4k(raw, x, output, ROWS: tl.constexpr, COLS: tl.constexpr,
                BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_COLS)
    mask = (row[:, None] < ROWS) & (col[None, :] < COLS)
    block = raw + row[:, None] * (COLS // 256 * 144) + col[None, :] // 256 * 144
    d_bits = tl.load(block, mask, other=0).to(tl.uint32) | (tl.load(block + 1, mask, other=0).to(tl.uint32) << 8)
    m_bits = tl.load(block + 2, mask, other=0).to(tl.uint32) | (tl.load(block + 3, mask, other=0).to(tl.uint32) << 8)
    d = d_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    dm = m_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    group = (col[None, :] % 256) // 32
    low_scale = tl.load(block + 4 + group % 4, mask, other=0).to(tl.int32)
    low_min = tl.load(block + 8 + group % 4, mask, other=0).to(tl.int32)
    mix = tl.load(block + 12 + group % 4, mask, other=0).to(tl.int32)
    scale = tl.where(group < 4, low_scale & 63, (mix & 15) | ((low_scale >> 2) & 48))
    minimum = tl.where(group < 4, low_min & 63, (mix >> 4) | ((low_min >> 2) & 48))
    packed = tl.load(block + 16 + (col[None, :] % 256) // 64 * 32 + col[None, :] % 32, mask, other=0).to(tl.int32)
    q = (packed >> ((group % 2) * 4)) & 15
    weight = (d * scale.to(tl.float32)) * q.to(tl.float32) - dm * minimum.to(tl.float32)
    activation = tl.load(x + col, col < COLS, other=0)
    dot = tl.sum(weight * activation[None, :], axis=1)
    tl.store(output + row, dot, row < ROWS)


@triton.jit
def _fused_q4k(
    raw,
    x,
    output,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """K-tiled Q4_K GEMV with the accumulator kept in registers.

    The legacy path used one 32768-lane vector for a 17408-wide projection.
    That shape is correct but creates an unnecessarily large register/live
    range for batch-1 decode.  K tiling preserves the exact Q4_K decode and
    reduction order while giving the scheduler bounded programs.
    """
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for col_start in range(0, COLS, BLOCK_COLS):
        col = col_start + tl.arange(0, BLOCK_COLS)
        col_mask = col < COLS
        mask = row_mask[:, None] & col_mask[None, :]
        block = raw + row[:, None] * (COLS // 256 * 144) + col[None, :] // 256 * 144
        d_bits = (
            tl.load(block, mask, other=0).to(tl.uint32)
            | (tl.load(block + 1, mask, other=0).to(tl.uint32) << 8)
        )
        m_bits = (
            tl.load(block + 2, mask, other=0).to(tl.uint32)
            | (tl.load(block + 3, mask, other=0).to(tl.uint32) << 8)
        )
        d = tl.cast(d_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
        dm = tl.cast(m_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
        group = (col[None, :] % 256) // 32
        low_scale = tl.load(
            block + 4 + group % 4, mask, other=0
        ).to(tl.int32)
        low_min = tl.load(
            block + 8 + group % 4, mask, other=0
        ).to(tl.int32)
        mix = tl.load(
            block + 12 + group % 4, mask, other=0
        ).to(tl.int32)
        scale = tl.where(
            group < 4, low_scale & 63, (mix & 15) | ((low_scale >> 2) & 48)
        )
        minimum = tl.where(
            group < 4, low_min & 63, (mix >> 4) | ((low_min >> 2) & 48)
        )
        packed = tl.load(
            block + 16 + (col[None, :] % 256) // 64 * 32 + col[None, :] % 32,
            mask,
            other=0,
        ).to(tl.int32)
        q = (packed >> ((group % 2) * 4)) & 15
        weight = (d * scale.to(tl.float32)) * q.to(tl.float32) - dm * minimum.to(tl.float32)
        activation = tl.load(x + col, col_mask, other=0)
        acc += tl.sum(weight * activation[None, :], axis=1)
    tl.store(output + row, acc, row_mask)


@triton.jit
def _direct_iq4nl(raw, x, partial, kvalues, ROWS: tl.constexpr, COLS: tl.constexpr,
                  CHUNKS: tl.constexpr, CHUNK_COLS: tl.constexpr,
                  BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    chunk = tl.program_id(1)
    local = tl.arange(0, BLOCK_COLS)
    col = chunk * CHUNK_COLS + local
    mask = (row[:, None] < ROWS) & (local[None, :] < CHUNK_COLS) & (col[None, :] < COLS)
    block = raw + row[:, None] * (COLS // 32 * 18) + col[None, :] // 32 * 18
    d_bits = tl.load(block, mask, other=0).to(tl.uint32) | (tl.load(block + 1, mask, other=0).to(tl.uint32) << 8)
    d = tl.cast(d_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
    packed = tl.load(block + 2 + col[None, :] % 16, mask, other=0).to(tl.int32)
    q = (packed >> (((col[None, :] % 32) // 16) * 4)) & 15
    weight = d * tl.load(kvalues + q, q < 16, other=0).to(tl.float32)
    activation = tl.load(x + col, col < COLS, other=0)
    dot = tl.sum(weight * activation[None, :], axis=1)
    tl.store(partial + row * CHUNKS + chunk, dot, row < ROWS)


@triton.jit
def _fused_iq4nl(raw, x, output, kvalues, ROWS: tl.constexpr, COLS: tl.constexpr,
                 BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    """Single-kernel IQ4_NL GEMV with bounded K tiles.

    The old path emitted one kernel per K chunk and a second torch reduction
    over the partial matrix.  Keeping the accumulator in registers removes the
    intermediate partial write/read and the extra launch boundary while
    preserving the original IQ4_NL decode.
    """
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for col_start in range(0, COLS, BLOCK_COLS):
        local = tl.arange(0, BLOCK_COLS)
        col = col_start + local
        mask = row_mask[:, None] & (col[None, :] < COLS)
        block = raw + row[:, None] * (COLS // 32 * 18) + col[None, :] // 32 * 18
        d_bits = (
            tl.load(block, mask, other=0).to(tl.uint32)
            | (tl.load(block + 1, mask, other=0).to(tl.uint32) << 8)
        )
        d = tl.cast(d_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
        packed = tl.load(
            block + 2 + col[None, :] % 16,
            mask,
            other=0,
        ).to(tl.int32)
        q = (packed >> (((col[None, :] % 32) // 16) * 4)) & 15
        weight = d * tl.load(
            kvalues + q,
            q < 16,
            other=0,
        ).to(tl.float32)
        activation = tl.load(x + col, col < COLS, other=0)
        acc += tl.sum(weight * activation[None, :], axis=1)
    tl.store(output + row, acc, row_mask)


class DirectQ4Projection:
    def __init__(
        self,
        raw: np.ndarray,
        cols: int,
        *,
        block_rows: int = 2,
        num_warps: int = 2,
        chunk_cols: int = 512,
    ):
        if raw.dtype != np.uint8 or raw.ndim != 2 or cols % 256 or raw.shape[1] != cols // 256 * 144:
            raise ValueError("expected row-major raw Q4_K tensor")
        if chunk_cols <= 0 or chunk_cols % 256:
            raise ValueError("chunk_cols must be a positive multiple of 256")
        self.raw = torch.from_numpy(np.array(raw, copy=True)).cuda()
        self.rows, self.cols = raw.shape[0], cols
        self.block_rows, self.num_warps = block_rows, num_warps
        self.chunk_cols = min(int(chunk_cols), cols)
        self.output = torch.empty(self.rows, device="cuda")

    def launch(self, device_x: torch.Tensor) -> None:
        _fused_q4k[(triton.cdiv(self.rows, self.block_rows),)](
            self.raw, device_x, self.output, ROWS=self.rows, COLS=self.cols,
            BLOCK_ROWS=self.block_rows, BLOCK_COLS=triton.next_power_of_2(self.chunk_cols),
            num_warps=self.num_warps, enable_fp_fusion=False,
        )


class DirectIQ4NLProjection:
    def __init__(self, raw: np.ndarray, cols: int, *, chunk_cols: int = 1024,
                 block_rows: int = 1, num_warps: int = 1):
        if raw.dtype != np.uint8 or raw.ndim != 2 or cols % 32 or raw.shape[1] != cols // 32 * 18:
            raise ValueError("expected row-major raw IQ4_NL tensor")
        self.raw = torch.from_numpy(np.array(raw, copy=True)).cuda()
        self.kvalues = torch.tensor(
            (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113),
            dtype=torch.float32, device="cuda",
        )
        self.rows, self.cols = raw.shape[0], cols
        if chunk_cols <= 0 or chunk_cols % 32:
            raise ValueError("chunk_cols must be a positive multiple of 32")
        self.chunk_cols = min(chunk_cols, cols)
        self.chunks = (cols + self.chunk_cols - 1) // self.chunk_cols
        self.block_rows, self.num_warps = block_rows, num_warps
        self.output = torch.empty(self.rows, device="cuda")
        self.partial = torch.empty((self.rows, self.chunks), device="cuda")

    def launch(self, device_x: torch.Tensor) -> None:
        _fused_iq4nl[(triton.cdiv(self.rows, self.block_rows),)](
            self.raw,
            device_x,
            self.output,
            self.kvalues,
            ROWS=self.rows,
            COLS=self.cols,
            BLOCK_ROWS=self.block_rows,
            BLOCK_COLS=triton.next_power_of_2(self.chunk_cols),
            num_warps=self.num_warps,
            num_stages=2,
            enable_fp_fusion=True,
        )


@triton.jit
def _residual_dot(packed, alpha, x, output, ROWS: tl.constexpr, COLS: tl.constexpr,
                  BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_COLS)
    mask = (row[:, None] < ROWS) & (col[None, :] < COLS)
    value = tl.load(packed + row[:, None] * (COLS // 2) + col[None, :] // 2, mask, other=0)
    q = ((value.to(tl.int32) >> ((col[None, :] % 2) * 4)) & 15)
    r = tl.where(q >= 8, q - 16, q).to(tl.float32)
    scale = tl.load(alpha + row[:, None] * (COLS // 32) + col[None, :] // 32, mask, other=0)
    activation = tl.load(x + col, col < COLS, other=0)
    dot = tl.sum(r * scale * activation[None, :], axis=1)
    tl.store(output + row, dot, row < ROWS)


@triton.jit
def _fused_gate_up_swiglu_tile(
    gate_packed, gate_alpha, up_packed, up_alpha, x,
    gate_base, up_base, gate_output, up_output, swiglu_output,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_COLS)
    mask = (row[:, None] < ROWS) & (col[None, :] < COLS)
    packed_offset = row[:, None] * (COLS // 2) + col[None, :] // 2
    shift = (col[None, :] % 2) * 4

    gate_value = tl.load(gate_packed + packed_offset, mask, other=0).to(tl.int32)
    gate_q = (gate_value >> shift) & 15
    gate_r = tl.where(gate_q >= 8, gate_q - 16, gate_q).to(tl.float32)
    gate_scale = tl.load(
        gate_alpha + row[:, None] * (COLS // 32) + col[None, :] // 32,
        mask,
        other=0,
    )

    up_value = tl.load(up_packed + packed_offset, mask, other=0).to(tl.int32)
    up_q = (up_value >> shift) & 15
    up_r = tl.where(up_q >= 8, up_q - 16, up_q).to(tl.float32)
    up_scale = tl.load(
        up_alpha + row[:, None] * (COLS // 32) + col[None, :] // 32,
        mask,
        other=0,
    )
    activation = tl.load(x + col, col < COLS, other=0)
    gate = tl.sum(gate_r * gate_scale * activation[None, :], axis=1)
    up = tl.sum(up_r * up_scale * activation[None, :], axis=1)
    row_mask = row < ROWS
    gate += tl.load(gate_base + row, row_mask, other=0)
    up += tl.load(up_base + row, row_mask, other=0)
    tl.store(gate_output + row, gate, row_mask)
    tl.store(up_output + row, up, row_mask)
    tl.store(swiglu_output + row, gate * tl.sigmoid(gate) * up, row_mask)


@triton.jit
def _fused_gate_up_residual_tile(
    gate_packed, gate_alpha, up_packed, up_alpha, x,
    gate_output, up_output,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_COLS)
    mask = (row[:, None] < ROWS) & (col[None, :] < COLS)
    packed_offset = row[:, None] * (COLS // 2) + col[None, :] // 2
    shift = (col[None, :] % 2) * 4

    gate_value = tl.load(gate_packed + packed_offset, mask, other=0).to(tl.int32)
    gate_q = (gate_value >> shift) & 15
    gate_r = tl.where(gate_q >= 8, gate_q - 16, gate_q).to(tl.float32)
    gate_scale = tl.load(
        gate_alpha + row[:, None] * (COLS // 32) + col[None, :] // 32,
        mask,
        other=0,
    )

    up_value = tl.load(up_packed + packed_offset, mask, other=0).to(tl.int32)
    up_q = (up_value >> shift) & 15
    up_r = tl.where(up_q >= 8, up_q - 16, up_q).to(tl.float32)
    up_scale = tl.load(
        up_alpha + row[:, None] * (COLS // 32) + col[None, :] // 32,
        mask,
        other=0,
    )
    activation = tl.load(x + col, col < COLS, other=0)
    row_mask = row < ROWS
    tl.store(
        gate_output + row,
        tl.sum(gate_r * gate_scale * activation[None, :], axis=1),
        row_mask,
    )
    tl.store(
        up_output + row,
        tl.sum(up_r * up_scale * activation[None, :], axis=1),
        row_mask,
    )


@triton.jit
def _fused_gate_up_base_residual(
    gate_packed, gate_alpha, up_packed, up_alpha,
    gate_coeff, up_coeff, group_sums, x,
    gate_output, up_output, swiglu_output,
    ROWS: tl.constexpr, COLS: tl.constexpr, GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """Full super-tile path: resident residual + resident base in one launch."""
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_COLS)
    group = tl.arange(0, BLOCK_GROUPS)
    row_mask = row < ROWS
    col_mask = col < COLS
    group_mask = group < GROUPS
    mask = row_mask[:, None] & col_mask[None, :]
    packed_offset = row[:, None] * (COLS // 2) + col[None, :] // 2
    shift = (col[None, :] % 2) * 4

    gate_value = tl.load(gate_packed + packed_offset, mask, other=0).to(tl.int32)
    gate_q = (gate_value >> shift) & 15
    gate_r = tl.where(gate_q >= 8, gate_q - 16, gate_q).to(tl.float32)
    gate_scale = tl.load(
        gate_alpha + row[:, None] * (COLS // 32) + col[None, :] // 32,
        mask, other=0,
    )
    up_value = tl.load(up_packed + packed_offset, mask, other=0).to(tl.int32)
    up_q = (up_value >> shift) & 15
    up_r = tl.where(up_q >= 8, up_q - 16, up_q).to(tl.float32)
    up_scale = tl.load(
        up_alpha + row[:, None] * (COLS // 32) + col[None, :] // 32,
        mask, other=0,
    )
    activation = tl.load(x + col, col_mask, other=0).to(tl.float32)
    gate_res = tl.sum(gate_r * gate_scale * activation[None, :], axis=1)
    up_res = tl.sum(up_r * up_scale * activation[None, :], axis=1)

    sums = tl.load(group_sums + group, group_mask, other=0).to(tl.float32)
    gate_c = tl.load(
        gate_coeff + row[:, None] * GROUPS + group[None, :],
        row_mask[:, None] & group_mask[None, :],
        other=0,
    ).to(tl.float32)
    up_c = tl.load(
        up_coeff + row[:, None] * GROUPS + group[None, :],
        row_mask[:, None] & group_mask[None, :],
        other=0,
    ).to(tl.float32)
    gate = gate_res + tl.sum(gate_c * sums[None, :], axis=1)
    up = up_res + tl.sum(up_c * sums[None, :], axis=1)
    tl.store(gate_output + row, gate, row_mask)
    tl.store(up_output + row, up, row_mask)
    tl.store(swiglu_output + row, gate * tl.sigmoid(gate) * up, row_mask)


@triton.jit
def _fused_gate_up_base_residual_tiled(
    gate_packed, gate_alpha, up_packed, up_alpha,
    gate_coeff, up_coeff, group_sums, x,
    gate_output, up_output, swiglu_output,
    ROWS: tl.constexpr, COLS: tl.constexpr, GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """K-tiled super-tile kernel with bounded register footprint.

    The previous super-tile kernel materialized the entire hidden dimension as
    one Triton vector (8192 lanes for a 5120-wide Qwen layer).  This variant
    keeps the same exact split/merge arithmetic but reduces the K tile so the
    scheduler can keep several row programs resident.
    """
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    gate_acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for col_start in range(0, COLS, BLOCK_COLS):
        col = col_start + tl.arange(0, BLOCK_COLS)
        col_mask = col < COLS
        mask = row_mask[:, None] & col_mask[None, :]
        packed_offset = row[:, None] * (COLS // 2) + col[None, :] // 2
        shift = (col[None, :] % 2) * 4

        gate_value = tl.load(
            gate_packed + packed_offset, mask=mask, other=0
        ).to(tl.int32)
        gate_q = (gate_value >> shift) & 15
        gate_r = tl.where(gate_q >= 8, gate_q - 16, gate_q).to(tl.float32)
        gate_scale = tl.load(
            gate_alpha + row[:, None] * GROUPS + col[None, :] // 32,
            mask=mask,
            other=0.0,
        )

        up_value = tl.load(
            up_packed + packed_offset, mask=mask, other=0
        ).to(tl.int32)
        up_q = (up_value >> shift) & 15
        up_r = tl.where(up_q >= 8, up_q - 16, up_q).to(tl.float32)
        up_scale = tl.load(
            up_alpha + row[:, None] * GROUPS + col[None, :] // 32,
            mask=mask,
            other=0.0,
        )

        activation = tl.load(x + col, mask=col_mask, other=0.0)
        gate_acc += tl.sum(
            gate_r * gate_scale * activation[None, :], axis=1
        )
        up_acc += tl.sum(
            up_r * up_scale * activation[None, :], axis=1
        )

    group = tl.arange(0, BLOCK_GROUPS)
    group_mask = group < GROUPS
    sums = tl.load(group_sums + group, mask=group_mask, other=0.0).to(tl.float32)
    gate_c = tl.load(
        gate_coeff + row[:, None] * GROUPS + group[None, :],
        mask=row_mask[:, None] & group_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up_c = tl.load(
        up_coeff + row[:, None] * GROUPS + group[None, :],
        mask=row_mask[:, None] & group_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gate = gate_acc + tl.sum(gate_c * sums[None, :], axis=1)
    up = up_acc + tl.sum(up_c * sums[None, :], axis=1)
    tl.store(gate_output + row, gate, mask=row_mask)
    tl.store(up_output + row, up, mask=row_mask)
    tl.store(swiglu_output + row, gate * tl.sigmoid(gate) * up, mask=row_mask)


def launch_residual_tile(
    packed: torch.Tensor,
    alpha: torch.Tensor,
    device_x: torch.Tensor,
    output: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_rows: int = 1,
    num_warps: int = 4,
) -> None:
    """Launch the resident residual dot for one independently transferred tile."""
    if packed.device.type != "cuda" or alpha.device.type != "cuda":
        raise ValueError("packed and alpha must be CUDA tensors")
    if device_x.device.type != "cuda" or output.device.type != "cuda":
        raise ValueError("activation and output must be CUDA tensors")
    if rows < 1 or cols < 32 or cols % 32 or block_rows not in (1, 2, 4, 8):
        raise ValueError("invalid tile dimensions")
    if packed.shape != (rows, cols // 2):
        raise ValueError("packed shape does not match tile dimensions")
    if alpha.shape != (rows, cols // 32):
        raise ValueError("alpha shape does not match tile dimensions")
    if device_x.numel() != cols or output.numel() != rows:
        raise ValueError("activation/output shape does not match tile dimensions")
    _residual_dot[(triton.cdiv(rows, block_rows),)](
        packed, alpha, device_x, output,
        ROWS=rows, COLS=cols,
        BLOCK_ROWS=block_rows, BLOCK_COLS=triton.next_power_of_2(cols),
        num_warps=num_warps, enable_fp_fusion=False,
    )


def launch_fused_gate_up_tile(
    gate_packed: torch.Tensor,
    gate_alpha: torch.Tensor,
    up_packed: torch.Tensor,
    up_alpha: torch.Tensor,
    device_x: torch.Tensor,
    gate_base: torch.Tensor,
    up_base: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    swiglu_output: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_rows: int = 1,
    num_warps: int = 4,
) -> None:
    tensors = (
        gate_packed, gate_alpha, up_packed, up_alpha, device_x,
        gate_base, up_base, gate_output, up_output, swiglu_output,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("fused gate/up tile requires CUDA tensors")
    if rows < 1 or cols < 32 or cols % 32 or block_rows not in (1, 2, 4, 8):
        raise ValueError("invalid fused tile dimensions")
    if gate_packed.shape != (rows, cols // 2) or up_packed.shape != (rows, cols // 2):
        raise ValueError("packed shape does not match fused tile dimensions")
    if gate_alpha.shape != (rows, cols // 32) or up_alpha.shape != (rows, cols // 32):
        raise ValueError("alpha shape does not match fused tile dimensions")
    if any(tensor.numel() != rows for tensor in (
        gate_base, up_base, gate_output, up_output, swiglu_output,
    )):
        raise ValueError("base/output shape does not match fused tile rows")
    if device_x.numel() != cols:
        raise ValueError("activation shape does not match fused tile columns")
    _fused_gate_up_swiglu_tile[(triton.cdiv(rows, block_rows),)](
        gate_packed, gate_alpha, up_packed, up_alpha, device_x,
        gate_base, up_base, gate_output, up_output, swiglu_output,
        ROWS=rows, COLS=cols,
        BLOCK_ROWS=block_rows, BLOCK_COLS=triton.next_power_of_2(cols),
        num_warps=num_warps, enable_fp_fusion=False,
    )


def launch_fused_gate_up_residual_tile(
    gate_packed: torch.Tensor,
    gate_alpha: torch.Tensor,
    up_packed: torch.Tensor,
    up_alpha: torch.Tensor,
    device_x: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_rows: int = 1,
    num_warps: int = 8,
) -> None:
    tensors = (
        gate_packed, gate_alpha, up_packed, up_alpha,
        device_x, gate_output, up_output,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("fused residual tile requires CUDA tensors")
    if rows < 1 or cols < 32 or cols % 32 or block_rows not in (1, 2, 4, 8):
        raise ValueError("invalid fused residual tile dimensions")
    if gate_packed.shape != (rows, cols // 2) or up_packed.shape != (rows, cols // 2):
        raise ValueError("packed shape does not match fused residual tile dimensions")
    if gate_alpha.shape != (rows, cols // 32) or up_alpha.shape != (rows, cols // 32):
        raise ValueError("alpha shape does not match fused residual tile dimensions")
    if device_x.numel() != cols or gate_output.numel() != rows or up_output.numel() != rows:
        raise ValueError("activation/output shape does not match fused residual tile")
    _fused_gate_up_residual_tile[(triton.cdiv(rows, block_rows),)](
        gate_packed, gate_alpha, up_packed, up_alpha, device_x,
        gate_output, up_output,
        ROWS=rows, COLS=cols,
        BLOCK_ROWS=block_rows, BLOCK_COLS=triton.next_power_of_2(cols),
        num_warps=num_warps, enable_fp_fusion=False,
    )


def launch_fused_gate_up_base_residual(
    gate_packed: torch.Tensor,
    gate_alpha: torch.Tensor,
    up_packed: torch.Tensor,
    up_alpha: torch.Tensor,
    gate_coeff: torch.Tensor,
    up_coeff: torch.Tensor,
    group_sums: torch.Tensor,
    device_x: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    swiglu_output: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_rows: int = 1,
    num_warps: int = 8,
    block_cols: int = 512,
) -> None:
    tensors = (
        gate_packed, gate_alpha, up_packed, up_alpha,
        gate_coeff, up_coeff, group_sums, device_x,
        gate_output, up_output, swiglu_output,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("fused base/residual path requires CUDA tensors")
    groups = cols // 32
    if rows < 1 or cols < 32 or cols % 32 or block_rows not in (1, 2, 4, 8):
        raise ValueError("invalid fused base/residual dimensions")
    if gate_packed.shape != (rows, cols // 2) or up_packed.shape != (rows, cols // 2):
        raise ValueError("packed shape does not match fused base/residual dimensions")
    if gate_alpha.shape != (rows, groups) or up_alpha.shape != (rows, groups):
        raise ValueError("alpha shape does not match fused base/residual dimensions")
    if gate_coeff.shape != (rows, groups) or up_coeff.shape != (rows, groups):
        raise ValueError("coefficient shape does not match fused base/residual dimensions")
    if group_sums.numel() != groups or device_x.numel() != cols:
        raise ValueError("activation/group shape does not match fused base/residual dimensions")
    if any(tensor.numel() != rows for tensor in (gate_output, up_output, swiglu_output)):
        raise ValueError("output shape does not match fused base/residual dimensions")
    if block_cols <= 0 or block_cols % 32:
        raise ValueError("block_cols must be a positive multiple of 32")
    block_cols = min(int(block_cols), cols)
    _fused_gate_up_base_residual_tiled[(triton.cdiv(rows, block_rows),)](
        gate_packed, gate_alpha, up_packed, up_alpha,
        gate_coeff, up_coeff, group_sums, device_x,
        gate_output, up_output, swiglu_output,
        ROWS=rows, COLS=cols, GROUPS=groups,
        BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
        BLOCK_GROUPS=triton.next_power_of_2(groups),
        num_warps=num_warps, num_stages=2, enable_fp_fusion=True,
    )


def launch_merge_swiglu(
    gate_residual: torch.Tensor,
    up_residual: torch.Tensor,
    gate_base: torch.Tensor,
    up_base: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    swiglu_output: torch.Tensor,
    *,
    rows: int,
) -> None:
    tensors = (
        gate_residual, up_residual, gate_base, up_base,
        gate_output, up_output, swiglu_output,
    )
    if any(tensor.device.type != "cuda" or tensor.numel() != rows for tensor in tensors):
        raise ValueError("merge tensors must be CUDA vectors matching rows")
    _merge_swiglu[(triton.cdiv(rows, 256),)](
        gate_residual, up_residual, gate_base, up_base,
        gate_output, up_output, swiglu_output,
        ROWS=rows, BLOCK=256, num_warps=4, enable_fp_fusion=False,
    )


@triton.jit
def _merge_swiglu(gate_r, up_r, gate_base, up_base, gate, up, output,
                  ROWS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < ROWS
    g = tl.load(gate_r + index, mask, other=0) + tl.load(gate_base + index, mask, other=0)
    u = tl.load(up_r + index, mask, other=0) + tl.load(up_base + index, mask, other=0)
    tl.store(gate + index, g, mask)
    tl.store(up + index, u, mask)
    tl.store(output + index, g * tl.sigmoid(g) * u, mask)


class ResidentGateUp:
    """Single-flight gate/up probe. Down and layer paging are deliberately outside its scope."""

    def __init__(self, artifact: ResidentArtifact, *, block_rows: int = 1, num_warps: int = 4):
        if block_rows not in (1, 2, 4, 8) or num_warps not in (4, 8):
            raise ValueError("invalid kernel launch configuration")
        if not all(p in artifact.projections for p in ("gate", "up")):
            raise ValueError("both gate and up must have compiled Q4_K projections")
        self.rows = artifact.projections["gate"]["rows"]
        self.cols = artifact.projections["gate"]["cols"]
        if any((artifact.projections[p]["rows"], artifact.projections[p]["cols"]) != (self.rows, self.cols)
               for p in ("gate", "up")):
            raise ValueError("gate/up dimensions differ")
        self.block_rows, self.num_warps = block_rows, num_warps
        self.stream = torch.cuda.Stream()
        self.host_x = torch.empty(self.cols, dtype=torch.float32, pin_memory=True)
        self.host_base = {p: torch.empty(self.rows, dtype=torch.float32, pin_memory=True) for p in ("gate", "up")}
        self.coefficient = {p: artifact.arrays[p]["coefficient"] for p in ("gate", "up")}
        self.device_x = torch.empty(self.cols, dtype=torch.float32, device="cuda")
        self.weights = {}
        self.base, self.residual, self.output = {}, {}, {}
        self.resident_bytes = 0
        for p in ("gate", "up"):
            self.weights[p] = {}
            for kind in ("residual", "alpha"):
                value = torch.from_numpy(np.array(artifact.arrays[p][kind], copy=True)).cuda()
                self.weights[p][kind] = value
                self.resident_bytes += value.numel() * value.element_size()
            self.base[p] = torch.empty(self.rows, device="cuda")
            self.residual[p] = torch.empty(self.rows, device="cuda")
            self.output[p] = torch.empty(self.rows, device="cuda")
        self.output["swiglu"] = torch.empty(self.rows, device="cuda")
        torch.cuda.synchronize()
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
        self.traffic = dict(weight_upload_bytes=self.resident_bytes, dynamic_h2d_bytes=0,
                            validation_d2h_bytes=0, weight_h2d_bytes_per_run=0)

    def launch_residuals(self) -> None:
        resources = []
        for p in ("gate", "up"):
            kernel = _residual_dot[(triton.cdiv(self.rows, self.block_rows),)](
                self.weights[p]["residual"], self.weights[p]["alpha"], self.device_x,
                self.residual[p], ROWS=self.rows, COLS=self.cols,
                BLOCK_ROWS=self.block_rows, BLOCK_COLS=triton.next_power_of_2(self.cols),
                num_warps=self.num_warps, enable_fp_fusion=False,
            )
            resources.append({"projection": p, "registers_per_thread": kernel.n_regs,
                              "spills": kernel.n_spills, "shared_bytes": kernel.metadata.shared})
        self.kernel_resources = resources

    def run(self, x: np.ndarray, *, return_outputs: bool = True, down=None) -> dict:
        values = np.asarray(x, dtype=np.float32)
        if values.shape != (self.cols,) or not np.isfinite(values).all():
            raise ValueError("finite one-token activation required")
        if down is not None and (down.cols != self.rows or down.rows != self.cols):
            raise ValueError("down projection dimensions must reverse gate/up dimensions")
        # Completion at the end of each run protects pinned buffers from premature reuse.
        begin = time.perf_counter()
        self.host_x.numpy()[:] = values
        e0, e1, e2, e3, e4, e5, e6 = self.events
        with torch.cuda.stream(self.stream):
            e0.record()
            self.device_x.copy_(self.host_x, non_blocking=True)
            e1.record()
            self.launch_residuals()
            e2.record()
        cpu_begin = time.perf_counter()
        sums = values.astype(np.float64).reshape(-1, 32).sum(axis=1)
        for p in ("gate", "up"):
            self.host_base[p].numpy()[:] = self.coefficient[p] @ sums
        cpu_ms = (time.perf_counter() - cpu_begin) * 1000
        with torch.cuda.stream(self.stream):
            e3.record()
            for p in ("gate", "up"):
                self.base[p].copy_(self.host_base[p], non_blocking=True)
            e4.record()
            _merge_swiglu[(triton.cdiv(self.rows, 256),)](
                self.residual["gate"], self.residual["up"], self.base["gate"], self.base["up"],
                self.output["gate"], self.output["up"], self.output["swiglu"],
                ROWS=self.rows, BLOCK=256, num_warps=4, enable_fp_fusion=False,
            )
            e5.record()
            if down is not None:
                down.launch(self.output["swiglu"])
            e6.record()
        e6.synchronize()
        wall_ms = (time.perf_counter() - begin) * 1000
        dynamic = 4 * (self.cols + 2 * self.rows)
        self.traffic["dynamic_h2d_bytes"] += dynamic
        result = {
            "timing": {
                "wall_ms": wall_ms, "cpu_base_ms": cpu_ms,
                "activation_h2d_ms": e0.elapsed_time(e1),
                "residual_stream_span_ms": e1.elapsed_time(e2),
                "exposed_cpu_submission_gap_ms": e2.elapsed_time(e3),
                "base_h2d_ms": e3.elapsed_time(e4),
                "merge_stream_span_ms": e4.elapsed_time(e5),
                "down_stream_span_ms": e5.elapsed_time(e6) if down is not None else 0.0,
                "stream_span_ms": e0.elapsed_time(e6),
            },
            "dynamic_h2d_bytes": dynamic,
        }
        if return_outputs:
            result.update({p: tensor.cpu().numpy() for p, tensor in self.output.items()})
            if down is not None:
                result["down"] = down.output.cpu().numpy()
                self.traffic["validation_d2h_bytes"] += down.rows * 4
            self.traffic["validation_d2h_bytes"] += self.rows * 3 * 4
        return result

    def graph_kernel_ms(self, repeats: int = 20) -> float:
        """Launch-overhead-reduced kernel span, NOT hardware occupancy."""
        with torch.cuda.stream(self.stream):
            self.launch_residuals()
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self.launch_residuals()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            begin.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
        end.synchronize()
        return begin.elapsed_time(end) / repeats
