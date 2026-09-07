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
def _fused_q4k_grouped(
    raw,
    x,
    output,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_QBLOCKS: tl.constexpr,
):
    """Q4_K GEMV that decodes each packed Q4_K block only once per CTA tile."""
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    qblock_lane = tl.arange(0, BLOCK_QBLOCKS)
    byte = tl.arange(0, 32)
    qblocks = COLS // 256
    row_bytes = qblocks * 144
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for qblock_start in range(0, qblocks, BLOCK_QBLOCKS):
        qblock = qblock_start + qblock_lane
        qblock_mask = qblock < qblocks
        block = raw + row[:, None] * row_bytes + qblock[None, :] * 144
        header_mask = row_mask[:, None] & qblock_mask[None, :]
        d_bits = (
            tl.load(block, header_mask, other=0).to(tl.uint32)
            | (tl.load(block + 1, header_mask, other=0).to(tl.uint32) << 8)
        )
        m_bits = (
            tl.load(block + 2, header_mask, other=0).to(tl.uint32)
            | (tl.load(block + 3, header_mask, other=0).to(tl.uint32) << 8)
        )
        d = tl.cast(d_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
        dm = tl.cast(m_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)

        for pair in range(0, 4):
            first_group = pair * 2
            second_group = first_group + 1
            first_index = first_group % 4
            second_index = second_group % 4
            first_scale_low = tl.load(block + 4 + first_index, header_mask, other=0).to(tl.int32)
            first_min_low = tl.load(block + 8 + first_index, header_mask, other=0).to(tl.int32)
            first_mix = tl.load(block + 12 + first_index, header_mask, other=0).to(tl.int32)
            second_scale_low = tl.load(block + 4 + second_index, header_mask, other=0).to(tl.int32)
            second_min_low = tl.load(block + 8 + second_index, header_mask, other=0).to(tl.int32)
            second_mix = tl.load(block + 12 + second_index, header_mask, other=0).to(tl.int32)
            first_scale = tl.where(
                first_group < 4,
                first_scale_low & 63,
                (first_mix & 15) | ((first_scale_low >> 2) & 48),
            )
            first_minimum = tl.where(
                first_group < 4,
                first_min_low & 63,
                (first_mix >> 4) | ((first_min_low >> 2) & 48),
            )
            second_scale = tl.where(
                second_group < 4,
                second_scale_low & 63,
                (second_mix & 15) | ((second_scale_low >> 2) & 48),
            )
            second_minimum = tl.where(
                second_group < 4,
                second_min_low & 63,
                (second_mix >> 4) | ((second_min_low >> 2) & 48),
            )
            value_mask = header_mask[:, :, None]
            packed = tl.load(
                block[:, :, None] + 16 + pair * 32 + byte[None, None, :],
                value_mask,
                other=0,
            ).to(tl.int32)
            first_q = packed & 15
            second_q = (packed >> 4) & 15
            activation_offset = qblock[:, None] * 256 + pair * 64 + byte[None, :]
            first_x = tl.load(
                x + activation_offset,
                mask=qblock_mask[:, None],
                other=0.0,
            )
            second_x = tl.load(
                x + activation_offset + 32,
                mask=qblock_mask[:, None],
                other=0.0,
            )
            first_weight = (
                d[:, :, None] * first_scale[:, :, None].to(tl.float32)
                * first_q.to(tl.float32)
                - dm[:, :, None] * first_minimum[:, :, None].to(tl.float32)
            )
            second_weight = (
                d[:, :, None] * second_scale[:, :, None].to(tl.float32)
                * second_q.to(tl.float32)
                - dm[:, :, None] * second_minimum[:, :, None].to(tl.float32)
            )
            block_dot = tl.sum(
                first_weight * first_x[None, :, :]
                + second_weight * second_x[None, :, :],
                axis=2,
            )
            acc += tl.sum(block_dot, axis=1)
    tl.store(output + row, acc, row_mask)


@triton.jit
def _fused_q5k_grouped(
    raw,
    x,
    output,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_QBLOCKS: tl.constexpr,
):
    """Q5_K GEMV sharing packed headers, low nibbles, and high-bit planes."""
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    qblock_lane = tl.arange(0, BLOCK_QBLOCKS)
    byte = tl.arange(0, 32)
    qblocks = COLS // 256
    row_bytes = qblocks * 176
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for qblock_start in range(0, qblocks, BLOCK_QBLOCKS):
        qblock = qblock_start + qblock_lane
        qblock_mask = qblock < qblocks
        block = raw + row[:, None] * row_bytes + qblock[None, :] * 176
        header_mask = row_mask[:, None] & qblock_mask[None, :]
        d_bits = (
            tl.load(block, header_mask, other=0).to(tl.uint32)
            | (tl.load(block + 1, header_mask, other=0).to(tl.uint32) << 8)
        )
        m_bits = (
            tl.load(block + 2, header_mask, other=0).to(tl.uint32)
            | (tl.load(block + 3, header_mask, other=0).to(tl.uint32) << 8)
        )
        d = tl.cast(d_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
        dm = tl.cast(m_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)

        for pair in range(0, 4):
            first_group = pair * 2
            second_group = first_group + 1
            first_index = first_group % 4
            second_index = second_group % 4
            first_scale_low = tl.load(block + 4 + first_index, header_mask, other=0).to(tl.int32)
            first_min_low = tl.load(block + 8 + first_index, header_mask, other=0).to(tl.int32)
            first_mix = tl.load(block + 12 + first_index, header_mask, other=0).to(tl.int32)
            second_scale_low = tl.load(block + 4 + second_index, header_mask, other=0).to(tl.int32)
            second_min_low = tl.load(block + 8 + second_index, header_mask, other=0).to(tl.int32)
            second_mix = tl.load(block + 12 + second_index, header_mask, other=0).to(tl.int32)
            first_scale = tl.where(
                first_group < 4,
                first_scale_low & 63,
                (first_mix & 15) | ((first_scale_low >> 2) & 48),
            )
            first_minimum = tl.where(
                first_group < 4,
                first_min_low & 63,
                (first_mix >> 4) | ((first_min_low >> 2) & 48),
            )
            second_scale = tl.where(
                second_group < 4,
                second_scale_low & 63,
                (second_mix & 15) | ((second_scale_low >> 2) & 48),
            )
            second_minimum = tl.where(
                second_group < 4,
                second_min_low & 63,
                (second_mix >> 4) | ((second_min_low >> 2) & 48),
            )
            value_mask = header_mask[:, :, None]
            packed_low = tl.load(
                block[:, :, None] + 48 + pair * 32 + byte[None, None, :],
                value_mask,
                other=0,
            ).to(tl.int32)
            high_plane = tl.load(
                block[:, :, None] + 16 + byte[None, None, :],
                value_mask,
                other=0,
            ).to(tl.int32)
            first_q = (packed_low & 15) | (((high_plane >> first_group) & 1) << 4)
            second_q = ((packed_low >> 4) & 15) | (((high_plane >> second_group) & 1) << 4)
            activation_offset = qblock[:, None] * 256 + pair * 64 + byte[None, :]
            first_x = tl.load(
                x + activation_offset,
                mask=qblock_mask[:, None],
                other=0.0,
            )
            second_x = tl.load(
                x + activation_offset + 32,
                mask=qblock_mask[:, None],
                other=0.0,
            )
            first_weight = (
                d[:, :, None] * first_scale[:, :, None].to(tl.float32)
                * first_q.to(tl.float32)
                - dm[:, :, None] * first_minimum[:, :, None].to(tl.float32)
            )
            second_weight = (
                d[:, :, None] * second_scale[:, :, None].to(tl.float32)
                * second_q.to(tl.float32)
                - dm[:, :, None] * second_minimum[:, :, None].to(tl.float32)
            )
            block_dot = tl.sum(
                first_weight * first_x[None, :, :]
                + second_weight * second_x[None, :, :],
                axis=2,
            )
            acc += tl.sum(block_dot, axis=1)
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


@triton.jit
def _fused_iq4xs(
    raw,
    x,
    output,
    kvalues,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_QBLOCKS: tl.constexpr,
):
    """IQ4_XS GEMV with one scale/header decode per 32-value group."""
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    qblock_lane = tl.arange(0, BLOCK_QBLOCKS)
    byte = tl.arange(0, 16)
    qblocks = COLS // 256
    row_bytes = qblocks * 136
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for qblock_start in range(0, qblocks, BLOCK_QBLOCKS):
        qblock = qblock_start + qblock_lane
        qblock_mask = qblock < qblocks
        block = raw + row[:, None] * row_bytes + qblock[None, :] * 136
        header_mask = row_mask[:, None] & qblock_mask[None, :]
        d_bits = (
            tl.load(block, header_mask, other=0).to(tl.uint32)
            | (tl.load(block + 1, header_mask, other=0).to(tl.uint32) << 8)
        )
        d = tl.cast(d_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
        scales_h = (
            tl.load(block + 2, header_mask, other=0).to(tl.int32)
            | (tl.load(block + 3, header_mask, other=0).to(tl.int32) << 8)
        )

        for group in range(0, 8):
            scale_byte = tl.load(block + 4 + group // 2, header_mask, other=0).to(tl.int32)
            scale_low = (scale_byte >> ((group % 2) * 4)) & 15
            scale = (scale_low | (((scales_h >> (group * 2)) & 3) << 4)) - 32
            value_mask = header_mask[:, :, None]
            packed = tl.load(
                block[:, :, None] + 8 + group * 16 + byte[None, None, :],
                value_mask,
                other=0,
            ).to(tl.int32)
            first_q = packed & 15
            second_q = (packed >> 4) & 15
            # Each group stores columns 0..15 in low nibbles and 16..31 in high nibbles.
            activation_offset = qblock[:, None] * 256 + group * 32 + byte[None, :]
            first_x = tl.load(
                x + activation_offset,
                mask=qblock_mask[:, None],
                other=0.0,
            )
            second_x = tl.load(
                x + activation_offset + 16,
                mask=qblock_mask[:, None],
                other=0.0,
            )
            first_weight = (
                d[:, :, None] * scale[:, :, None].to(tl.float32)
                * tl.load(kvalues + first_q, first_q < 16, other=0).to(tl.float32)
            )
            second_weight = (
                d[:, :, None] * scale[:, :, None].to(tl.float32)
                * tl.load(kvalues + second_q, second_q < 16, other=0).to(tl.float32)
            )
            block_dot = tl.sum(
                first_weight * first_x[None, :, :]
                + second_weight * second_x[None, :, :],
                axis=2,
            )
            acc += tl.sum(block_dot, axis=1)
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
        kernel: str = "legacy",
        block_qblocks: int = 1,
    ):
        if raw.dtype != np.uint8 or raw.ndim != 2 or cols % 256 or raw.shape[1] != cols // 256 * 144:
            raise ValueError("expected row-major raw Q4_K tensor")
        if chunk_cols <= 0 or chunk_cols % 256:
            raise ValueError("chunk_cols must be a positive multiple of 256")
        if kernel not in ("legacy", "grouped"):
            raise ValueError("unsupported Q4_K kernel")
        if block_qblocks not in (1, 2, 4):
            raise ValueError("block_qblocks must be 1, 2, or 4")
        self.raw = torch.from_numpy(np.array(raw, copy=True)).cuda()
        self.rows, self.cols = raw.shape[0], cols
        self.block_rows, self.num_warps = block_rows, num_warps
        self.chunk_cols = min(int(chunk_cols), cols)
        self.kernel, self.block_qblocks = kernel, block_qblocks
        self.output = torch.empty(self.rows, device="cuda")

    def launch(self, device_x: torch.Tensor) -> None:
        if self.kernel == "grouped":
            _fused_q4k_grouped[(triton.cdiv(self.rows, self.block_rows),)](
                self.raw, device_x, self.output, ROWS=self.rows, COLS=self.cols,
                BLOCK_ROWS=self.block_rows, BLOCK_QBLOCKS=self.block_qblocks,
                num_warps=self.num_warps, enable_fp_fusion=False,
            )
            return
        _fused_q4k[(triton.cdiv(self.rows, self.block_rows),)](
            self.raw, device_x, self.output, ROWS=self.rows, COLS=self.cols,
            BLOCK_ROWS=self.block_rows, BLOCK_COLS=triton.next_power_of_2(self.chunk_cols),
            num_warps=self.num_warps, enable_fp_fusion=False,
        )


class DirectQ5Projection:
    def __init__(
        self,
        raw: np.ndarray,
        cols: int,
        *,
        block_rows: int = 2,
        num_warps: int = 2,
        block_qblocks: int = 1,
    ):
        if raw.dtype != np.uint8 or raw.ndim != 2 or cols % 256 or raw.shape[1] != cols // 256 * 176:
            raise ValueError("expected row-major raw Q5_K tensor")
        if block_qblocks not in (1, 2, 4):
            raise ValueError("block_qblocks must be 1, 2, or 4")
        self.raw = torch.from_numpy(np.array(raw, copy=True)).cuda()
        self.rows, self.cols = raw.shape[0], cols
        self.block_rows, self.num_warps = block_rows, num_warps
        self.block_qblocks = block_qblocks
        self.kernel = "grouped_q5_k"
        self.output = torch.empty(self.rows, device="cuda")

    def launch(self, device_x: torch.Tensor) -> None:
        _fused_q5k_grouped[(triton.cdiv(self.rows, self.block_rows),)](
            self.raw,
            device_x,
            self.output,
            ROWS=self.rows,
            COLS=self.cols,
            BLOCK_ROWS=self.block_rows,
            BLOCK_QBLOCKS=self.block_qblocks,
            num_warps=self.num_warps,
            enable_fp_fusion=False,
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


class DirectIQ4XSProjection:
    def __init__(
        self,
        raw: np.ndarray,
        cols: int,
        *,
        block_rows: int = 2,
        num_warps: int = 2,
        block_qblocks: int = 1,
    ):
        if raw.dtype != np.uint8 or raw.ndim != 2 or cols % 256 or raw.shape[1] != cols // 256 * 136:
            raise ValueError("expected row-major raw IQ4_XS tensor")
        if block_qblocks not in (1, 2, 4):
            raise ValueError("block_qblocks must be 1, 2, or 4")
        self.raw = torch.from_numpy(np.array(raw, copy=True)).cuda()
        self.kvalues = torch.tensor(
            (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113),
            dtype=torch.float32,
            device="cuda",
        )
        self.rows, self.cols = raw.shape[0], cols
        self.block_rows, self.num_warps = block_rows, num_warps
        self.block_qblocks = block_qblocks
        self.kernel = "grouped_iq4_xs"
        self.output = torch.empty(self.rows, device="cuda")

    def launch(self, device_x: torch.Tensor) -> None:
        _fused_iq4xs[(triton.cdiv(self.rows, self.block_rows),)](
            self.raw,
            device_x,
            self.output,
            self.kvalues,
            ROWS=self.rows,
            COLS=self.cols,
            BLOCK_ROWS=self.block_rows,
            BLOCK_QBLOCKS=self.block_qblocks,
            num_warps=self.num_warps,
            enable_fp_fusion=False,
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
def _residual_dot_grouped(
    packed, alpha, x, output,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_GROUPS: tl.constexpr,
):
    """Decode one 32-code group from each packed 16-byte segment.

    This is intentionally a separate candidate from ``_residual_dot``.  The
    grouped layout loads the group scale once, loads every packed byte once,
    and accumulates the low/high nibbles together before moving to the next
    group tile.
    """
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    group_lane = tl.arange(0, BLOCK_GROUPS)
    byte = tl.arange(0, 16)
    groups = COLS // 32
    accum = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for group_start in range(0, groups, BLOCK_GROUPS):
        group = group_start + group_lane
        group_mask = group < groups
        mask = row_mask[:, None, None] & group_mask[None, :, None]
        packed_offset = (
            row[:, None, None] * (COLS // 2)
            + group[None, :, None] * 16
            + byte[None, None, :]
        )
        value = tl.load(packed + packed_offset, mask=mask, other=0).to(tl.int32)
        low = value & 15
        high = (value >> 4) & 15
        low = tl.where(low >= 8, low - 16, low).to(tl.float32)
        high = tl.where(high >= 8, high - 16, high).to(tl.float32)
        scale = tl.load(
            alpha + row[:, None] * groups + group[None, :],
            mask=row_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        low_x = tl.load(
            x + group[:, None] * 32 + byte[None, :] * 2,
            mask=group_mask[:, None],
            other=0.0,
        )
        high_x = tl.load(
            x + group[:, None] * 32 + byte[None, :] * 2 + 1,
            mask=group_mask[:, None],
            other=0.0,
        )
        group_dot = tl.sum(
            (low * low_x[None, :, :] + high * high_x[None, :, :])
            * scale[:, :, None],
            axis=2,
        )
        accum += tl.sum(group_dot, axis=1)

    tl.store(output + row, accum, row_mask)


@triton.jit
def _fused_gate_up_residual_grouped(
    gate_packed, gate_alpha, up_packed, up_alpha, x,
    gate_output, up_output,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_GROUPS: tl.constexpr,
):
    """Grouped gate/up residual GEMV sharing one activation decode per CTA."""
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    group_lane = tl.arange(0, BLOCK_GROUPS)
    byte = tl.arange(0, 16)
    groups = COLS // 32
    gate_accum = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    up_accum = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for group_start in range(0, groups, BLOCK_GROUPS):
        group = group_start + group_lane
        group_mask = group < groups
        mask = row_mask[:, None, None] & group_mask[None, :, None]
        packed_offset = (
            row[:, None, None] * (COLS // 2)
            + group[None, :, None] * 16
            + byte[None, None, :]
        )
        low_x = tl.load(
            x + group[:, None] * 32 + byte[None, :] * 2,
            mask=group_mask[:, None],
            other=0.0,
        )
        high_x = tl.load(
            x + group[:, None] * 32 + byte[None, :] * 2 + 1,
            mask=group_mask[:, None],
            other=0.0,
        )

        gate_value = tl.load(
            gate_packed + packed_offset, mask=mask, other=0
        ).to(tl.int32)
        gate_low = gate_value & 15
        gate_high = (gate_value >> 4) & 15
        gate_low = tl.where(gate_low >= 8, gate_low - 16, gate_low).to(tl.float32)
        gate_high = tl.where(gate_high >= 8, gate_high - 16, gate_high).to(tl.float32)
        gate_scale = tl.load(
            gate_alpha + row[:, None] * groups + group[None, :],
            mask=row_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        gate_dot = tl.sum(
            (gate_low * low_x[None, :, :] + gate_high * high_x[None, :, :])
            * gate_scale[:, :, None],
            axis=2,
        )
        gate_accum += tl.sum(gate_dot, axis=1)

        up_value = tl.load(
            up_packed + packed_offset, mask=mask, other=0
        ).to(tl.int32)
        up_low = up_value & 15
        up_high = (up_value >> 4) & 15
        up_low = tl.where(up_low >= 8, up_low - 16, up_low).to(tl.float32)
        up_high = tl.where(up_high >= 8, up_high - 16, up_high).to(tl.float32)
        up_scale = tl.load(
            up_alpha + row[:, None] * groups + group[None, :],
            mask=row_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        up_dot = tl.sum(
            (up_low * low_x[None, :, :] + up_high * high_x[None, :, :])
            * up_scale[:, :, None],
            axis=2,
        )
        up_accum += tl.sum(up_dot, axis=1)

    tl.store(gate_output + row, gate_accum, row_mask)
    tl.store(up_output + row, up_accum, row_mask)


def _launch_residual_dot(
    packed: torch.Tensor,
    alpha: torch.Tensor,
    device_x: torch.Tensor,
    output: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_rows: int,
    num_warps: int,
    kernel: str,
    block_groups: int = 32,
) -> object:
    if kernel == "legacy":
        return _residual_dot[(triton.cdiv(rows, block_rows),)](
            packed, alpha, device_x, output,
            ROWS=rows, COLS=cols,
            BLOCK_ROWS=block_rows, BLOCK_COLS=triton.next_power_of_2(cols),
            num_warps=num_warps, enable_fp_fusion=False,
        )
    if kernel == "grouped":
        return _residual_dot_grouped[(triton.cdiv(rows, block_rows),)](
            packed, alpha, device_x, output,
            ROWS=rows, COLS=cols,
            BLOCK_ROWS=block_rows, BLOCK_GROUPS=block_groups,
            num_warps=num_warps, enable_fp_fusion=False,
        )
    raise ValueError(f"unsupported residual kernel: {kernel}")


def _launch_fused_gate_up_residual_grouped(
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
    block_rows: int,
    num_warps: int,
    block_groups: int,
) -> object:
    return _fused_gate_up_residual_grouped[(triton.cdiv(rows, block_rows),)](
        gate_packed, gate_alpha, up_packed, up_alpha, device_x,
        gate_output, up_output,
        ROWS=rows, COLS=cols,
        BLOCK_ROWS=block_rows, BLOCK_GROUPS=block_groups,
        num_warps=num_warps, enable_fp_fusion=False,
    )


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
    kernel: str = "legacy",
    block_groups: int = 32,
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
    _launch_residual_dot(
        packed, alpha, device_x, output,
        rows=rows, cols=cols,
        block_rows=block_rows, num_warps=num_warps, kernel=kernel,
        block_groups=block_groups,
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

    def __init__(
        self,
        artifact: ResidentArtifact,
        *,
        block_rows: int = 1,
        num_warps: int = 4,
        residual_kernel: str = "legacy",
        residual_block_groups: int = 32,
    ):
        if block_rows not in (1, 2, 4, 8) or num_warps not in (4, 8):
            raise ValueError("invalid kernel launch configuration")
        if residual_kernel not in ("legacy", "grouped", "grouped_fused"):
            raise ValueError("invalid residual kernel")
        if residual_block_groups not in (8, 16, 32, 64, 128):
            raise ValueError("invalid residual block group count")
        if not all(p in artifact.projections for p in ("gate", "up")):
            raise ValueError("both gate and up must have compiled Q4_K projections")
        self.rows = artifact.projections["gate"]["rows"]
        self.cols = artifact.projections["gate"]["cols"]
        if any((artifact.projections[p]["rows"], artifact.projections[p]["cols"]) != (self.rows, self.cols)
               for p in ("gate", "up")):
            raise ValueError("gate/up dimensions differ")
        self.block_rows, self.num_warps = block_rows, num_warps
        self.residual_kernel = residual_kernel
        self.residual_block_groups = residual_block_groups
        self.stream = torch.cuda.Stream()
        self.host_x = torch.empty(self.cols, dtype=torch.float32, pin_memory=True)
        self.host_base = {p: torch.empty(self.rows, dtype=torch.float32, pin_memory=True) for p in ("gate", "up")}
        self.group_sums = np.empty(self.cols // 32, dtype=np.float32)
        self.coefficient = {
            p: np.array(artifact.arrays[p]["coefficient"], dtype=np.float32, order="C", copy=True)
            for p in ("gate", "up")
        }
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
        if self.residual_kernel == "grouped_fused":
            compiled = _launch_fused_gate_up_residual_grouped(
                self.weights["gate"]["residual"], self.weights["gate"]["alpha"],
                self.weights["up"]["residual"], self.weights["up"]["alpha"],
                self.device_x, self.residual["gate"], self.residual["up"],
                rows=self.rows, cols=self.cols,
                block_rows=self.block_rows, num_warps=self.num_warps,
                block_groups=self.residual_block_groups,
            )
            self.kernel_resources = [{
                "projection": "gate_up_fused",
                "registers_per_thread": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
            }]
            return
        for p in ("gate", "up"):
            compiled = _launch_residual_dot(
                self.weights[p]["residual"], self.weights[p]["alpha"], self.device_x,
                self.residual[p],
                rows=self.rows, cols=self.cols,
                block_rows=self.block_rows, num_warps=self.num_warps,
                kernel=self.residual_kernel,
                block_groups=self.residual_block_groups,
            )
            resources.append({"projection": p, "registers_per_thread": compiled.n_regs,
                              "spills": compiled.n_spills, "shared_bytes": compiled.metadata.shared})
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
        np.sum(values.reshape(-1, 32), axis=1, dtype=np.float32, out=self.group_sums)
        for p in ("gate", "up"):
            np.matmul(self.coefficient[p], self.group_sums, out=self.host_base[p].numpy())
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
