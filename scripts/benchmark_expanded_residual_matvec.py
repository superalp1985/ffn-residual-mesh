from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resident_residual_format import ResidentArtifact


@triton.jit
def _fp16_pair_gemv(
    gate, up, x, gate_out, up_out,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    acc_gate = tl.zeros((BLOCK_ROWS, 1), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_ROWS, 1), dtype=tl.float32)
    for col_start in range(0, COLS, BLOCK_COLS):
        col = col_start + tl.arange(0, BLOCK_COLS)
        col_mask = col < COLS
        x_tile = tl.load(x + col, mask=col_mask, other=0.0).to(tl.float16)
        gate_tile = tl.load(
            gate + row[:, None] * COLS + col[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        up_tile = tl.load(
            up + row[:, None] * COLS + col[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc_gate += tl.dot(gate_tile, x_tile[:, None], out_dtype=tl.float32)
        acc_up += tl.dot(up_tile, x_tile[:, None], out_dtype=tl.float32)
    tl.store(gate_out + row, tl.reshape(acc_gate, (BLOCK_ROWS,)), mask=row_mask)
    tl.store(up_out + row, tl.reshape(acc_up, (BLOCK_ROWS,)), mask=row_mask)


def expand_residual(array: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    packed = np.asarray(array, dtype=np.uint8)
    low = packed & 0x0F
    high = packed >> 4
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
    codes[:, 0::2] = np.where(low < 8, low, low.astype(np.int16) - 16)
    codes[:, 1::2] = np.where(high < 8, high, high.astype(np.int16) - 16)
    scales = np.repeat(np.asarray(alpha, dtype=np.float32), 32, axis=1)
    return (codes.astype(np.float32) * scales).astype(np.float16)


def measure(operation, repeats: int) -> dict[str, float]:
    values: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        operation()
        end.record()
        end.synchronize()
        values.append(float(begin.elapsed_time(end)))
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def run(args: argparse.Namespace) -> None:
    with ResidentArtifact.open(args.artifact, verify_hashes=False) as artifact:
        gate = expand_residual(
            artifact.arrays["gate"]["residual"],
            artifact.arrays["gate"]["alpha"],
        )
        up = expand_residual(
            artifact.arrays["up"]["residual"],
            artifact.arrays["up"]["alpha"],
        )
        print(
            {
                "shape": gate.shape,
                "gate_bytes": int(gate.nbytes),
                "up_bytes": int(up.nbytes),
                "cat_bytes": int((gate.nbytes + up.nbytes)),
            }
        )
        gate_d = torch.from_numpy(gate).cuda()
        up_d = torch.from_numpy(up).cuda()
        cat_d = torch.cat((gate_d, up_d), dim=0)
        x = torch.randn(gate.shape[1], device="cuda", dtype=torch.float16)
        gate_out = torch.empty(gate.shape[0], device="cuda", dtype=torch.float16)
        up_out = torch.empty(up.shape[0], device="cuda", dtype=torch.float16)
        cat_out = torch.empty(cat_d.shape[0], device="cuda", dtype=torch.float16)
        row_out = torch.empty(1, cat_d.shape[0], device="cuda", dtype=torch.float16)
        pair_gate_out = torch.empty(gate.shape[0], device="cuda", dtype=torch.float32)
        pair_up_out = torch.empty(up.shape[0], device="cuda", dtype=torch.float32)

        operations = {
            "gate_mv": lambda: torch.mv(gate_d, x, out=gate_out),
            "up_mv": lambda: torch.mv(up_d, x, out=up_out),
            "two_mv": lambda: (
                torch.mv(gate_d, x, out=gate_out),
                torch.mv(up_d, x, out=up_out),
            ),
            "cat_mv": lambda: torch.mv(cat_d, x, out=cat_out),
            "cat_mm_w_times_x": lambda: torch.mm(
                cat_d, x.view(-1, 1), out=cat_out.view(-1, 1)
            ),
            "cat_mm_x_times_wt": lambda: torch.mm(
                x.view(1, -1), cat_d.t(), out=row_out
            ),
        }
        for block_rows, block_cols, warps in (
            (8, 64, 4),
            (16, 64, 4),
            (16, 128, 4),
            (32, 64, 8),
            (32, 128, 8),
        ):
            grid = (triton.cdiv(gate.shape[0], block_rows),)
            def launch_pair(
                block_rows=block_rows,
                block_cols=block_cols,
                warps=warps,
                grid=grid,
            ):
                _fp16_pair_gemv[grid](
                    gate_d, up_d, x, pair_gate_out, pair_up_out,
                    ROWS=gate.shape[0], COLS=gate.shape[1],
                    BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
                    num_warps=warps, num_stages=2,
                )
            operations[
                f"triton_pair_r{block_rows}_k{block_cols}_w{warps}"
            ] = launch_pair
        for operation in operations.values():
            for _ in range(5):
                operation()
        torch.cuda.synchronize()
        for name, operation in operations.items():
            print({name: measure(operation, args.repeats)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
