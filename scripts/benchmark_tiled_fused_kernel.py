from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resident_residual_format import ResidentArtifact
from resident_residual_cuda import launch_merge_swiglu
from resident_tiled_ffn import TiledResidentGateUp


@triton.jit
def fused_k_tiled(
    gate_packed, gate_alpha, up_packed, up_alpha,
    gate_coeff, up_coeff, group_sums, x,
    gate_output, up_output, swiglu_output,
    ROWS: tl.constexpr, COLS: tl.constexpr, GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    gate_acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for col_start in range(0, COLS, BLOCK_COLS):
        col = col_start + tl.arange(0, BLOCK_COLS)
        col_mask = col < COLS
        mask = row_mask[:, None] & col_mask[None, :]
        packed = row[:, None] * (COLS // 2) + col[None, :] // 2
        shift = (col[None, :] % 2) * 4

        gate_q = (
            tl.load(gate_packed + packed, mask=mask, other=0).to(tl.int32)
            >> shift
        ) & 15
        up_q = (
            tl.load(up_packed + packed, mask=mask, other=0).to(tl.int32)
            >> shift
        ) & 15
        gate_r = tl.where(gate_q >= 8, gate_q - 16, gate_q).to(tl.float32)
        up_r = tl.where(up_q >= 8, up_q - 16, up_q).to(tl.float32)
        group = col[None, :] // 32
        gate_scale = tl.load(
            gate_alpha + row[:, None] * GROUPS + group,
            mask=mask,
            other=0.0,
        )
        up_scale = tl.load(
            up_alpha + row[:, None] * GROUPS + group,
            mask=mask,
            other=0.0,
        )
        activation = tl.load(x + col, mask=col_mask, other=0.0)
        gate_acc += tl.sum(gate_r * gate_scale * activation[None, :], axis=1)
        up_acc += tl.sum(up_r * up_scale * activation[None, :], axis=1)

    group = tl.arange(0, BLOCK_GROUPS)
    group_mask = group < GROUPS
    sums = tl.load(group_sums + group, mask=group_mask, other=0.0)
    gate_c = tl.load(
        gate_coeff + row[:, None] * GROUPS + group[None, :],
        mask=row_mask[:, None] & group_mask[None, :],
        other=0.0,
    )
    up_c = tl.load(
        up_coeff + row[:, None] * GROUPS + group[None, :],
        mask=row_mask[:, None] & group_mask[None, :],
        other=0.0,
    )
    gate = gate_acc + tl.sum(gate_c * sums[None, :], axis=1)
    up = up_acc + tl.sum(up_c * sums[None, :], axis=1)
    tl.store(gate_output + row, gate, mask=row_mask)
    tl.store(up_output + row, up, mask=row_mask)
    tl.store(swiglu_output + row, gate * tl.sigmoid(gate) * up, mask=row_mask)


@triton.jit
def residual_k_tiled(
    gate_packed, gate_alpha, up_packed, up_alpha, x,
    gate_output, up_output,
    ROWS: tl.constexpr, COLS: tl.constexpr, GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < ROWS
    gate_acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for col_start in range(0, COLS, BLOCK_COLS):
        col = col_start + tl.arange(0, BLOCK_COLS)
        col_mask = col < COLS
        mask = row_mask[:, None] & col_mask[None, :]
        packed = row[:, None] * (COLS // 2) + col[None, :] // 2
        shift = (col[None, :] % 2) * 4
        gate_q = (
            tl.load(gate_packed + packed, mask=mask, other=0).to(tl.int32)
            >> shift
        ) & 15
        up_q = (
            tl.load(up_packed + packed, mask=mask, other=0).to(tl.int32)
            >> shift
        ) & 15
        gate_r = tl.where(gate_q >= 8, gate_q - 16, gate_q).to(tl.float32)
        up_r = tl.where(up_q >= 8, up_q - 16, up_q).to(tl.float32)
        group = col[None, :] // 32
        gate_scale = tl.load(
            gate_alpha + row[:, None] * GROUPS + group,
            mask=mask,
            other=0.0,
        )
        up_scale = tl.load(
            up_alpha + row[:, None] * GROUPS + group,
            mask=mask,
            other=0.0,
        )
        activation = tl.load(x + col, mask=col_mask, other=0.0)
        gate_acc += tl.sum(gate_r * gate_scale * activation[None, :], axis=1)
        up_acc += tl.sum(up_r * up_scale * activation[None, :], axis=1)
    tl.store(gate_output + row, gate_acc, mask=row_mask)
    tl.store(up_output + row, up_acc, mask=row_mask)


def run(args: argparse.Namespace) -> None:
    with ResidentArtifact.open(args.artifact, verify_hashes=False) as artifact:
        runner = TiledResidentGateUp(
            artifact,
            tile_rows=int(artifact.projections["gate"]["rows"]),
            persistent=True,
            base_on_gpu=True,
        )
        try:
            x = torch.randn(runner.cols, device="cuda", dtype=torch.float32)
            sums = torch.empty(runner.cols // 32, device="cuda", dtype=torch.float32)
            torch.sum(x.view(-1, 32), dim=1, out=sums)
            package = runner.cache.package(0)
            assert package is not None
            residual_gate = torch.empty_like(runner.output["gate"])
            residual_up = torch.empty_like(runner.output["up"])
            gate_base_vec = torch.empty_like(runner.output["gate"])
            up_base_vec = torch.empty_like(runner.output["up"])
            configs = [
                (1, 256, 2),
                (1, 256, 4),
                (1, 256, 8),
                (1, 512, 2),
                (1, 512, 4),
                (1, 512, 8),
                (1, 1024, 2),
                (1, 1024, 4),
                (2, 256, 2),
                (2, 256, 4),
                (2, 512, 2),
                (2, 512, 4),
                (4, 256, 2),
                (4, 256, 4),
                (4, 512, 2),
                (4, 512, 4),
                (8, 256, 2),
                (8, 256, 4),
            ]
            for block_rows, block_cols, warps in configs:
                grid = (triton.cdiv(runner.rows, block_rows),)
                for _ in range(3):
                    fused_k_tiled[grid](
                        package["gate.residual"], package["gate.alpha"],
                        package["up.residual"], package["up.alpha"],
                        runner.base_resident["gate"], runner.base_resident["up"],
                        sums, x, runner.output["gate"], runner.output["up"],
                        runner.output["swiglu"],
                        ROWS=runner.rows, COLS=runner.cols, GROUPS=runner.cols // 32,
                        BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
                        BLOCK_GROUPS=triton.next_power_of_2(runner.cols // 32),
                        num_warps=warps, num_stages=2,
                        enable_fp_fusion=True,
                    )
                torch.cuda.synchronize()
                values = []
                for _ in range(args.repeats):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    fused_k_tiled[grid](
                        package["gate.residual"], package["gate.alpha"],
                        package["up.residual"], package["up.alpha"],
                        runner.base_resident["gate"], runner.base_resident["up"],
                        sums, x, runner.output["gate"], runner.output["up"],
                        runner.output["swiglu"],
                        ROWS=runner.rows, COLS=runner.cols, GROUPS=runner.cols // 32,
                        BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
                        BLOCK_GROUPS=triton.next_power_of_2(runner.cols // 32),
                        num_warps=warps, num_stages=2,
                        enable_fp_fusion=True,
                    )
                    end.record()
                    end.synchronize()
                    values.append(begin.elapsed_time(end))
                print(
                    {
                        "block_rows": block_rows,
                        "block_cols": block_cols,
                        "warps": warps,
                        "median_ms": float(np.median(values)),
                        "p95_ms": float(np.percentile(values, 95)),
                    }
                )
            print("residual-only + torch merge")
            for block_rows, block_cols, warps in (
                (1, 512, 2),
                (2, 256, 2),
                (2, 512, 2),
                (4, 256, 4),
            ):
                grid = (triton.cdiv(runner.rows, block_rows),)
                def launch_residual() -> None:
                    residual_k_tiled[grid](
                        package["gate.residual"], package["gate.alpha"],
                        package["up.residual"], package["up.alpha"], x,
                        residual_gate, residual_up,
                        ROWS=runner.rows, COLS=runner.cols,
                        GROUPS=runner.cols // 32,
                        BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
                        num_warps=warps, num_stages=2,
                        enable_fp_fusion=True,
                    )
                def launch_merge() -> None:
                    torch.mv(
                        runner.base_resident["gate"], sums, out=gate_base_vec
                    )
                    torch.mv(
                        runner.base_resident["up"], sums, out=up_base_vec
                    )
                    launch_merge_swiglu(
                        residual_gate,
                        residual_up,
                        gate_base_vec,
                        up_base_vec,
                        runner.output["gate"],
                        runner.output["up"],
                        runner.output["swiglu"],
                        rows=runner.rows,
                    )
                for _ in range(4):
                    launch_residual()
                    launch_merge()
                torch.cuda.synchronize()
                values = []
                for _ in range(args.repeats):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    launch_residual()
                    launch_merge()
                    end.record()
                    end.synchronize()
                    values.append(begin.elapsed_time(end))
                print(
                    {
                        "residual_block_rows": block_rows,
                        "residual_block_cols": block_cols,
                        "residual_warps": warps,
                        "residual_plus_merge_median_ms": float(np.median(values)),
                        "residual_plus_merge_p95_ms": float(np.percentile(values, 95)),
                    }
                )
        finally:
            runner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
