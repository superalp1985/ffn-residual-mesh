from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


QK_K = 256
SUBGROUPS = 8
Q4_K_BLOCK_BYTES = 144


def mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def projection_geometry(q4_result: dict, projection: str) -> tuple[int, int, int]:
    shape = q4_result["projections"][projection]["shape"]
    rows, blocks, values = (int(v) for v in shape)
    if values != QK_K:
        raise ValueError(f"unexpected Q4_K block width: {values}")
    return rows, blocks, values


def package_bytes(rows: int, blocks: int, residual_bits: int) -> int:
    values = rows * blocks * QK_K
    residual_code = math.ceil(values * residual_bits / 8)
    alpha_fp16 = rows * blocks * SUBGROUPS * 2
    return residual_code + alpha_fp16


def tile_package_bytes(tile_rows: int, blocks: int, residual_bits: int, projections: int = 2) -> int:
    return projections * package_bytes(tile_rows, blocks, residual_bits)


def row_tiles(rows: int, tile_rows: int) -> int:
    return math.ceil(rows / tile_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Placement ledger for host-base/GPU-residual FFN with output-row tiles"
    )
    parser.add_argument("q4k_result", type=Path)
    parser.add_argument("--residual-bits", default="1,2,3")
    parser.add_argument("--tile-rows", default="64,256,1024,6144")
    parser.add_argument("--token-windows", default="1,4,16,64,256")
    parser.add_argument("--staging-buffers", type=int, default=2)
    parser.add_argument("--activation-bytes", type=int, default=2)
    parser.add_argument("--output-bytes", type=int, default=2)
    parser.add_argument("--pcie-gbps", type=float, default=24.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    q4k = json.loads(args.q4k_result.read_text(encoding="utf-8"))
    rows, blocks, _ = projection_geometry(q4k, "gate")
    up_rows, up_blocks, _ = projection_geometry(q4k, "up")
    if (rows, blocks) != (up_rows, up_blocks):
        raise ValueError("gate/up geometry differs; pair ledger is undefined")

    hidden = blocks * QK_K
    ffn = rows
    full_gate_up_q4 = sum(
        int(q4k["projections"][name]["q4k_bytes"]) for name in ("gate", "up")
    )
    input_exchange = hidden * args.activation_bytes
    base_gate_up_output = 2 * ffn * args.output_bytes
    # The GPU residual dot performs one low-code term per input value for gate
    # and up. This is deliberately a code-term count, not a claim about a
    # particular CUDA instruction sequence.
    residual_code_terms = 2 * rows * hidden

    rows_out: list[dict] = []
    for bits in [int(v) for v in args.residual_bits.split(",") if v.strip()]:
        full_residual_package = tile_package_bytes(rows, blocks, bits)
        for tile_rows in [int(v) for v in args.tile_rows.split(",") if v.strip()]:
            if tile_rows <= 0 or tile_rows > rows:
                raise ValueError(f"tile rows must be in [1, {rows}]")
            tile_count = row_tiles(rows, tile_rows)
            peak_tile = tile_package_bytes(min(tile_rows, rows), blocks, bits)
            peak_resident = peak_tile * max(args.staging_buffers, 1)
            for window in [int(v) for v in args.token_windows.split(",") if v.strip()]:
                if window <= 0:
                    raise ValueError("token windows must be positive")
                # A cached layer package is loaded once per token window. If
                # the runtime evicts tiles, this is the conservative upper
                # bound; a resident layer can reduce it to zero after warmup.
                weight_h2d_amortized = math.ceil(full_residual_package / window)
                merge_gpu_h2d = input_exchange + base_gate_up_output
                merge_cpu_exchange = input_exchange + base_gate_up_output
                total_merge_gpu = weight_h2d_amortized + merge_gpu_h2d
                total_merge_cpu = weight_h2d_amortized + merge_cpu_exchange
                # PCIe time is a modelled lower bound, not a device benchmark.
                pcie_bytes_per_s = args.pcie_gbps * 1_000_000_000.0
                rows_out.append(
                    {
                        "residual_bits_gpu": bits,
                        "tile_rows": tile_rows,
                        "tile_count_per_projection": tile_count,
                        "token_window_for_weight_reuse": window,
                        "gpu_residual_package_bytes_gate_up": full_residual_package,
                        "gpu_residual_package_mib_gate_up": mib(full_residual_package),
                        "gpu_peak_resident_bytes_with_staging": peak_resident,
                        "gpu_peak_resident_mib_with_staging": mib(peak_resident),
                        "weight_h2d_bytes_amortized_per_token": weight_h2d_amortized,
                        "weight_h2d_mib_amortized_per_token": mib(weight_h2d_amortized),
                        "input_cpu_gpu_exchange_bytes_per_token": input_exchange,
                        "base_gate_up_output_exchange_bytes_per_token": base_gate_up_output,
                        "merge_gpu_total_pcie_bytes_per_token": total_merge_gpu,
                        "merge_cpu_total_pcie_bytes_per_token": total_merge_cpu,
                        "merge_gpu_modelled_pcie_us": total_merge_gpu / pcie_bytes_per_s * 1e6,
                        "merge_cpu_modelled_pcie_us": total_merge_cpu / pcie_bytes_per_s * 1e6,
                        "gpu_residual_code_terms_per_token": residual_code_terms,
                        "gpu_code_terms_per_amortized_h2d_byte": residual_code_terms
                        / max(weight_h2d_amortized, 1),
                        "full_gate_up_q4_weight_bytes": full_gate_up_q4,
                        "weight_h2d_reduction_vs_full_q4_if_streamed": 1.0
                        - weight_h2d_amortized / full_gate_up_q4,
                    }
                )

    result = {
        "experiment": "host_base_gpu_residual_tiled_placement_ledger",
        "scope": "analytical placement ledger; no kernel or PCIe benchmark",
        "source_q4k_result": str(args.q4k_result),
        "dimensions": {"hidden": hidden, "ffn": ffn, "q4k_blocks_per_row": blocks},
        "assumptions": {
            "placement": "host RAM/CPU computes base; GPU receives residual tiles",
            "merge_gpu": "CPU sends x to host for base, then sends gate/up base vectors to GPU; residual output stays on GPU",
            "merge_cpu": "exchange column is retained for comparison; actual SiLU/down placement is runtime-dependent",
            "token_window": "residual weight pages can be reused for this many tokens before eviction",
            "staging_buffers": args.staging_buffers,
            "pcie_gbps_for_model_only": args.pcie_gbps,
            "gpu_residual_code_terms": "one logical low-code contribution per input value; instruction count depends on kernel packing",
        },
        "rows": rows_out,
        "interpretation": {
            "primary_metric": "The useful number is merge_gpu_total_pcie_bytes_per_token after weight reuse, not the static residual package size.",
            "residency": "Tile size controls peak VRAM; token window controls amortized weight traffic.",
            "bottleneck_warning": "This ledger does not prove CPU base throughput. A practical implementation still needs a base evaluator that avoids a full high-code scan or demonstrates that CPU scan is hidden by GPU residual work.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows_out), "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
