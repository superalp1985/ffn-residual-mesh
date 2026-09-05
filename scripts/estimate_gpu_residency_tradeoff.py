from __future__ import annotations

import argparse
import json
from pathlib import Path


def mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FFN split artifacts into GPU residency and H2D ledgers")
    parser.add_argument("centered_result", type=Path)
    parser.add_argument("q4k_result", type=Path)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument(
        "--placement",
        choices=("residual_gpu_base_host", "base_gpu_residual_host"),
        default="residual_gpu_base_host",
        help="Preferred architecture keeps the bulky base in host RAM and stages only residual tiles to GPU.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    centered = json.loads(args.centered_result.read_text(encoding="utf-8"))
    q4k = json.loads(args.q4k_result.read_text(encoding="utf-8"))
    block = int(centered["block_size"])
    hidden = int(centered["dimensions"]["hidden"])
    ffn = int(centered["dimensions"]["ffn"])
    blocks = int(centered["blocks"])
    # Two projections (gate/up) share activation block means. A base scalar is
    # stored per output row and input block for each projection.
    base_fp16 = 2 * ffn * blocks * 2
    scale_fp16 = 2 * ffn * blocks * 2
    original_gate_up = int(q4k["dimensions"]["gate"]["q4k_bytes"] + q4k["dimensions"]["up"]["q4k_bytes"])
    rows = []
    for row in centered["rows"]:
        bits = int(row["bits"])
        residual_code = int(row["artifact"]["residual_code_bytes"])
        # Preferred placement: the bulky base is expanded in host RAM and
        # CPU-evaluated or precomputed by a layer-specific formula. Only the
        # residual package is staged to GPU. Host bytes are reported as
        # context, never as a success metric.
        if args.placement == "residual_gpu_base_host":
            resident_bytes = residual_code
            h2d_per_token = residual_code
            host_base_bytes = base_fp16 + scale_fp16
        else:
            resident_bytes = base_fp16 + scale_fp16
            h2d_per_token = residual_code
            host_base_bytes = residual_code
        full_weight_h2d_per_token = original_gate_up
        rows.append(
            {
                "bits": bits,
                "gpu_resident_compact_bytes": resident_bytes,
                "gpu_resident_compact_mib": mib(resident_bytes),
                "host_base_or_residual_bytes": host_base_bytes,
                "h2d_residual_bytes_per_token": h2d_per_token,
                "h2d_residual_mib_per_token": mib(h2d_per_token),
                "full_gate_up_q4_bytes_if_streamed_per_token": full_weight_h2d_per_token,
                "h2d_reduction_vs_full_stream": 1.0 - h2d_per_token / full_weight_h2d_per_token,
                "cpu_to_gpu_base_output_bytes_per_token": 2 * ffn * 2,
                "cpu_to_gpu_optional_input_bytes_per_token": hidden * 2,
                "note": "Residual code bytes are a lower bound and still represent a dense residual artifact; base-output exchange is separate.",
            }
        )

    result = {
        "experiment": "gpu_residency_tradeoff",
        "scope": "GPU VRAM residency and H2D ledger; host RAM intentionally excluded from primary metrics",
        "source_centered": str(args.centered_result),
        "source_q4k": str(args.q4k_result),
        "layer": centered["layer"],
        "block_size": block,
        "dimensions": {"hidden": hidden, "ffn": ffn, "blocks": blocks},
        "placement": args.placement,
        "tokens": args.tokens,
        "rows": rows,
        "interpretation": {
            "priority": "GPU resident bytes and dynamic H2D bytes dominate; host expanded base size is intentionally not a success metric.",
            "architecture_constraint": "Preferred placement keeps bulky base data in host RAM/CPU and stages only residual tiles to GPU. This is useful only if base evaluation/merge does not require sending a full base output vector to GPU.",
            "runtime_caveat": "The H2D ledger treats one dense residual artifact as a per-token transfer. A real kernel must stream only the residual tile requested by the current token/batch and overlap copies with compute.",
            "merge_requirement": "The base result must be merged on CPU, or represented by a compact GPU-side accumulator/formula; otherwise moving a full base output vector can erase the weight-transfer saving.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
