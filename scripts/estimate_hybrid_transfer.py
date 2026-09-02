from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate CPU/GPU traffic for routed FFN approximation")
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--full-ffn-bytes", type=int, default=23_000_000)
    parser.add_argument("--input-bytes", type=int, default=8_192)
    parser.add_argument("--code-bytes", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--superblock", type=int, default=256 * 1024)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.experiment.read_text(encoding="utf-8"))
    rows = []
    for row in data["rows"]:
        rank = int(row["rank"])
        coeff_bytes = rank * 4
        for policy in row["policies"]:
            holdout = policy["holdout"]
            approx = float(holdout["approx_fraction"])
            fallback = float(holdout["fallback_fraction"])
            # Batch all approximate coefficients together. A superblock models
            # the minimum DMA granularity for fallback weight pages.
            approx_h2d = approx * (coeff_bytes + args.code_bytes)
            fallback_h2d = fallback * args.full_ffn_bytes
            expected_h2d = approx_h2d + fallback_h2d
            exact_page_h2d = ((fallback * args.full_ffn_bytes + args.superblock - 1) // args.superblock) * args.superblock
            # For batch=1, this is intentionally conservative: a real ring
            # buffer can amortize the page rounding across consecutive tokens.
            if args.batch > 1:
                exact_page_h2d = exact_page_h2d / args.batch
            rows.append({
                "rank": rank,
                "train_p95_budget": policy["train_p95_budget"],
                "approx_fraction": approx,
                "fallback_fraction": fallback,
                "cpu_d2h_input_bytes_per_token": args.input_bytes,
                "h2d_coeff_and_code_bytes_per_token": approx_h2d,
                "h2d_exact_weight_lower_bound_bytes_per_token": fallback_h2d,
                "h2d_exact_weight_superblock_bytes_per_token": exact_page_h2d,
                "h2d_hybrid_lower_bound_bytes_per_token": expected_h2d,
                "h2d_hybrid_superblock_estimate_bytes_per_token": approx_h2d + exact_page_h2d,
                "reduction_vs_full_weight_lower_bound": 1.0 - expected_h2d / args.full_ffn_bytes,
            })
    result = {
        "experiment": str(args.experiment),
        "full_ffn_bytes": args.full_ffn_bytes,
        "input_bytes": args.input_bytes,
        "superblock_bytes": args.superblock,
        "batch": args.batch,
        "rows": rows,
        "caveat": "Fallback bytes are a traffic proxy, not a llama.cpp implementation measurement; page rounding is pessimistic for batch=1 and improves with a persistent pinned ring buffer.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
