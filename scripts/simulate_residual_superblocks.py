from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rounded(bytes_count: int, block: int) -> int:
    return int(math.ceil(bytes_count / block) * block) if bytes_count else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Superblock transfer simulation for FFN base/residual packets")
    parser.add_argument("--approx-bytes", type=int, default=4112)
    parser.add_argument("--fallback-weight-bytes", type=int, default=10321920)
    parser.add_argument("--windows", default="1,4,16,64,256,1024")
    parser.add_argument("--superblocks", default="65536,262144,1048576")
    parser.add_argument("--fallback-fractions", default="0,0.02,0.05,0.10")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for tokens in [int(v) for v in args.windows.split(",") if v.strip()]:
        for superblock in [int(v) for v in args.superblocks.split(",") if v.strip()]:
            approx_exact = rounded(tokens * args.approx_bytes, superblock)
            for fallback_fraction in [float(v) for v in args.fallback_fractions.split(",") if v.strip()]:
                fallback_count = int(round(tokens * fallback_fraction))
                approx_count = max(0, tokens - fallback_count)
                # Approximate packets are contiguous in a pinned ring buffer.
                approx_transfer = rounded(approx_count * args.approx_bytes, superblock)
                # A fallback weight page can be reused by all fallback tokens in this window.
                fallback_transfer_once = rounded(args.fallback_weight_bytes, superblock) if fallback_count else 0
                # Worst case when the fallback weight is evicted between tokens.
                fallback_transfer_per_token = fallback_count * rounded(args.fallback_weight_bytes, superblock)
                coalesced = approx_transfer + fallback_transfer_once
                rows.append({
                    "tokens": tokens,
                    "superblock_bytes": superblock,
                    "fallback_fraction": fallback_fraction,
                    "fallback_tokens": fallback_count,
                    "approx_tokens": approx_count,
                    "approx_transfer_bytes": approx_transfer,
                    "fallback_transfer_once_bytes": fallback_transfer_once,
                    "fallback_transfer_evicted_bytes": fallback_transfer_per_token,
                    "coalesced_total_bytes": coalesced,
                    "coalesced_bytes_per_token": coalesced / max(tokens, 1),
                    "evicted_total_bytes": approx_transfer + fallback_transfer_per_token,
                    "evicted_bytes_per_token": (approx_transfer + fallback_transfer_per_token) / max(tokens, 1),
                    "approx_only_bytes_per_token": approx_exact / max(tokens, 1),
                    "approx_waste_ratio": (approx_transfer - approx_count * args.approx_bytes) / max(approx_count * args.approx_bytes, 1),
                })
    result = {
        "experiment": "residual_packet_superblock_simulation",
        "assumptions": {
            "approx_packet": "base output + residual values + route metadata",
            "approx_bytes": args.approx_bytes,
            "fallback_weight_bytes": args.fallback_weight_bytes,
            "coalescing": "approximate packets are contiguous in a pinned host ring buffer",
            "fallback_once": "fallback weight pages are reusable within a token window",
            "fallback_evicted": "conservative upper bound if fallback pages are evicted every token",
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "approx_bytes": args.approx_bytes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
