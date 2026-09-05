from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
from gguf import GGUFReader

from scan_q4k_hierarchical_code_split import Q4_K_BLOCK_BYTES, QK_K, load_q4k_codes


def entropy_binary(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def summarize(codes: np.ndarray) -> dict:
    # [rows, 8 Q4_K blocks, 256] -> [all 32-value subgroups, 32].
    groups = codes.reshape(-1, 32).astype(np.uint8)
    rows = []
    for bit in range(4):
        plane = ((groups >> bit) & 1).astype(np.uint8)
        p = float(plane.mean())
        constant_fraction = float(np.mean(plane.sum(axis=1) == 0) + np.mean(plane.sum(axis=1) == 32))
        # Adjacent transitions approximate run-length friendliness. A low
        # transition rate means a plane may be sent as a few runs/templates.
        transitions = float(np.mean(plane[:, 1:] != plane[:, :-1]))
        rows.append(
            {
                "bit": bit,
                "one_fraction": p,
                "binary_entropy_bits": entropy_binary(p),
                "constant_group_fraction": constant_fraction,
                "adjacent_transition_fraction": transitions,
                "raw_bytes_per_group": 4.0,
                "entropy_lower_bound_bytes_per_group": 32.0 * entropy_binary(p) / 8.0,
            }
        )
    # Exact duplicate 32-code vectors are a lower-bound test for dictionary
    # coding. The count is expected to be low for dense trained weights.
    _, counts = np.unique(groups, axis=0, return_counts=True)
    duplicate_values = int(np.sum(counts[counts > 1] - 1))
    return {
        "groups": int(len(groups)),
        "unique_32_code_vectors": int(len(counts)),
        "duplicate_vector_fraction": duplicate_values / len(groups),
        "planes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Q4_K bit-plane entropy and template reuse")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    projections = {}
    for projection in ("gate", "up"):
        codes, _, _, _, q4_bytes = load_q4k_codes(args.model, args.layer, projection)
        projections[projection] = {
            "shape": list(codes.shape),
            "q4k_bytes": q4_bytes,
            "structure": summarize(codes),
        }
    result = {
        "experiment": "q4k_bitplane_structure",
        "scope": "static code structure only; no activation or transfer benchmark",
        "model": str(args.model),
        "layer": args.layer,
        "platform": platform.platform(),
        "projections": projections,
        "interpretation": {
            "entropy": "A plane near one bit/group has no information-theoretic byte reduction without an additional structure such as runs or a dictionary.",
            "dictionary": "Duplicate-vector fraction is a conservative template-reuse probe; near zero means exact template lookup cannot carry the weight.",
            "traffic": "Even a compressed static artifact helps runtime only if base/template data is resident and dynamic residual traffic is measured separately.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
