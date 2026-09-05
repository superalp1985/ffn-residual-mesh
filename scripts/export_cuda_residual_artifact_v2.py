from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_exact_radix_split_pipeline import load_q4_projection, quantize_groupwise_q8


def pack_row_codes(codes: np.ndarray, bits: int = 2) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint8)
    if values.ndim != 2 or bits != 2 or values.shape[1] % 4:
        raise ValueError("expected a 2-D row matrix and exactly 2 bits")
    if np.any(values > 3):
        raise ValueError("2-bit residual codes must be in [0, 3]")
    return (
        values[:, 0::4]
        | (values[:, 1::4] << 2)
        | (values[:, 2::4] << 4)
        | (values[:, 3::4] << 6)
    ).astype(np.uint8, copy=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def artifact_manifest(
    *,
    projection: str,
    layer: int,
    rows: int,
    hidden: int,
    residual_bits: int,
    code_bytes: int,
    alpha_bytes: int,
) -> dict[str, object]:
    return {
        "projection": projection,
        "layer": layer,
        "rows": rows,
        "hidden": hidden,
        "residual_bits": residual_bits,
        "lossless": True,
        "runtime_weight_reads": 0,
        "runtime_contract": "packed exact residual and alpha only; raw GGUF is cold-start-only",
        "packed_code_bytes": code_bytes,
        "alpha_bytes": alpha_bytes,
        "payload_bytes": code_bytes + alpha_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export exact row-packed Q4_K low-code residuals for CUDA")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    projections: dict[str, dict[str, object]] = {}
    for projection in ("gate", "up"):
        codes, alpha, _, source_bytes = load_q4_projection(args.model, args.layer, projection)
        residual = pack_row_codes((codes & 3).reshape(codes.shape[0], -1))
        code_path = args.out / f"{projection}.qlo2.rowpacked.bin"
        alpha_path = args.out / f"{projection}.alpha.f32.bin"
        code_bytes = np.ascontiguousarray(residual).tobytes()
        alpha_bytes = np.ascontiguousarray(alpha.astype("<f4", copy=False)).tobytes()
        code_path.write_bytes(code_bytes)
        alpha_path.write_bytes(alpha_bytes)
        manifest = artifact_manifest(
            projection=projection,
            layer=args.layer,
            rows=codes.shape[0],
            hidden=codes.shape[1] * codes.shape[2],
            residual_bits=2,
            code_bytes=len(code_bytes),
            alpha_bytes=len(alpha_bytes),
        )
        manifest.update(
            {
                "source_q4_bytes": int(source_bytes),
                "packed_code_file": code_path.name,
                "packed_code_sha256": sha256_bytes(code_bytes),
                "alpha_file": alpha_path.name,
                "alpha_sha256": sha256_bytes(alpha_bytes),
                "alpha_shape": list(alpha.shape),
                "packed_row_bytes": residual.shape[1],
            }
        )
        projections[projection] = manifest

    input_values = np.fromfile(args.input, dtype="<f4")
    hidden = int(next(iter(projections.values()))["hidden"])
    if input_values.size != hidden:
        raise ValueError(f"expected {hidden} float32 activation values, found {input_values.size}")
    activation_codes, activation_scales = quantize_groupwise_q8(input_values.reshape(1, hidden), group_size=32)
    z_path = args.out / "activation.z.i8.bin"
    scale_path = args.out / "activation.scale.f32.bin"
    z_bytes = np.ascontiguousarray(activation_codes[0]).tobytes()
    scale_bytes = np.ascontiguousarray(activation_scales[0].astype("<f4", copy=False)).tobytes()
    z_path.write_bytes(z_bytes)
    scale_path.write_bytes(scale_bytes)

    result = {
        "format": "FFN_EXACT_QLO2_ROWPACKED_V2",
        "layer": args.layer,
        "lossless": True,
        "runtime_weight_reads": 0,
        "activation": {
            "source": str(args.input),
            "z_file": z_path.name,
            "z_bytes": len(z_bytes),
            "scale_file": scale_path.name,
            "scale_bytes": len(scale_bytes),
            "group_size": 32,
        },
        "projections": projections,
    }
    (args.out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
