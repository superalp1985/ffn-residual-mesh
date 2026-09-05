from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from scan_q4k_hierarchical_code_split import load_q4k_codes


MAGIC = "FFN_HOST_BASE_GPU_RESIDUAL_V1"


def pack_bitplanes(values: np.ndarray, bits: int) -> bytes:
    """Pack an unsigned stream as independent little-endian bit planes."""
    if bits < 1 or bits > 4:
        raise ValueError("bits must be in [1, 4]")
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    planes = []
    for bit in range(bits):
        plane = ((flat >> bit) & 1).astype(np.uint8, copy=False)
        planes.append(np.packbits(plane, bitorder="little"))
    return b"".join(np.ascontiguousarray(plane).tobytes() for plane in planes)


def unpack_bitplanes(raw: bytes, count: int, bits: int) -> np.ndarray:
    plane_bytes = (count + 7) // 8
    if len(raw) != plane_bytes * bits:
        raise ValueError("bitplane payload length does not match manifest")
    out = np.zeros(count, dtype=np.uint8)
    for bit in range(bits):
        plane = np.frombuffer(raw, dtype=np.uint8, count=plane_bytes, offset=bit * plane_bytes)
        unpacked = np.unpackbits(plane, bitorder="little")[:count]
        out |= unpacked.astype(np.uint8) << bit
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_array(path: Path, array: np.ndarray, dtype: str) -> dict:
    value = np.ascontiguousarray(array.astype(dtype, copy=False))
    path.write_bytes(value.tobytes())
    return {"file": path.name, "shape": list(value.shape), "dtype": dtype, "bytes": path.stat().st_size}


def compile_projection(model: Path, layer: int, projection: str, residual_bits: int, out_dir: Path) -> dict:
    codes, alpha, beta, _, q4_bytes = load_q4k_codes(model, layer, projection)
    rows, blocks, width = codes.shape
    if width != 256:
        raise ValueError("expected Q4_K width 256")
    low_mask = (1 << residual_bits) - 1
    low = (codes & low_mask).astype(np.uint8, copy=False)
    high = (codes >> residual_bits).astype(np.uint8, copy=False)
    count = int(codes.size)
    high_path = out_dir / f"{projection}.host_high.bitplanes"
    low_path = out_dir / f"{projection}.gpu_residual.bitplanes"
    high_path.write_bytes(pack_bitplanes(high, 4 - residual_bits))
    low_path.write_bytes(pack_bitplanes(low, residual_bits))
    alpha_info = write_array(out_dir / f"{projection}.alpha.f32", alpha, "<f4")
    beta_info = write_array(out_dir / f"{projection}.beta.f32", beta, "<f4")
    # Decode the just-written streams to make the artifact self-consistency
    # check independent of the original in-memory arrays.
    high_check = unpack_bitplanes(high_path.read_bytes(), count, 4 - residual_bits).reshape(codes.shape)
    low_check = unpack_bitplanes(low_path.read_bytes(), count, residual_bits).reshape(codes.shape)
    if not np.array_equal((high_check << residual_bits) | low_check, codes):
        raise RuntimeError(f"artifact code reconstruction failed for {projection}")
    return {
        "projection": projection,
        "shape": [rows, blocks, width],
        "q4k_source_bytes": int(q4_bytes),
        "host_high_bits": 4 - residual_bits,
        "gpu_residual_bits": residual_bits,
        "host_high_bitplanes": {
            "file": high_path.name,
            "bytes": high_path.stat().st_size,
            "sha256": sha256(high_path),
        },
        "gpu_residual_bitplanes": {
            "file": low_path.name,
            "bytes": low_path.stat().st_size,
            "sha256": sha256(low_path),
        },
        "alpha": alpha_info,
        "beta": beta_info,
        "host_artifact_bytes": high_path.stat().st_size + beta_path_bytes(beta_info) + alpha_path_bytes(alpha_info),
        "gpu_residual_package_bytes": low_path.stat().st_size + alpha_path_bytes(alpha_info),
    }


def alpha_path_bytes(info: dict) -> int:
    return int(info["bytes"])


def beta_path_bytes(info: dict) -> int:
    return int(info["bytes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Q4_K into host high-code and GPU residual artifacts")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--residual-bits", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.residual_bits < 1 or args.residual_bits > 3:
        raise ValueError("residual bits must be 1, 2, or 3")
    args.out.mkdir(parents=True, exist_ok=True)
    projections = [compile_projection(args.model, args.layer, p, args.residual_bits, args.out) for p in ("gate", "up")]
    manifest = {
        "magic": MAGIC,
        "version": 1,
        "model": str(args.model),
        "layer": args.layer,
        "residual_bits": args.residual_bits,
        "host_high_bits": 4 - args.residual_bits,
        "compile_phase": "raw Q4_K is scanned once here; runtime must use this manifest and payloads only",
        "runtime_formula": "host_base = alpha * 2^r * dot(x, q_hi) + beta * sum(x); gpu_residual = alpha * dot(x, q_lo); merged = host_base + gpu_residual",
        "platform": platform.platform(),
        "projections": {row["projection"]: row for row in projections},
        "pair_totals": {
            "host_artifact_bytes": sum(row["host_artifact_bytes"] for row in projections),
            "gpu_residual_package_bytes": sum(row["gpu_residual_package_bytes"] for row in projections),
            "q4k_source_bytes": sum(row["q4k_source_bytes"] for row in projections),
        },
        "access_contract": {
            "cold_start_raw_weight_scan": True,
            "runtime_raw_weight_scan": False,
            "runtime_reads": ["manifest.json", "host_high.bitplanes", "gpu_residual.bitplanes", "alpha", "beta"],
            "fallback": "raw GGUF may be opened only by an explicitly counted exact fallback path",
        },
        "candidate_status": {
            "class": "exact_reference_artifact",
            "not_yet_accepted_for_runtime": "A runtime that streams host_high.bitplanes per token still scans a compiled weight stream. The next compiler must emit aggregate formulas or partial-sum tables that avoid that scan.",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.out / 'manifest.json'), "pair_totals": manifest["pair_totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
