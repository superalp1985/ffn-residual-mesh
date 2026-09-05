from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import Q4_K, dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resident_residual_format import FORMAT, file_sha256  # noqa: E402


def decode_q4k(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = raw.shape[0]
    blocks = raw.reshape(rows, -1, 144)
    sc, minimum = Q4_K.get_scale_min(blocks[:, :, 4:16].reshape(-1, 12))
    d = blocks[:, :, :2].copy().view("<f2").astype(np.float32)
    dm = blocks[:, :, 2:4].copy().view("<f2").astype(np.float32)
    alpha = (d * sc.reshape(rows, -1, 8)).reshape(rows, -1)
    beta = (-dm * minimum.reshape(rows, -1, 8)).reshape(rows, -1)
    packed = blocks[:, :, 16:].reshape(rows, -1, 4, 32)
    codes = np.stack((packed & 15, packed >> 4), axis=3).reshape(rows, -1)
    return codes, alpha, beta


def compile_layer(
    model: Path, layer: int, bits: int, out_dir: Path, *, chunk_rows: int = 128,
) -> dict:
    if bits != 4:
        raise ValueError("exact v1 supports only 4-bit residuals; no clipping permitted")
    if layer < 0 or chunk_rows <= 0:
        raise ValueError("invalid layer or chunk_rows")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifact: {out_dir}")
    started = time.perf_counter()
    reader = GGUFReader(model)
    tensors = {item.name: item for item in reader.tensors}
    for name in ("gate", "up", "down"):
        if f"blk.{layer}.ffn_{name}.weight" not in tensors:
            raise ValueError(f"missing layer {layer} projection {name}")
    source_stat = model.stat()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": FORMAT, "layer": layer, "residual_bits": 4, "group_size": 32,
        "runtime_requires_table_lookup": False,
        "exactness": "integer codes exact; original Q4_K decoded FP32 weights bitwise reconstructed",
        "runtime_float_arithmetic": "split dot uses different reduction order; measure output error separately",
        "source": {"path": str(model.resolve()), "bytes": source_stat.st_size,
                   "mtime_ns": source_stat.st_mtime_ns},
        "projections": {}, "fallbacks": {},
    }
    ledger = dict(residual_code_bytes=0, resident_gate_up_bytes=0, host_base_bytes=0,
                  verification_only_bytes=0, artifact_payload_bytes=0,
                  source_gate_up_bytes=0, fallback_bytes=0)
    for name in ("gate", "up", "down"):
        tensor = tensors[f"blk.{layer}.ffn_{name}.weight"]
        source = dict(name=tensor.name, type_name=GGMLQuantizationType(tensor.tensor_type).name,
                      shape=[int(x) for x in tensor.shape], offset=int(tensor.data_offset),
                      bytes=int(tensor.n_bytes))
        if int(tensor.tensor_type) != int(GGMLQuantizationType.Q4_K) or name == "down":
            manifest["fallbacks"][name] = {**source, "reason": "preserve original format; not compiled in v1",
                                          "runtime_cost_included": False}
            ledger["fallback_bytes"] += int(tensor.n_bytes)
            continue
        rows, cols = int(tensor.shape[1]), int(tensor.shape[0])
        specs = {
            "residual": ("|u1", [rows, cols // 2]),
            "base": ("|u1", [rows, cols // 32]),
            "alpha": ("<f4", [rows, cols // 32]),
            "beta": ("<f4", [rows, cols // 32]),
            "coefficient": ("<f8", [rows, cols // 32]),
        }
        files = {}
        mismatch_count = 0
        with ExitStack() as stack:
            streams = {kind: stack.enter_context((out_dir / f"{name}.{kind}.bin").open("xb"))
                       for kind in specs}
            for start in range(0, rows, chunk_rows):
                stop = min(start + chunk_rows, rows)
                raw = tensor.data[start:stop]
                q, alpha, beta = decode_q4k(raw)
                groups = q.reshape(stop - start, -1, 32)
                base = ((groups.min(axis=-1).astype(np.int16)
                         + groups.max(axis=-1).astype(np.int16) + 1) // 2).astype(np.uint8)
                residual = (groups.astype(np.int16) - base[:, :, None]).reshape(q.shape)
                if residual.min() < -8 or residual.max() > 7:
                    raise ValueError("internal error: exact signed nibble overflow")
                nibble = (residual & 15).astype(np.uint8)
                packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
                coefficients = alpha.astype(np.float64) * base + beta.astype(np.float64)
                reconstructed = (alpha[:, :, None] * groups.astype(np.float32)
                                 + beta[:, :, None]).reshape(q.shape)
                oracle = dequantize(raw, tensor.tensor_type)
                mismatch_count += int(np.count_nonzero(
                    reconstructed.view(np.uint32) != oracle.view(np.uint32)))
                if not np.isfinite(coefficients).all() or mismatch_count:
                    raise ValueError("Q4_K decode disagrees with upstream oracle")
                arrays = dict(residual=packed, base=base, alpha=alpha, beta=beta,
                              coefficient=coefficients)
                for kind, value in arrays.items():
                    np.ascontiguousarray(value, dtype=specs[kind][0]).tofile(streams[kind])
        for kind, (dtype, shape) in specs.items():
            path = out_dir / f"{name}.{kind}.bin"
            files[kind] = dict(file=path.name, dtype=dtype, shape=shape,
                               bytes=path.stat().st_size, sha256=file_sha256(path))
        manifest["projections"][name] = {
            "rows": rows, "cols": cols, "source": source, "files": files,
            "verified_weight_bit_mismatches": mismatch_count,
        }
        ledger["residual_code_bytes"] += files["residual"]["bytes"]
        ledger["resident_gate_up_bytes"] += files["residual"]["bytes"] + files["alpha"]["bytes"]
        ledger["host_base_bytes"] += files["coefficient"]["bytes"]
        ledger["verification_only_bytes"] += files["base"]["bytes"] + files["beta"]["bytes"]
        ledger["artifact_payload_bytes"] += sum(entry["bytes"] for entry in files.values())
        ledger["source_gate_up_bytes"] += int(tensor.n_bytes)
    after = model.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source model changed while compiling")
    manifest["byte_ledger"] = ledger
    manifest["scope"] = "gate/up only; fallback runtime and full model are NOT implemented"
    manifest["cold_start_seconds"] = time.perf_counter() - started
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"path": str(path.resolve()), "byte_ledger": ledger,
            "compiled_projections": list(manifest["projections"]),
            "fallbacks": list(manifest["fallbacks"]), "cold_start_seconds": manifest["cold_start_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile exact Q4_K gate/up resident residuals offline")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_layer(args.model, args.layer, args.bits, args.out,
                                  chunk_rows=args.chunk_rows), indent=2))


if __name__ == "__main__":
    main()
