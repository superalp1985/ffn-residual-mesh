from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader


MODEL_ID = "Qwen/Qwen3.8-27B"
EXPECTED = {"hidden": 5120, "ffn": 17408, "layers": 64}
FFN_PROJECTIONS = ("ffn_down", "ffn_gate", "ffn_up")


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_dimensions(config_path: Path | None, reader: GGUFReader) -> dict[str, int]:
    architecture = reader.fields["general.architecture"].contents()
    nextn = reader.fields.get(f"{architecture}.nextn_predict_layers")
    mtp = int(nextn.contents()) if nextn else 0
    dimensions = {
        "hidden": int(reader.fields[f"{architecture}.embedding_length"].contents()),
        "ffn": int(reader.fields[f"{architecture}.feed_forward_length"].contents()),
        "layers": int(reader.fields[f"{architecture}.block_count"].contents()) - mtp,
    }
    if config_path and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = config.get("text_config", config)
        external = {
            "hidden": int(text_config["hidden_size"]),
            "ffn": int(text_config["intermediate_size"]),
            "layers": int(text_config["num_hidden_layers"]),
        }
        if external != dimensions:
            raise ValueError(f"config dimensions {external} disagree with GGUF {dimensions}")
    return dimensions


def build_manifest(
    model: Path, config_path: Path | None = None, *,
    expected_dimensions: dict[str, int] | None = None,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    if not model.is_file():
        raise FileNotFoundError(f"missing GGUF: {model}")

    reader = GGUFReader(str(model))
    try:
        return _build_manifest(model, reader, config_path, expected_dimensions, expected_sha256)
    finally:
        reader.data._mmap.close()


def _build_manifest(model, reader, config_path, expected_dimensions, expected_sha256):
    dimensions = _read_dimensions(config_path, reader)
    expected = EXPECTED if expected_dimensions is None else expected_dimensions
    if dimensions != expected:
        raise ValueError(f"unexpected dimensions: {dimensions}, expected {expected}")
    before = model.stat()

    tensors: list[dict[str, object]] = []
    tensor_types: Counter[str] = Counter()
    tensors_by_name = {item.name: item for item in reader.tensors}
    for layer in range(dimensions["layers"]):
        for projection in FFN_PROJECTIONS:
            name = f"blk.{layer}.{projection}.weight"
            if name not in tensors_by_name:
                raise ValueError(f"missing FFN tensor: {name}")
            tensor = tensors_by_name[name]
            quant_type = GGMLQuantizationType(int(tensor.tensor_type))
            rows, hidden = [int(value) for value in tensor.shape]
            if projection == "ffn_down":
                expected_shape = (dimensions["ffn"], dimensions["hidden"])
            else:
                expected_shape = (dimensions["hidden"], dimensions["ffn"])
            if (rows, hidden) != expected_shape:
                raise ValueError(f"unexpected shape for {name}: {(rows, hidden)}, expected {expected_shape}")
            offset, size = int(tensor.data_offset), int(tensor.n_bytes)
            if offset < int(reader.data_offset) or offset + size > before.st_size:
                raise ValueError(f"tensor outside GGUF payload: {name}")
            tensors.append(
                {
                    "layer": layer,
                    "projection": projection,
                    "name": name,
                    "shape": [rows, hidden],
                    "type": int(tensor.tensor_type),
                    "type_name": quant_type.name,
                    "bytes": int(tensor.n_bytes),
                    "data_offset": int(getattr(tensor, "data_offset", -1)),
                }
            )
            tensor_types[quant_type.name] += 1

    digest = sha256_file(model)
    after = model.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("GGUF changed while hashing")
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError("GGUF SHA256 differs from expected source digest")
    named = {item["name"]: item for item in tensors}
    compilable = [
        layer for layer in range(dimensions["layers"])
        if all(named[f"blk.{layer}.ffn_{p}.weight"]["type_name"] == "Q4_K" for p in ("gate", "up"))
    ]
    architecture = reader.fields["general.architecture"].contents()
    total_blocks = int(reader.fields[f"{architecture}.block_count"].contents())
    return {
        "model_id": MODEL_ID,
        "model_id_basis": "experiment target label; see metadata for file-reported identity",
        "metadata": {
            "architecture": reader.fields["general.architecture"].contents(),
            "name": reader.fields["general.name"].contents(),
            "gguf_total_blocks": total_blocks,
            "mtp_layers": total_blocks - dimensions["layers"],
        },
        "quantization": "UD-Q4_K_M (mixed)" if "UD-Q4_K_M" in model.name else "see_tensor_types",
        "quantization_scope": "filename preset only; each tensor retains its actual type",
        "file": {
            "path": str(model.resolve()),
            "bytes": before.st_size,
            "sha256": digest,
        },
        "integrity": {
            "status": "source_digest_matched" if expected_sha256 else "local_digest_only",
            "expected_source_sha256": expected_sha256,
        },
        "dimensions": dimensions,
        "tensors": tensors,
        "tensor_type_counts": dict(sorted(tensor_types.items())),
        "compilable_gate_up_layers": compilable,
        "extra_ffn_tensors": [item.name for item in reader.tensors
                              if ".ffn_" in item.name and item.name not in named],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen3.8-27B GGUF and write a download manifest")
    parser.add_argument("model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--out", type=Path, default=Path("results/qwen38_27b_download_manifest.json"))
    args = parser.parse_args()

    manifest = build_manifest(args.model, args.config, expected_sha256=args.expected_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "model_id": manifest["model_id"],
        "file_bytes": manifest["file"]["bytes"],
        "sha256": manifest["file"]["sha256"],
        "tensor_count": len(manifest["tensors"]),
        "tensor_type_counts": manifest["tensor_type_counts"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
