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
ALLOWED_TYPES = frozenset({"Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ4_NL", "IQ3_S", "Q3_K"})


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return _decode(value.tolist())
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _read_dimensions(config_path: Path | None, reader: GGUFReader) -> dict[str, int]:
    if config_path and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = config.get("text_config", config)
        return {
            "hidden": int(text_config["hidden_size"]),
            "ffn": int(text_config["intermediate_size"]),
            "layers": int(text_config["num_hidden_layers"]),
        }

    architecture = _decode(reader.fields["general.architecture"].parts[-1])
    prefix = f"{architecture}."
    return {
        "hidden": int(reader.fields[f"{prefix}embedding_length"].parts[-1]),
        "ffn": int(reader.fields[f"{prefix}feed_forward_length"].parts[-1]),
        "layers": int(reader.fields[f"{prefix}block_count"].parts[-1]),
    }


def build_manifest(model: Path, config_path: Path | None = None) -> dict[str, object]:
    if not model.is_file():
        raise FileNotFoundError(f"missing GGUF: {model}")

    reader = GGUFReader(str(model))
    dimensions = _read_dimensions(config_path, reader)
    if dimensions != EXPECTED:
        raise ValueError(f"unexpected dimensions: {dimensions}, expected {EXPECTED}")

    tensors: list[dict[str, object]] = []
    tensor_types: Counter[str] = Counter()
    tensors_by_name = {item.name: item for item in reader.tensors}
    for layer in range(dimensions["layers"]):
        for projection in FFN_PROJECTIONS:
            name = f"blk.{layer}.{projection}.weight"
            tensor = tensors_by_name[name]
            quant_type = GGMLQuantizationType(int(tensor.tensor_type))
            if quant_type.name not in ALLOWED_TYPES:
                raise ValueError(
                    f"expected a Q4_K_M-compatible 4/5-bit tensor for {name}, got {quant_type.name}"
                )
            rows, hidden = [int(value) for value in tensor.shape]
            if projection == "ffn_down":
                expected_shape = (dimensions["ffn"], dimensions["hidden"])
            else:
                expected_shape = (dimensions["hidden"], dimensions["ffn"])
            if (rows, hidden) != expected_shape:
                raise ValueError(f"unexpected shape for {name}: {(rows, hidden)}, expected {expected_shape}")
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

    return {
        "model_id": MODEL_ID,
        "source_model_card": "https://huggingface.co/Qwen/Qwen3.8-27B",
        "quantization": "Q4_K",
        "quantization_scope": "Unsloth UD-Q4_K_M mixed Q4-family tensor manifest; do not assume every tensor is Q4_K",
        "file": {
            "path": str(model.resolve()),
            "bytes": model.stat().st_size,
            "sha256": sha256_file(model),
        },
        "dimensions": dimensions,
        "tensors": tensors,
        "tensor_type_counts": dict(sorted(tensor_types.items())),
        "runtime_requires_table_lookup": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen3.8-27B GGUF and write a download manifest")
    parser.add_argument("model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/qwen38_27b_download_manifest.json"))
    args = parser.parse_args()

    manifest = build_manifest(args.model, args.config)
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
