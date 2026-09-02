from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gguf import GGUFReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reader = GGUFReader(str(args.model))
    fields = {}
    for key, field in reader.fields.items():
        value = field.parts[-1]
        if hasattr(value, "tolist"):
            value = value.tolist()
        value = _json_safe(value)
        if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
            if any(token in key for token in ("architecture", "name", "license", "repo_url", "tokenizer")):
                try:
                    value = bytes(value).decode("utf-8")
                except UnicodeDecodeError:
                    pass
        fields[key] = value

    tensors = []
    for tensor in reader.tensors:
        tensors.append(
            {
                "name": tensor.name,
                "shape": [int(x) for x in tensor.shape],
                "dtype": str(tensor.tensor_type),
                "n_bytes": int(tensor.n_bytes),
            }
        )

    payload = {
        "model": str(args.model),
        "fields": fields,
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tensor_count": len(tensors), "field_count": len(fields), "out": str(args.out)}))


def _json_safe(value):
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            try:
                return bytes(value.tolist()).decode("utf-8")
            except UnicodeDecodeError:
                return value.tolist()
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
