from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as QuantType, GGUFWriter


def write_fixture(path: Path, *, missing_up: bool = False, mtp_layers: int = 0,
                  quantized_down: bool = False) -> dict:
    """Small real GGUF; its random Q4_K bytes exercise all nibble/scale bits."""
    rng = np.random.default_rng(513)
    writer = GGUFWriter(path, "qwen35")
    writer.add_name("fixture")
    writer.add_block_count(1 + mtp_layers)
    if mtp_layers:
        writer.add_uint32("qwen35.nextn_predict_layers", mtp_layers)
    writer.add_embedding_length(256)
    writer.add_feed_forward_length(256)
    original = {}
    for name in ("gate", "up"):
        if name == "up" and missing_up:
            continue
        raw = rng.integers(0, 256, (256, 144), dtype=np.uint8)
        raw[:, :2] = np.array([0.002], dtype="<f2").view(np.uint8)
        raw[:, 2:4] = np.array([0.001], dtype="<f2").view(np.uint8)
        writer.add_tensor(f"blk.0.ffn_{name}.weight", raw, raw_dtype=QuantType.Q4_K)
        original[name] = raw
    if quantized_down:
        down = rng.integers(0, 256, (256, 8, 18), dtype=np.uint8)
        down[:, :, :2] = np.array([0.001], dtype="<f2").view(np.uint8)
        down = down.reshape(256, -1)
        writer.add_tensor("blk.0.ffn_down.weight", down, raw_dtype=QuantType.IQ4_NL)
    else:
        down = rng.standard_normal((256, 256)).astype("<f4") / 16
        writer.add_tensor("blk.0.ffn_down.weight", down)
    original["down"] = down
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return original
