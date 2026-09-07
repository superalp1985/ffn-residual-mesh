from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as QuantType, GGUFWriter


def write_fixture(path: Path, *, missing_up: bool = False, mtp_layers: int = 0,
                  quantized_down: bool = False, q4k_down: bool = False,
                  q5k_down: bool = False, iq4xs_down: bool = False,
                  gate_up_types: tuple[str, str] = ("Q4_K", "Q4_K")) -> dict:
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
    for name, type_name in zip(("gate", "up"), gate_up_types):
        if name == "up" and missing_up:
            continue
        quant = QuantType[type_name]
        block_bytes = {"Q4_K": 144, "Q5_K": 176}[type_name]
        raw = rng.integers(0, 256, (256, block_bytes), dtype=np.uint8)
        raw[:, :2] = np.array([0.002], dtype="<f2").view(np.uint8)
        raw[:, 2:4] = np.array([0.001], dtype="<f2").view(np.uint8)
        writer.add_tensor(f"blk.0.ffn_{name}.weight", raw, raw_dtype=quant)
        original[name] = raw
    if sum((quantized_down, q4k_down, q5k_down, iq4xs_down)) > 1:
        raise ValueError("select at most one quantized down fixture")
    if q4k_down:
        down = rng.integers(0, 256, (256, 144), dtype=np.uint8)
        down[:, :2] = np.array([0.001], dtype="<f2").view(np.uint8)
        down[:, 2:4] = np.array([0.0005], dtype="<f2").view(np.uint8)
        writer.add_tensor(
            "blk.0.ffn_down.weight", down, raw_dtype=QuantType.Q4_K
        )
    elif q5k_down:
        down = rng.integers(0, 256, (256, 176), dtype=np.uint8)
        down[:, :2] = np.array([0.001], dtype="<f2").view(np.uint8)
        down[:, 2:4] = np.array([0.0005], dtype="<f2").view(np.uint8)
        writer.add_tensor(
            "blk.0.ffn_down.weight", down, raw_dtype=QuantType.Q5_K
        )
    elif iq4xs_down:
        down = rng.integers(0, 256, (256, 136), dtype=np.uint8)
        down[:, :2] = np.array([0.001], dtype="<f2").view(np.uint8)
        writer.add_tensor(
            "blk.0.ffn_down.weight", down, raw_dtype=QuantType.IQ4_XS
        )
    elif quantized_down:
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
