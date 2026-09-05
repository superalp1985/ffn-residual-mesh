from __future__ import annotations

import argparse
import json
from pathlib import Path


def plan_residency(manifest, config, *, vram_bytes, context_tokens,
                   fixed_reserve_bytes, batch=1, kv_element_bytes=2):
    text = config.get("text_config", config)
    if min(vram_bytes, context_tokens, fixed_reserve_bytes) < 0 or min(batch, kv_element_bytes) < 1:
        raise ValueError("invalid memory budget")
    layers = int(text["num_hidden_layers"])
    types = text["layer_types"]
    if len(types) != layers or any(t not in ("linear_attention", "full_attention") for t in types):
        raise ValueError("explicit per-layer attention types required")
    full = types.count("full_attention")
    linear = types.count("linear_attention")
    kv = batch * context_tokens * full * text["num_key_value_heads"] * text["head_dim"] * 2 * kv_element_bytes
    recurrent = convolution = 0
    if linear:
        hv, hk = text["linear_num_value_heads"], text["linear_num_key_heads"]
        dk, dv = text["linear_key_head_dim"], text["linear_value_head_dim"]
        recurrent = batch * linear * hv * dk * dv * 4
        convolution = batch * linear * (2 * hk * dk + hv * dv) * (text["linear_conv_kernel_dim"] - 1) * 4
    reserve = kv + recurrent + convolution + fixed_reserve_bytes
    per_layer = [0] * layers
    raw = [0] * layers
    compiled = set(manifest["compilable_gate_up_layers"])
    for tensor in manifest["tensors"]:
        layer = tensor["layer"]
        raw[layer] += tensor["bytes"]
        size = tensor["bytes"]
        if layer in compiled and tensor["projection"] in ("ffn_gate", "ffn_up"):
            n = text["hidden_size"] * text["intermediate_size"]
            size = n // 2 + n // 32 * 4
        per_layer[layer] += size
    total = sum(per_layer)
    budget = max(0, vram_bytes - reserve)
    fits = reserve <= vram_bytes and total <= budget
    windows = []
    for width in (1, 4, 8):
        if width > layers:
            continue
        # Conservative active + next window capacity, including the cyclic boundary.
        working = max(sum(per_layer[(start + k) % layers] for k in range(min(2 * width, layers)))
                      for start in range(layers))
        if working <= budget:
            windows.append({"window_layers": width, "double_buffer_payload_bytes": working,
                            "working_set_bytes": working + reserve,
                            "steady_weight_h2d_bytes_per_token_if_no_persistent_layers": 0 if fits else total})
    return {
        "status": "capacity_estimate_not_runtime_measurement",
        "kv_bytes": kv, "linear_recurrent_state_bytes": recurrent,
        "linear_convolution_state_bytes": convolution,
        "fixed_attention_workspace_os_reserve_bytes": fixed_reserve_bytes,
        "context_tokens": context_tokens, "batch": batch, "kv_element_bytes": kv_element_bytes,
        "vram_bytes": vram_bytes, "budget_for_ffn_bytes": budget,
        "raw_ffn_bytes": sum(raw), "mixed_v1_ffn_payload_bytes": total,
        "compiled_gate_up_layers": sorted(compiled), "all_layers_resident": fits,
        "windows": windows, "tokens_per_second": None,
        "limits": [
            "Raw fallback sizes are capacity placeholders; their full runtime kernels are not implemented.",
            "FP16 KV and FP32 recurrent/conv state assumptions require backend verification.",
            "Fixed reserve is user-configurable and does not prove attention weights/workspace fit.",
            "Layer windows reduce peak capacity, NOT bytes streamed per decode sweep.",
            "Prefetch cannot be guaranteed to hide transfer latency; measure dependency critical paths.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Plan capacity only; no runtime or token-rate claims")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--vram-gib", type=float, default=8)
    parser.add_argument("--fixed-reserve-gib", type=float, default=1)
    parser.add_argument("--contexts", default="4096,8192,16384,32768")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reports = [plan_residency(manifest, config, vram_bytes=int(args.vram_gib * 2**30),
                             fixed_reserve_bytes=int(args.fixed_reserve_gib * 2**30),
                             context_tokens=int(n)) for n in args.contexts.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
