"""Complete FFN worker loopback used before attaching a wired device."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from phone_ffn_loopback import Frame, LoopbackCluster, LoopbackWorker, REQUEST, RESPONSE


def silu(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (1.0 + np.exp(-values))


def direct_ffn(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    gate_out = values @ np.asarray(gate, dtype=np.float32).T
    up_out = values @ np.asarray(up, dtype=np.float32).T
    return (silu(gate_out) * up_out) @ np.asarray(down, dtype=np.float32).T


@dataclass
class FullFfnLayer:
    layer_id: int
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray

    @property
    def hidden(self) -> int:
        return int(self.gate.shape[1])

    @property
    def ffn(self) -> int:
        return int(self.gate.shape[0])

    def encode_input(self, x: np.ndarray) -> bytes:
        values = np.asarray(x, dtype=np.float16)
        if values.ndim != 2 or values.shape[1] != self.hidden:
            raise ValueError("input shape does not match FFN hidden size")
        return values.astype("<f2", copy=False).tobytes()

    def worker(self, delay_ms: float = 0.0) -> LoopbackWorker:
        def compute(request: Frame) -> bytes:
            if request.block_id != self.layer_id:
                raise ValueError("unexpected layer id")
            if len(request.payload) % (self.hidden * 2):
                raise ValueError("input payload is not fp16 hidden rows")
            rows = len(request.payload) // (self.hidden * 2)
            values = np.frombuffer(request.payload, dtype="<f2").astype(np.float32).reshape(rows, self.hidden)
            output = direct_ffn(values, self.gate, self.up, self.down)
            return np.asarray(output, dtype="<f2").tobytes()

        return LoopbackWorker(self.layer_id, self.layer_id, 1, compute, delay_ms=delay_ms)


def run_full_ffn_layer(
    layer: FullFfnLayer,
    x: np.ndarray,
    deadline_ms: float = 1000.0,
    delay_ms: float = 0.0,
) -> np.ndarray:
    request = Frame(
        REQUEST,
        1,
        1,
        layer.layer_id,
        layer.layer_id,
        layer.layer_id,
        1,
        layer.encode_input(x),
    )
    result = LoopbackCluster([layer.worker(delay_ms)]).dispatch([request], deadline_ms)
    if result.fallback_tiles:
        raise TimeoutError(f"full FFN worker fallback for layer {layer.layer_id}")
    response = result.responses[0]
    rows = x.shape[0]
    if len(response.payload) != rows * layer.hidden * 2:
        raise ValueError("worker returned an unexpected hidden shape")
    return np.frombuffer(response.payload, dtype="<f2").astype(np.float32).reshape(rows, layer.hidden)


def random_layer(layer_id: int, hidden: int, ffn: int, seed: int = 0) -> FullFfnLayer:
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(hidden)
    return FullFfnLayer(
        layer_id,
        (rng.standard_normal((ffn, hidden)) * scale).astype(np.float32),
        (rng.standard_normal((ffn, hidden)) * scale).astype(np.float32),
        (rng.standard_normal((hidden, ffn)) * scale).astype(np.float32),
    )


__all__ = ["FullFfnLayer", "direct_ffn", "run_full_ffn_layer", "random_layer", "silu"]
