from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_exact_radix_split_pipeline import (
    encode_signed_base4_states,
    projection_from_group_dots,
)


class NativeRadixBase:
    """Reusable CPU-side base evaluator.

    The C++ benchmark is the performance implementation. This Python-facing
    adapter keeps the same cold-start artifact contract and is used by tests
    and orchestration code until the shared-library ABI is stabilized.
    """

    def __init__(
        self,
        artifact_dir: Path,
        projection: str,
        rows: int,
        hidden: int,
        block_size: int,
        *,
        threads: int = 1,
        prefetch: int = 0,
        mode: str = "fused",
    ) -> None:
        if mode not in {"legacy", "fused"}:
            raise ValueError("mode must be legacy or fused")
        if threads < 1 or prefetch < 0:
            raise ValueError("threads must be positive and prefetch non-negative")
        if hidden % 32 or block_size not in (2, 4) or hidden % block_size:
            raise ValueError("hidden must be divisible by 32 and block size")
        self.rows, self.hidden, self.block_size = rows, hidden, block_size
        self.groups = hidden // 32
        self.blocks = hidden // block_size
        self.blocks_per_group = 32 // block_size
        root = Path(artifact_dir)
        self.table = np.memmap(
            root / f"{projection}.table.u8.bin",
            mode="r",
            dtype=np.uint8,
            shape=(self.blocks, 4**block_size, rows),
        )
        self.high_sum = np.memmap(
            root / f"{projection}.high_sum.i16.bin",
            mode="r",
            dtype="<i2",
            shape=(rows, self.groups),
        )
        self.alpha = np.memmap(
            root / f"{projection}.alpha.f32.bin",
            mode="r",
            dtype="<f4",
            shape=(rows, self.groups),
        )
        self.beta = np.memmap(
            root / f"{projection}.beta.f32.bin",
            mode="r",
            dtype="<f4",
            shape=(rows, self.groups),
        )
        self.threads, self.prefetch, self.mode = threads, prefetch, mode

    def evaluate(self, z: np.ndarray, scales: np.ndarray, output: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.int8)
        scales = np.asarray(scales, dtype=np.float32)
        output = np.asarray(output)
        if z.shape != (1, self.hidden):
            raise ValueError(f"z must have shape (1,{self.hidden})")
        if scales.shape not in {(1, self.groups), (self.groups,)}:
            raise ValueError(f"scales must have shape (1,{self.groups}) or ({self.groups},)")
        if output.shape != (self.rows,):
            raise ValueError(f"output must have shape ({self.rows},)")
        states = encode_signed_base4_states(z[0], self.block_size)
        selected = np.zeros((self.blocks, self.rows), dtype=np.int32)
        block_index = np.arange(self.blocks)
        for digit in range(4):
            selected += (4**digit) * self.table[block_index, states[digit]].astype(np.int32)
        group_dot = selected.reshape(self.groups, self.blocks_per_group, self.rows).sum(axis=1).T
        group_dot -= 128 * self.high_sum
        scale = scales.reshape(-1)
        z_sum = z.reshape(self.groups, 32).astype(np.int32).sum(axis=1)
        output[:] = projection_from_group_dots(
            group_dot[:, :, None] if False else group_dot,
            z,
            scale.reshape(1, -1),
            self.alpha,
            self.beta,
            code_multiplier=4,
        )

    def close(self) -> None:
        for value in (self.table, self.high_sum, self.alpha, self.beta):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "NativeRadixBase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
