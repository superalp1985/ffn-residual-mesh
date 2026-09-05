from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FORMAT = "FFN_RESIDENT_CENTERED_Q4_V1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResidentArtifact:
    def __init__(self, directory: Path, manifest: dict):
        self.directory = directory
        self.manifest = manifest
        self.projections = manifest["projections"]
        self.fallbacks = manifest["fallbacks"]
        self.arrays: dict[str, dict[str, np.memmap]] = {}

    @classmethod
    def open(cls, path: Path, *, verify_hashes: bool = False) -> ResidentArtifact:
        manifest_path = path / "manifest.json" if path.is_dir() else path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("residual_bits") != 4:
            raise ValueError("unsupported resident artifact format")
        result = cls(manifest_path.parent.resolve(), manifest)
        try:
            for name, projection in result.projections.items():
                rows, cols = projection["rows"], projection["cols"]
                shapes = {
                    "residual": ([rows, cols // 2], "|u1"),
                    "base": ([rows, cols // 32], "|u1"),
                    "alpha": ([rows, cols // 32], "<f4"),
                    "beta": ([rows, cols // 32], "<f4"),
                    "coefficient": ([rows, cols // 32], "<f8"),
                }
                if rows <= 0 or cols <= 0 or cols % 256:
                    raise ValueError("invalid projection dimensions")
                result.arrays[name] = {}
                for kind, (shape, dtype) in shapes.items():
                    entry = projection["files"][kind]
                    target = (result.directory / entry["file"]).resolve()
                    if target.parent != result.directory:
                        raise ValueError("artifact payload escapes directory")
                    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                    if entry["shape"] != shape or np.dtype(entry["dtype"]) != np.dtype(dtype):
                        raise ValueError(f"invalid payload descriptor: {target}")
                    if entry["bytes"] != size or target.stat().st_size != size:
                        raise ValueError(f"invalid payload byte size: {target}")
                    if verify_hashes and file_sha256(target) != entry["sha256"]:
                        raise ValueError(f"SHA256 mismatch: {target}")
                    result.arrays[name][kind] = np.memmap(target, mode="r", dtype=dtype, shape=tuple(shape))
        except Exception:
            result.close()
            raise
        return result

    def close(self) -> None:
        for projection in self.arrays.values():
            for array in projection.values():
                array._mmap.close()
        self.arrays.clear()

    def __enter__(self) -> ResidentArtifact:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def residual_bytes(self, projection: str | None = None) -> int:
        names = (projection,) if projection else tuple(self.projections)
        return sum(self.projections[name]["files"]["residual"]["bytes"] for name in names)

    def gate_up_bytes(self) -> int:
        return sum(self.projections[name]["files"][kind]["bytes"]
                   for name in ("gate", "up") if name in self.projections
                   for kind in ("residual", "alpha"))

    def unpack_residual(self, name: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        packed = self.arrays[name]["residual"][start:stop]
        output = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
        for shift, offset in ((0, 0), (4, 1)):
            nibble = ((packed >> shift) & 15).astype(np.int16)
            output[:, offset::2] = np.where(nibble >= 8, nibble - 16, nibble)
        return output

    def reconstruct_codes(self, name: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        residual = self.unpack_residual(name, start, stop)
        codes = residual.reshape(residual.shape[0], -1, 32).astype(np.int16)
        codes += self.arrays[name]["base"][start:stop, :, None]
        if np.any((codes < 0) | (codes > 15)):
            raise ValueError("reconstructed code outside Q4 range")
        return codes.reshape(residual.shape).astype(np.uint8)

    def reconstruct_weights(self, name: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        codes = self.reconstruct_codes(name, start, stop)
        groups = codes.reshape(codes.shape[0], -1, 32).astype(np.float32)
        return (self.arrays[name]["alpha"][start:stop, :, None] * groups
                + self.arrays[name]["beta"][start:stop, :, None]).reshape(codes.shape)

    def project_parts(self, name: str, activation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(activation, dtype=np.float64)
        if x.shape != (self.projections[name]["cols"],):
            raise ValueError("activation must match projection input width")
        grouped = x.reshape(-1, 32)
        base = self.arrays[name]["coefficient"] @ grouped.sum(axis=1)
        output = np.empty(self.projections[name]["rows"], dtype=np.float64)
        for start in range(0, len(output), 128):
            stop = min(start + 128, len(output))
            residual = self.unpack_residual(name, start, stop).reshape(stop - start, -1, 32)
            dots = np.einsum("rgi,gi->rg", residual, grouped, optimize=True)
            output[start:stop] = (dots * self.arrays[name]["alpha"][start:stop]).sum(axis=1)
        return base, output
