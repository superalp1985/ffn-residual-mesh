"""Protocol-level loopback for the distributed FFN base-worker design.

This module deliberately does not implement an Android runtime.  It gives the
ComfyUI wrapper a deterministic worker contract to test before replacing the
executor with TCP/USB/QUIC transports.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import struct
import time
import zlib
from typing import Callable, Iterable


MAGIC = b"PFFN"
VERSION = 1
REQUEST = 1
RESPONSE = 2
HEADER = struct.Struct("!4sBB8I")


class ProtocolError(ValueError):
    """Raised when a worker frame is malformed or fails integrity checks."""


@dataclass(frozen=True)
class Frame:
    kind: int
    run_id: int
    step_id: int
    block_id: int
    tile_id: int
    row_start: int
    row_count: int
    payload: bytes

    def encode(self) -> bytes:
        checksum = zlib.crc32(self.payload) & 0xFFFFFFFF
        header = HEADER.pack(
            MAGIC,
            VERSION,
            self.kind,
            self.run_id,
            self.step_id,
            self.block_id,
            self.tile_id,
            self.row_start,
            self.row_count,
            len(self.payload),
            checksum,
        )
        return header + self.payload

    @classmethod
    def decode(cls, packet: bytes) -> "Frame":
        if len(packet) < HEADER.size:
            raise ProtocolError("short frame")
        (
            magic,
            version,
            kind,
            run_id,
            step_id,
            block_id,
            tile_id,
            row_start,
            row_count,
            payload_len,
            checksum,
        ) = HEADER.unpack_from(packet)
        if magic != MAGIC or version != VERSION:
            raise ProtocolError("unsupported frame header")
        payload = packet[HEADER.size:]
        if payload_len != len(payload):
            raise ProtocolError("payload length mismatch")
        if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
            raise ProtocolError("payload checksum mismatch")
        if kind not in (REQUEST, RESPONSE):
            raise ProtocolError("unknown frame kind")
        return cls(
            kind,
            run_id,
            step_id,
            block_id,
            tile_id,
            row_start,
            row_count,
            payload,
        )


@dataclass(frozen=True)
class DispatchResult:
    responses: tuple[Frame, ...]
    fallback_tiles: tuple[int, ...]
    deadline_misses: int
    protocol_errors: int


class LoopbackWorker:
    """A row-sharded worker with the same request/response contract as a device."""

    def __init__(
        self,
        worker_id: int,
        row_start: int,
        row_count: int,
        compute: Callable[[Frame], bytes],
        delay_ms: float = 0.0,
    ) -> None:
        if worker_id < 0 or row_start < 0 or row_count <= 0:
            raise ValueError("invalid worker shard")
        self.worker_id = worker_id
        self.row_start = row_start
        self.row_count = row_count
        self.compute = compute
        self.delay_ms = max(0.0, delay_ms)

    def accepts(self, request: Frame) -> bool:
        return (
            request.kind == REQUEST
            and request.row_start >= self.row_start
            and request.row_start + request.row_count <= self.row_start + self.row_count
        )

    def handle(self, request: Frame) -> Frame:
        if not self.accepts(request):
            raise ProtocolError(f"worker {self.worker_id} rejected row shard")
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        payload = bytes(self.compute(request))
        return Frame(
            RESPONSE,
            request.run_id,
            request.step_id,
            request.block_id,
            request.tile_id,
            request.row_start,
            request.row_count,
            payload,
        )


class LoopbackCluster:
    """Concurrent row-shard scheduler with deadline and exact GPU fallback."""

    def __init__(self, workers: Iterable[LoopbackWorker]) -> None:
        self.workers = tuple(workers)
        if not self.workers:
            raise ValueError("at least one worker is required")

    def dispatch(self, requests: Iterable[Frame], deadline_ms: float) -> DispatchResult:
        requests = tuple(requests)
        if any(request.kind != REQUEST for request in requests):
            raise ProtocolError("dispatch accepts request frames only")
        if deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")

        started = time.monotonic()
        futures = {}
        responses: list[Frame] = []
        fallback: list[int] = []
        protocol_errors = 0
        with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            for request in requests:
                worker = next((item for item in self.workers if item.accepts(request)), None)
                if worker is None:
                    fallback.append(request.tile_id)
                    continue
                futures[pool.submit(worker.handle, request)] = request

            for future in as_completed(futures):
                request = futures[future]
                remaining = deadline_ms / 1000.0 - (time.monotonic() - started)
                if remaining <= 0:
                    fallback.append(request.tile_id)
                    continue
                try:
                    response = future.result(timeout=remaining)
                    # A real transport decodes a wire packet here.  Round-trip
                    # through the codec so tests cover framing and checksum.
                    responses.append(Frame.decode(response.encode()))
                except Exception:
                    protocol_errors += 1
                    fallback.append(request.tile_id)

        responses.sort(key=lambda item: item.tile_id)
        fallback.sort()
        return DispatchResult(
            tuple(responses),
            tuple(fallback),
            len(fallback),
            protocol_errors,
        )
