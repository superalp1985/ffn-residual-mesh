import sys
import time
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phone_ffn_loopback import (  # noqa: E402
    Frame,
    LoopbackCluster,
    LoopbackWorker,
    ProtocolError,
    REQUEST,
)


def request(tile_id: int, row_start: int = 0, row_count: int = 4) -> Frame:
    return Frame(REQUEST, 7, 2, 3, tile_id, row_start, row_count, b"descriptor")


class PhoneFfnLoopbackTests(unittest.TestCase):
    def test_frame_round_trip_and_checksum(self) -> None:
        frame = request(4)
        self.assertEqual(Frame.decode(frame.encode()), frame)
        corrupt = bytearray(frame.encode())
        corrupt[-1] ^= 1
        with self.assertRaises(ProtocolError):
            Frame.decode(bytes(corrupt))

    def test_workers_return_ordered_rows_concurrently(self) -> None:
        workers = [
            LoopbackWorker(0, 0, 4, lambda frame: b"base-a"),
            LoopbackWorker(1, 4, 4, lambda frame: b"base-b"),
        ]
        result = LoopbackCluster(workers).dispatch(
            [request(1, 0), request(2, 4)], deadline_ms=100
        )
        self.assertEqual([item.tile_id for item in result.responses], [1, 2])
        self.assertEqual([item.payload for item in result.responses], [b"base-a", b"base-b"])
        self.assertEqual(result.fallback_tiles, ())
        self.assertEqual(result.deadline_misses, 0)

    def test_deadline_miss_is_explicit_fallback(self) -> None:
        worker = LoopbackWorker(0, 0, 4, lambda frame: b"late", delay_ms=30)
        started = time.monotonic()
        result = LoopbackCluster([worker]).dispatch([request(1)], deadline_ms=1)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        self.assertEqual(result.responses, ())
        self.assertEqual(result.fallback_tiles, (1,))
        self.assertEqual(result.deadline_misses, 1)
        self.assertLess(elapsed_ms, 100)


if __name__ == "__main__":
    unittest.main()
