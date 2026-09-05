from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmark_exact_radix_split_pipeline import (
    compile_radix_table,
    encode_signed_base4_states,
    load_q4_projection,
    quantize_groupwise_q8,
)


def compile_cpp(source: Path, output: Path) -> None:
    subprocess.run(
        ["g++", "-O3", "-march=native", "-std=c++17", str(source), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Native single-thread exact radix table lookup benchmark")
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--block-size", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    x = np.fromfile(args.input, dtype="<f4").reshape(1, -1)
    z, _ = quantize_groupwise_q8(x, group_size=32)
    source = Path(__file__).resolve().parents[1] / "src" / "exact_radix_table_bench.cpp"
    with tempfile.TemporaryDirectory(prefix="radix-table-") as temp:
        temp_root = Path(temp)
        executable = temp_root / "exact_radix_table_bench.exe"
        compile_cpp(source, executable)
        rows = []
        for projection in ("gate", "up"):
            codes, _, _, _ = load_q4_projection(args.model, args.layer, projection)
            table_start = time.perf_counter()
            table, _ = compile_radix_table(codes >> 2, block_size=args.block_size)
            compile_seconds = time.perf_counter() - table_start
            states = encode_signed_base4_states(z.reshape(-1), block_size=args.block_size)
            table_path = temp_root / f"{projection}.table.bin"
            states_path = temp_root / f"{projection}.states.u16"
            table_path.write_bytes(table.tobytes(order="C"))
            states.astype("<u2", copy=False).tofile(states_path)
            blocks = table.shape[0]
            output_rows = table.shape[2]
            groups = blocks // (32 // args.block_size)
            command = [
                str(executable),
                str(table_path),
                str(states_path),
                str(output_rows),
                str(blocks),
                str(groups),
                str(32 // args.block_size),
                str(4**args.block_size),
                str(args.repeats),
            ]
            native = json.loads(subprocess.check_output(command, text=True))
            rows.append(
                {
                    "projection": projection,
                    "cold_compile_seconds": compile_seconds,
                    "table_bytes": int(table.nbytes),
                    "runtime_logical_read_bytes_per_token": int(native["logical_table_read_bytes"]),
                    "runtime_logical_read_mib_per_token": native["logical_table_read_bytes"] / 2**20,
                    "native_single_thread": native,
                }
            )

    result = {"experiment": "native_exact_radix_table_lookup", "layer": args.layer, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
