from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_exact_radix_split_pipeline import load_q4_projection


def main() -> None:
    parser = argparse.ArgumentParser(description="AVX2 radix, prefetch, and direct packed-Q4 CPU comparison")
    parser.add_argument("model", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--prefetch", default="0,1,2,4,8")
    parser.add_argument("--projections", nargs="+", default=["gate", "up"], choices=["gate", "up"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens + args.threads + [args.repeats]) < 1:
        parser.error("tokens, threads and repeats must be positive")
    manifest = json.loads((args.artifact / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "experiment": "radix_cpu_avx2_prefetch",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "layer": args.layer,
        "block_size": manifest["block_size"],
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "flags": ["-O3", "-mavx2", "-ffp-contract=off", "-std=c++17", "-pthread"],
        "scope": {
            "direct": "full Q4 projection; cold-repacked nibbles and expanded FP32 alpha/beta; NOT ggml Q4_K",
            "table": "high/base contribution ONLY, excluding GPU residual and transport",
            "timed": "per-input state/sum preparation, persistent worker dispatch, projection, metadata merge",
            "excluded": "file IO, cold packing/transposition, Q8 quantization, output verification/checksum",
            "tokens": "1=captured Q8 input repeated; >1=captured input then seeded synthetic int8 stress inputs",
            "cache": "warm allocations, changing input cases interleaved; NOT a whole-model cold-cache benchmark",
        },
        "runs": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="radix-perf-", dir=args.out.parent) as temp:
        directory = Path(temp)
        exe = directory / "bench.exe"
        subprocess.run(
            ["g++", *result["flags"], str(ROOT / "src" / "radix_cpu_bench.cpp"), "-o", str(exe)],
            check=True, capture_output=True, text=True,
        )
        for name in args.projections:
            q, alpha, beta, source_bytes = load_q4_projection(args.model, args.layer, name)
            rows, groups, _ = q.shape
            hidden = groups * 32
            q.tofile(directory / "q.bin")
            alpha.astype("<f4").tofile(directory / "alpha.bin")
            beta.astype("<f4").tofile(directory / "beta.bin")
            for stem, suffix in [("table", "table.u8"), ("high_sum", "high_sum.i16")]:
                source = (args.artifact / f"{name}.{suffix}.bin").resolve()
                target = directory / f"{stem}.bin"
                if target.exists():
                    target.unlink()
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            for tokens in args.tokens:
                rng = np.random.default_rng(49)
                z = rng.integers(-128, 128, (tokens, hidden), dtype=np.int16).astype("i1")
                z[0] = np.fromfile(args.artifact / "activation.z.i8.bin", dtype="i1")
                scale = np.fromfile(args.artifact / "activation.scale.f32.bin", dtype="<f4")
                z.tofile(directory / "z.bin")
                np.broadcast_to(scale, (tokens, groups)).copy().tofile(directory / "scale.bin")
                for threads in args.threads:
                    print(f"measuring {name}, block={manifest['block_size']}, tokens={tokens}, threads={threads}", flush=True)
                    completed = subprocess.run(
                        [str(exe), str(directory), str(rows), str(hidden), str(manifest["block_size"]),
                         str(tokens), str(args.repeats), str(threads), args.prefetch],
                        check=True, capture_output=True, text=True, timeout=300,
                    )
                    native = json.loads(completed.stdout)
                    if native["integer_mismatches"] or native["max_scaled_abs_error"] > 1e-3:
                        raise RuntimeError("native correctness check failed")
                    native.update(projection=name, source_q4_bytes=source_bytes, rows=rows, hidden=hidden)
                    result["runs"].append(native)
                    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    print(" ".join(f"{m['name']}={m['median_ms']:.3f}ms" for m in native["methods"]), flush=True)


if __name__ == "__main__":
    main()
