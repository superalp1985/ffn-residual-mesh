from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from build_and_run_cuda_residual_runner import find_nvcc, msvc_environment


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "exact_main_residual_cuda_runner.cu"
EXECUTABLE = ROOT / "src" / "exact_main_residual_cuda_runner.exe"


def compile_runner(arch: str) -> None:
    command = [
        str(find_nvcc()),
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        str(SOURCE),
        "-o",
        str(EXECUTABLE),
    ]
    subprocess.run(command, cwd=ROOT, env=msvc_environment(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU radix main plus GPU exact residual benchmark")
    parser.add_argument("table_artifact", type=Path)
    parser.add_argument("residual_artifact", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-rows", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()
    if args.force_build or not EXECUTABLE.is_file() or SOURCE.stat().st_mtime > EXECUTABLE.stat().st_mtime:
        compile_runner(args.arch)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(EXECUTABLE), str(args.table_artifact.resolve()), str(args.residual_artifact.resolve()),
         str(output), str(args.block_size), str(args.tile_rows), str(args.repeats), str(args.threads)],
        cwd=ROOT,
        check=True,
    )
    print(json.dumps(json.loads(output.read_text(encoding="utf-8")), ensure_ascii=False))


if __name__ == "__main__":
    main()
