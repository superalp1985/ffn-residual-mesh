from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "exact_residual_cuda_runner.cu"
EXECUTABLE = ROOT / "src" / "exact_residual_cuda_runner.exe"


def find_nvcc() -> Path:
    candidates = [
        Path(os.environ.get("CUDA_PATH", "")) / "bin" / "nvcc.exe",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("nvcc.exe was not found")


def find_vcvars64() -> Path:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("vcvars64.bat was not found")


def msvc_environment() -> dict[str, str]:
    vcvars = find_vcvars64()
    completed = subprocess.run(
        f'call "{vcvars}" && set',
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"vcvars64.bat failed: {completed.stderr.strip() or completed.stdout.strip()}")
    environment = os.environ.copy()
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            environment[key] = value
    return environment


def compile_runner(nvcc: Path, arch: str) -> None:
    environment = msvc_environment()
    command = [
        str(nvcc),
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        str(SOURCE),
        "-o",
        str(EXECUTABLE),
    ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run the exact QLO2 CUDA residual benchmark")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-rows", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    output = args.out.resolve()
    if not (artifact / "manifest.json").is_file():
        raise FileNotFoundError(f"missing artifact manifest: {artifact / 'manifest.json'}")
    if args.force_build or not EXECUTABLE.is_file() or SOURCE.stat().st_mtime > EXECUTABLE.stat().st_mtime:
        compile_runner(find_nvcc(), args.arch)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(EXECUTABLE), str(artifact), str(output), str(args.tile_rows), str(args.repeats)],
        cwd=ROOT,
        check=True,
    )
    print(json.dumps(json.loads(output.read_text(encoding="utf-8")), ensure_ascii=False))


if __name__ == "__main__":
    main()
