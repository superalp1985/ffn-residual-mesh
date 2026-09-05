# Resident Residual Qwen3.8-27B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace runtime radix-table compilation and residual-weight H2D transfers with offline static FFN decomposition and GPU-resident 4-bit residual weights for a Qwen3.8-27B-class model.

**Architecture:** The cold-start compiler reads the quantized GGUF once and writes a memory-mappable artifact containing static base coefficients, residual codes, scales, layer metadata, and exact fallback data. Runtime keeps the full base artifact in system RAM, keeps the current residual layer/window in VRAM, computes dynamic residual dots on CUDA, merges gate/up before SwiGLU, and pages only residual windows when all-layer residency does not fit.

**Tech Stack:** Python 3.12, NumPy, gguf-py, C++17, AVX2, CUDA 13 / PyTorch 2.9.1+cu130, Triton, existing llama.cpp GGUF loader.

**Spec:** `docs/math_principles_report.md`, `docs/log/2026-09-05-iteration-50.md`

## Global Constraints

- Do not make radix lookup a runtime prerequisite.
- Cold-start weight scanning and artifact compilation are excluded from steady-state latency.
- Gate and up residuals must merge before SiLU and elementwise multiplication.
- Exact integer reconstruction and approximate paths must be reported separately.
- GPU residency, H2D bytes, GPU wait, and layer critical path are primary metrics; FLOPs alone are insufficient.
- The first target is one Qwen3.8-27B layer, then a 4-layer window; no full-model claim before measured end-to-end evidence.
- Keep the original quantized GGUF as exact fallback; never delete or rewrite user downloads.

---

### Task 1: Acquire and validate the Qwen3.8-27B GGUF

**Files:**
- Create: `results/qwen38_27b_download_manifest.json`
- Modify: `docs/asset_manifest.md`
- Test: `tests/test_qwen38_asset_manifest.py`

**Interfaces:**
- Consumes: user-downloaded `Qwen3.8-27B-UD-Q4_K_M.gguf`.
- Produces: verified path, byte size, SHA256, GGUF tensor names, quantization types, and layer dimensions.

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_requires_qwen38_27b_gguf_and_ffn_tensors():
    manifest = load_manifest(...)
    assert manifest["model_id"] == "Qwen/Qwen3.8-27B"
    assert manifest["quantization"] == "Q4_K"
    assert manifest["dimensions"] == {"hidden": 5120, "ffn": 17408, "layers": 64}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_qwen38_asset_manifest -v`

Expected: FAIL because the downloaded model manifest does not exist.

- [ ] **Step 3: Implement the manifest validator**

Read the GGUF with the existing `scan_q4k_hierarchical_code_split.load_q4k_codes` loader, inspect one gate/up/down tensor per layer, record tensor type and dimensions, and compute SHA256 without loading the entire file into RAM.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_qwen38_asset_manifest -v`

Expected: PASS and all 64 layers expose gate/up/down tensors with hidden `5120` and intermediate `17408`.

- [ ] **Step 5: Commit**

```powershell
git add results/qwen38_27b_download_manifest.json docs/asset_manifest.md tests/test_qwen38_asset_manifest.py
git commit -m "Validate Qwen3.8-27B GGUF assets"
```

### Task 2: Build the offline static/residual compiler

**Files:**
- Create: `scripts/compile_resident_residual_artifact.py`
- Create: `src/resident_residual_format.py`
- Create: `tests/test_resident_residual_artifact.py`
- Modify: `docs/weight_code_split_spec.md`

**Interfaces:**
- `compile_layer(model: Path, layer: int, bits: int, out_dir: Path) -> dict`
- `ResidentArtifact.open(path: Path) -> ResidentArtifact`
- `ResidentArtifact.gate_up_bytes() -> int`
- `ResidentArtifact.residual_bytes() -> int`

- [ ] **Step 1: Write the failing test**

```python
def test_offline_artifact_roundtrips_q4_projection_exactly():
    artifact = compile_layer(model, layer=0, bits=4, out_dir=tempdir)
    loaded = ResidentArtifact.open(Path(artifact["path"]))
    np.testing.assert_array_equal(loaded.reconstruct_codes("gate"), original_gate_codes)
    assert loaded.manifest["runtime_requires_table_lookup"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_resident_residual_artifact -v`

Expected: FAIL because the offline compiler and artifact reader do not exist.

- [ ] **Step 3: Implement the compiler**

For every Q4_K group, choose a static base `c`, write `q = c + r`, store packed signed residual `r`, and preserve alpha/beta correction terms. Emit:

```text
manifest.json
gate.base.f16.bin
gate.residual.q4.bin
gate.alpha.f16.bin
gate.beta.f16.bin
up.base.f16.bin
up.residual.q4.bin
up.alpha.f16.bin
up.beta.f16.bin
down.q4.bin
fallback.index.json
```

The compiler must not emit a radix table and must retain an exact fallback pointer to the source GGUF tensor.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_resident_residual_artifact -v`

Expected: PASS with zero integer-code mismatches and exact Q4 reconstruction.

- [ ] **Step 5: Commit**

```powershell
git add scripts/compile_resident_residual_artifact.py src/resident_residual_format.py tests/test_resident_residual_artifact.py docs/weight_code_split_spec.md
git commit -m "Add offline resident residual artifact compiler"
```

### Task 3: Add GPU-resident residual projection

**Files:**
- Create: `src/resident_residual_cuda.cu`
- Create: `scripts/benchmark_resident_residual_cuda.py`
- Create: `tests/test_resident_residual_cuda.py`
- Modify: `scripts/build_and_run_cuda_residual_runner.py`

**Interfaces:**
- CUDA kernel `resident_q4_residual_dot(...)`
- Python benchmark fields: `resident_vram_bytes`, `dynamic_h2d_bytes`, `kernel_ms`, `effective_vram_read_gbps`, `max_abs_error`.

- [ ] **Step 1: Write the failing test**

```python
def test_resident_kernel_does_not_copy_residual_weights_per_token():
    report = run_resident_benchmark(...)
    assert report["dynamic_h2d_bytes"] <= activation_bytes + base_output_bytes
    assert report["resident_vram_bytes"] == artifact_residual_bytes
    assert report["max_abs_error"] < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_resident_residual_cuda -v`

Expected: FAIL because the resident kernel is not implemented.

- [ ] **Step 3: Implement the kernel**

Upload residual codes, alpha, and beta once before timing. For every token, transfer only activation descriptor/base output and compute `sum(r_i*z_i)` from VRAM-resident packed residual codes. Use tiled rows, pinned host buffers, and a CUDA stream. Record upload bytes separately from resident reads.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_resident_residual_cuda -v`

Expected: PASS with no per-token residual-weight H2D transfer and numerical error within the configured tolerance.

- [ ] **Step 5: Commit**

```powershell
git add src/resident_residual_cuda.cu scripts/benchmark_resident_residual_cuda.py tests/test_resident_residual_cuda.py scripts/build_and_run_cuda_residual_runner.py
git commit -m "Add GPU-resident residual projection kernel"
```

### Task 4: Implement layer/window residency scheduler

**Files:**
- Create: `src/resident_window_scheduler.py`
- Create: `scripts/simulate_resident_residual_windows.py`
- Create: `tests/test_resident_window_scheduler.py`
- Modify: `docs/layer_policy_manifest.example.json`

**Interfaces:**
- `ResidentWindowScheduler(window_layers: int, vram_budget_bytes: int)`
- `prefetch_window(layer_index: int) -> None`
- `acquire_layer(layer_index: int) -> ResidentLayer`
- `release_window(layer_index: int) -> None`

- [ ] **Step 1: Write the failing test**

```python
def test_scheduler_prefetches_next_window_without_evicting_active_layer():
    scheduler = ResidentWindowScheduler(window_layers=4, vram_budget_bytes=6 * 2**30)
    scheduler.prefetch_window(4)
    assert scheduler.active_layers() == [0, 1, 2, 3]
    assert scheduler.pending_layers() == [4, 5, 6, 7]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_resident_window_scheduler -v`

Expected: FAIL because the scheduler does not exist.

- [ ] **Step 3: Implement scheduler and budget accounting**

Support one-layer and four-layer windows first. Track active residual bytes, down bytes, CUDA workspace, attention/KV reserve, and fallback state. Use double-buffered asynchronous uploads for the next window. Never evict a layer while its gate/up/down kernels are outstanding.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_resident_window_scheduler -v`

Expected: PASS with deterministic eviction and no active-window overcommit.

- [ ] **Step 5: Commit**

```powershell
git add src/resident_window_scheduler.py scripts/simulate_resident_residual_windows.py tests/test_resident_window_scheduler.py docs/layer_policy_manifest.example.json
git commit -m "Add resident residual layer window scheduler"
```

### Task 5: Integrate gate/up merge, SwiGLU, and down

**Files:**
- Create: `scripts/benchmark_resident_ffn_pipeline.py`
- Create: `tests/test_resident_ffn_pipeline.py`
- Modify: `src/exact_cpu_base_gpu_full_ffn_runner.cu`

**Interfaces:**
- `run_resident_ffn(layer_artifact, hidden, scheduler, repeats) -> dict`
- Report keys: `cpu_base_ms`, `resident_residual_kernel_ms`, `swiglu_down_ms`, `critical_ms`, `dynamic_h2d_bytes`, `resident_vram_peak_bytes`, `output_rel_l2`.

- [ ] **Step 1: Write the failing test**

```python
def test_resident_ffn_merges_before_swiglu_and_matches_reference():
    report = run_resident_ffn(...)
    assert report["merge_order"] == "gate_up_before_swiglu"
    assert report["output_rel_l2"] < 2e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_resident_ffn_pipeline -v`

Expected: FAIL because the resident FFN pipeline is not integrated.

- [ ] **Step 3: Implement the pipeline**

Keep `gate_base` and `up_base` in host RAM, upload only base tiles/results and activations, compute residuals from resident VRAM codes, merge gate/up, execute SwiGLU, and run down using resident or tiled Q4 weights. Add exact center-GPU fallback on deadline or error-budget failure.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_resident_ffn_pipeline -v`

Expected: PASS with output error and byte counters recorded.

- [ ] **Step 5: Commit**

```powershell
git add scripts/benchmark_resident_ffn_pipeline.py tests/test_resident_ffn_pipeline.py src/exact_cpu_base_gpu_full_ffn_runner.cu
git commit -m "Integrate resident residual FFN pipeline"
```

### Task 6: Qwen3.8-27B end-to-end gate

**Files:**
- Create: `scripts/benchmark_qwen38_resident_end_to_end.py`
- Create: `docs/log/YYYY-MM-DD-qwen38-resident-benchmark.md`
- Modify: `README.md`

**Interfaces:**
- Output JSON fields: `tokens_per_second`, `first_token_seconds`, `steady_state_ms`, `attention_ms`, `ffn_ms`, `dynamic_h2d_bytes_per_token`, `resident_vram_peak_bytes`, `fallback_rate`, `status`.

- [ ] **Step 1: Write the failing test**

```python
def test_qwen38_result_refuses_unverified_20_tps_claim():
    report = run_benchmark(...)
    assert report["status"] in {"measured", "blocked_missing_asset"}
    if report["status"] == "measured":
        assert report["tokens_per_second"] <= report["verified_upper_bound"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_qwen38_resident_end_to_end -v`

Expected: FAIL until the model asset and runtime benchmark are present.

- [ ] **Step 3: Implement end-to-end benchmark**

Run a short fixed prompt, warm up separately, measure at least 32 generated tokens, record attention/KV and FFN separately, and refuse to label a result “20 token/s” unless the measured steady-state rate is at least 20 token/s with no unreported fallback or OOM.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_qwen38_resident_end_to_end -v`

Expected: PASS with either a measured result or an explicit missing-asset status.

- [ ] **Step 5: Commit**

```powershell
git add scripts/benchmark_qwen38_resident_end_to_end.py docs/log README.md tests/test_qwen38_resident_end_to_end.py
git commit -m "Add Qwen3.8 resident residual end-to-end gate"
```
