# GSQ-RCO Native Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after the user resumes. Steps use checkbox (`- [ ]`) syntax for tracking. Do not start an overnight run or create an automation from this document.

**Goal:** Establish whether the published GSQ-RCO 27B checkpoint improves useful single-user inference on this 8 GiB GPU, before investing in new residual kernels.

**Architecture:** Preserve the current split implementation and compare the existing UD checkpoint with the published GSQ-RCO IQ3_XXS checkpoint through one pinned native llama.cpp backend. Measure capacity, generation speed and quality separately. Humming and RCO integration are conditional follow-up investigations, not requirements for the first baseline.

**Tech Stack:** Windows, RTX 4070 Laptop GPU, 32 GiB system RAM, existing llama.cpp CUDA runtime, Python at `E:\MiniMaxH3\ComfyUI\venv\Scripts\python.exe`, existing PyTorch 2.9.1+cu130 environment, GGUF readers, unittest, local HTTP/JSON.

**Spec:** The approved scope below records the user's decision following the GSQ-RCO article discussion on 2026-09-11. Numerical baseline: `docs/log/2026-09-11-iteration-81.md`, commit `55a6879`.

## Approved Scope

- [x] Research article, official model card, GSQ/RCO repositories and allocation dumps.
- [x] Agree on native baseline first, kernel comparison second, selective split policy last.
- [x] Write and preserve this plan only. No experiment or model download tonight.
- [ ] Resume with the user on 2026-09-12 or their next requested session.

The objective remains a useful 27B dense model on this machine, with less
host/device traffic and less GPU waiting. Every layer need not use our split.
Offline preprocessing may be expensive; inference-time cost must be measured.

## Global Constraints

- Repository: `D:\ffn-residual-lab`; branch: `codex/resident-residual-qwen38`.
- Preserve existing models, compiled artifacts, default FP32 behavior and cu130 environment.
- Do not run upstream install scripts in the shared ComfyUI environment. Any new backend gets its own environment/directory after compatibility checks.
- Do not quantize the existing Q4 checkpoint again and present that as the published GSQ-RCO model. Use the released checkpoint for the first comparison.
- No full-model GSQ calibration/search, MTP, multimodal projector, phone cluster or KV algorithm work in this phase.
- KV/recurrent state and workspaces still count toward capacity even though their algorithms are unchanged.
- Each experiment starts with a focused official-source check and a written hypothesis. Pin the downloaded model revision and backend build.
- Prefer hidden local server processes and HTTP requests over interactive CLI sessions. Bound necessary inspection commands and redirect verbose output to files.
- Never infer physical H2D bytes, active-SM utilization or GPU idle time from file sizes or CUDA-event spans alone.
- Keep one GPU workload active at a time; preserve failures/OOMs/timeouts in results instead of silently dropping them.
- Default to inline execution. Review and commit independently testable changes; do not launch parallel GPU benchmarks.

## What We Know

Iteration 81 measured two resident FFN layers, not a complete model. FP16
coefficient storage improved their measured median latency by about 4%, but
introduced approximately `1e-4` relative-L2 error. The FP16 split payload
remained larger than original packed weights. It is still opt-in.

The published IQ3_XXS model card lists approximately 10.1 GB; this is a file
size, not its runtime GPU working set. The IQ2_XS alternative is listed as
8.4 GB and must not be advertised as automatically fitting this GPU.

The IQ3_XXS allocation dump inspected on 2026-09-11 listed 192 main FFN
tensors across ten formats. Only four of its 128 gate/up tensors were literal
Q4_K/Q5_K types accepted by our current affine-v2 compiler. This is not a
count of compatible complete layers; both gate and up must qualify.

For example, that dump assigns Layer 25 gate/up to IQ3_XXS and down to IQ3_S.
The existing Layer 25 artifact contains Q5_K tensors. Reusing it would test a
different checkpoint, not the new checkpoint's split implementation.

## Assets And Sources

Download on resumption, after checking free disk and source revision:

```text
Repository: ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
Primary: Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf
Target directory: E:\Qwen3.8-27B\GSQ-RCO\
Allocation: tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.rco-allocation.txt
```

FDM-compatible discovery URL (pin the revision before execution):

```text
https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf?download=true
```

No BF16 model, imatrix, MTP variant, projector or second quantization is
required for the first native baseline. Do not queue all variants by default.

Official sources already inspected; recheck only relevant updates tomorrow:

- Model card/allocation: https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
- GSQ paper: https://arxiv.org/abs/2604.18556
- GSQ implementation: https://github.com/IST-DASLab/GSQ
- RCO paper: https://arxiv.org/abs/2605.00649
- RCO implementation: https://github.com/IST-DASLab/RCO
- Humming: https://github.com/inclusionAI/humming
- Native backend: https://github.com/ggml-org/llama.cpp

GSQ's optimization and the particular release's IQ representation are not
interchangeable. Humming's integer kernels are not evidence of direct
support for these GGUF IQ codes. RCO's supplied single-budget formulation
does not automatically solve joint VRAM, transfer and latency constraints.

## File Map

Read/reuse without changing their current behavior:

- `scripts/validate_qwen38_manifest.py`: GGUF dimensions, FFN types, byte counts and hash.
- `scripts/inspect_gguf.py`: full tensor inventory, including non-FFN tensors.
- `scripts/diagnose_resident_latency.py`: later same-checkpoint resident FFN comparison.
- `scripts/run_baseline.ps1`: historical 2B benchmark; do not repurpose its fixed model path.
- `src/resident_residual_format.py`, `scripts/compile_resident_residual_artifact.py`: current format boundary.
- `vendor/llama.cpp/tools/server/README.md`: local reference for request/timing fields; installed binary help takes precedence.

Create only when executing the baseline task:

- `scripts/benchmark_native_gguf_http.py`: bounded sequential HTTP measurements and JSON output.
- `tests/test_native_gguf_http_benchmark.py`: timing/request validation without a real model.
- `docs/log/2026-09-12-iteration-82.md`: findings, provenance and next decision. Use actual date/next free iteration number if execution happens later.
- `results/gsq_rco_baseline/`: downloaded metadata snapshots, hashes, raw responses and manifests; generated data stays ignored.

Update `docs/asset_manifest.md` only after the new model is actually downloaded
and checked. No production kernel edits are planned in this phase.

## Task 1: Verify Model And Backend Compatibility

**Deliverable:** Verified asset manifest and a native model-load smoke result.

- [ ] Read this plan and iteration 81; inspect Git status and preserve any newer work.
- [ ] Record GPU/driver, available dedicated VRAM, RAM, disk space, power state and background GPU use. Do not change system power settings without a separate reason.
- [ ] Inspect the installed llama.cpp build/version/help in a bounded process. Confirm model architecture and the actual IQ formats are supported; if not, use a separate verified build, never overwrite the known-good runtime.
- [ ] Resolve a fixed Hugging Face revision; fetch its model card, allocation dump and source digest metadata. Download only the primary file into the separate target directory.
- [ ] Run `build_manifest()` on the new file and compare its SHA256 with the trusted upstream file digest. If unavailable, mark `local_digest_only`, not verified source integrity.
- [ ] Inspect all tensor bytes, main-layer/MTP counts and actual format assignments. Reconcile the downloaded header against the allocation dump. Keep the v1-only `compilable_gate_up_layers` field separate from a Q4/Q5-v2 compatibility assessment.
- [ ] Start one server bound to `127.0.0.1` on an unused port, with a conservative GPU-offload setting. Record the exact launch arguments, binary hash and logs. Disable speculative decoding and use one request slot.
- [ ] Verify health, perform one short completion, and save backend-reported model/KV/workspace allocations. Stop only the server process created by this task.
- [ ] Document any loader failure separately from algorithm performance. Commit the verified asset documentation, not the model or generated files.

Reuse the existing callable instead of writing a second GGUF parser:

```python
manifest = build_manifest(
    model_path,
    expected_sha256=source_sha256,
)
assert manifest["dimensions"] == {"hidden": 5120, "ffn": 17408, "layers": 64}
assert len(manifest["tensors"]) == 192
```

`model_path` is the downloaded file; `source_sha256` is the independently
retrieved digest, or `None` with the explicit integrity limitation above.
Do not allocate dequantized full-model arrays just to collect this inventory.

## Task 2: Test And Build A Small Native Measurement Harness

**Deliverable:** Reproducible request runner with unambiguous timing fields.

**Interfaces:**

- `summarize_response(response: dict, *, wall_seconds: float) -> dict`:
  returns `generated_tokens`, `prompt_tokens`, `decode_ms`,
  `decode_tokens_per_second`, `request_wall_ms`, `request_tokens_per_second`.
- `benchmark_requests(base_url: str, requests: list[dict], *, timeout_seconds: float = 300.0) -> list[dict]`:
  runs sequential requests; each record includes the original response,
  request fields, measured wall time, success/failure status and error details.

These are planned interfaces, not existing implementations. Required logic:
`decode_tokens_per_second = predicted_n * 1000 / predicted_ms` and
`request_tokens_per_second = predicted_n / wall_seconds`. Use actual counts,
never requested `n_predict`, and never mix prompt time into decode-only time.

- [ ] Write failing unittest cases for correct units, missing timings, zero/negative/non-finite durations, invalid token counts, premature EOS and failed HTTP responses.
- [ ] Run the focused suite and confirm failure at the missing implementation, before adding production code.
- [ ] Implement the two functions with standard HTTP/JSON tools. Save failed records with status/error; the script exits nonzero on an incomplete run, and generates no aggregate success result for failures.
- [ ] Keep request fields compatible with the inspected binary. Save actual seed, sampling/chat template, prompt tokens, cache settings, KV type, context size, CPU threads, batch/ubatch and offload policy alongside the response.
- [ ] Run tests, a small-model smoke test if an existing model is available, then the Task 1 server smoke case. Commit harness/tests separately from benchmark conclusions.

Minimal expected-value test:

```python
def test_distinguishes_decode_rate_from_request_rate(self):
    response = {"timings": {"prompt_n": 512, "predicted_n": 256,
                            "predicted_ms": 12800.0}}
    out = summarize_response(response, wall_seconds=15.0)
    self.assertEqual(out["decode_tokens_per_second"], 20.0)
    self.assertAlmostEqual(out["request_tokens_per_second"], 256 / 15)

def test_rejects_nonfinite_generation_time(self):
    response = {"timings": {"prompt_n": 512, "predicted_n": 256,
                            "predicted_ms": float("nan")}}
    with self.assertRaises(ValueError):
        summarize_response(response, wall_seconds=15.0)
```

Use literal expected values, plus a local HTTP fixture for failure/timeout
tests. Non-streamed request duration is not time-to-first-token; report TTFT
as unavailable unless separately measured from a streamed response.

## Task 3: Measure Capacity And Native Decode

**Deliverable:** Two-checkpoint report, not a split-kernel speedup claim.

- [ ] Pin one backend build for both models. Existing baseline is `E:\Qwen3.8-27B\Qwen3.8-27B-UD-Q4_K_M.gguf`; new checkpoint is the downloaded IQ3_XXS file.
- [ ] Freeze context at 4096, sequence concurrency at one, no MTP and identical KV types. Use the same tokenizer/template settings; document any metadata differences that prevent a controlled comparison.
- [ ] Find feasible offload settings progressively (8, 16, 24, 32, then steps of 4 near the budget limit). Stop escalation on OOM, shared-memory spill or inadequate headroom. Do not assume 64 GPU layers fit because the file is smaller.
- [ ] Reserve at least 512 MiB of dedicated VRAM after measured peak, or more if desktop use fluctuates. Include attention, recurrent state, KV, graph/workspace buffers and output projection; sample memory throughout the request.
- [ ] Run a shared-configuration comparison first, using the largest common safe offload count. Then measure each model's individually best safe setting under the same memory/context constraints. Label these two comparisons separately.
- [ ] Use identical tokenized prompts of approximately 512 and 2048 tokens, with 256 requested output tokens. First run is warmup; retain five measured runs per prompt/setting in each of two reversed-order model sessions. All GPU runs are sequential.
- [ ] For timing-only fixed-length probes, explicitly record `ignore_eos=true` and `cache_prompt=false`; never reuse these as quality samples. Record actual prompt/generated token counts and context shifting status.
- [ ] Save cold-start time separately. Summarize steady decode rate, request throughput and P95 duration. Preserve outliers rather than trimming results into a target rate.
- [ ] Collect actual transfer/idle measurements only when supported by a profiler. Otherwise mark those fields unavailable; CPU-resident inference does not necessarily stream those weights to GPU.
- [ ] Re-run the default test suite before committing code changes; report the actual current test count, not the old 120-test total by assumption.

Command for regression, run from repository root:

```powershell
& 'E:\MiniMaxH3\ComfyUI\venv\Scripts\python.exe' -m unittest discover -s tests -q
```

The 20 token/s target is assessed on actual steady single-sequence generation,
with prompt/context and placement disclosed. Also report user-visible request
throughput including prefill. Do not convert `1000 / single_layer_ms` or
batched/speculative throughput into this target.

## Task 4: Quality Check And Next Decision

**Deliverable:** An honest go/no-go report and the next bounded experiment.

- [ ] Before evaluating, freeze 20 unseen prompts: five Chinese instruction-following, five arithmetic with known answers, five code tasks with assertions, five structured-output tasks with schema checks. Use identical sampling and normal EOS for both models.
- [ ] Record pass counts, raw outputs and failure categories. This is a smoke check, not proof of model-wide equivalence or reproduction of published AIME/LCB scores.
- [ ] The provisional smoke acceptance rule is no new failures on deterministic arithmetic/code/schema cases that the existing checkpoint passes, plus no severe new Chinese instruction-following failure. Report differences for review rather than changing prompts or acceptance rules after seeing outputs.
- [ ] Keep quality-versus-quantization separate from kernel correctness: two kernels operating on the same quantized weights must be checked against that checkpoint's independent dequantized reference.
- [ ] If this native IQ3_XXS baseline is faster and passes the smoke check, treat it as the deployment reference to beat, not as a win for our split algorithm.
- [ ] If slower, use available evidence to distinguish CPU compute, exposed transfer, GPU decode cost and capacity spill before suggesting another representation. A smaller file is not itself a latency result.
- [ ] Record model/backend revisions, exact commands, all settings, integrity status, raw-result locations, median/P95 and limitations in iteration 82. Commit/push only completed, reviewed deliverables.

### Conditional Follow-Ups (Not Implemented By This Plan)

**Humming feasibility:** Check exact format support, Windows build support,
Ada/cu130 compatibility, group sizes and batch-one execution before installing.
If native Windows is unsupported, record that limitation and ask before an
environment migration. First compare the same representable quantized matrix
and the same FP32/BF16 activation convention. Do not unpack IQ weights to FP16,
requantize into INT3 and call that an equivalent kernel substitution. Measure
all runtime packed bytes, scales, temporary buffers, complete FFN latency and
numerical error; GEMM-only gains or larger-batch gains are insufficient.

**Selective policy/RCO:** First build a small table of measured candidate
representations and placements per layer: bytes, exposed transfer, latency and
held-out quality. Only then design a selection objective. CPU/GPU dependencies
and overlapping transfers mean measured layer times are not automatically
additive. The existing RCO single-budget method does not supply this complete
cost model or guarantee a multiple-constraint solution out of the box.

**GSQ optimization of residuals:** Consider only after the native/kernel
baseline identifies a worthwhile gap. Requires calibration, held-out quality
validation and a separately budgeted offline optimization environment. Added
quantization error must be labeled; do not inherit the released checkpoint's
quality claims after modifying its weights.

If a kernel candidate fails compatibility, same-weight correctness, runtime
byte accounting or repeated complete-FFN timing, stop that candidate and retain
the native baseline. No requirement to make every layer use a split.

## Resumption Checklist

- [ ] Confirm the user has asked to resume; inspect current date and Git state.
- [ ] Start at Task 1, not Humming installation or full-model GSQ training.
- [ ] Download the single IQ3_XXS model only after source/disk checks.
- [ ] Keep defaults and shared environments unchanged; save all experiment provenance.
- [ ] Decide the next experiment from measured results, not the article's headline.
