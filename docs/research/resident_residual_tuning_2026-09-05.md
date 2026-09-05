# Resident Residual Tuning Research

Date: 2026-09-05

## Objective

The Qwen3.8-27B download is still in progress. This note records the tuning
work that can be completed before the real checkpoint arrives, so the first
27B run does not repeat the 2B mistakes.

Target runtime:

```text
static base and fallback weights -> system RAM
residual weights -> VRAM, uploaded once per resident layer/window
dynamic activation residual dot -> CUDA
gate/up merge -> before SwiGLU
down -> resident or tiled
```

Runtime radix lookup is optional. It is not part of the required architecture.

## Research findings

### 1. LUT layout is not the same as a good resident residual layout

Microsoft T-MAC demonstrates that small activation-derived tables and CPU
shuffle instructions can be effective for low-bit kernels. Its technique uses
carefully sized tables and vectorized lookup, not a 768 MiB-per-projection
activation-indexed table. NVIDIA CUDA guidance likewise recommends explicit
asynchronous global-to-shared copies and pipelined producer/consumer stages for
global-memory kernels. These references support the following design choices:

- keep residual codes in a contiguous `[layer, projection, output_tile,
  input_group, packed_code]` layout;
- align tile starts to at least 128 bytes;
- stage residual tiles into shared memory/register fragments;
- use double or triple buffering;
- fuse unpack, scale, dot accumulation, and epilogue where register pressure
  permits.

They do not prove our FFN split is fast. Every claim still needs local timing.

### 2. Existing 2B results reject two naive settings

On Qwen3.5-2B layer 23:

- 32-value scalar mean/midrange base with 2-bit residual had roughly `0.55–0.62`
  clipped code outlier fraction and projection relative error around `0.6`.
- 4-bit residual reduced error to near zero only because it retained nearly all
  original code information; the package was approximately the original Q4
  size.
- Subblock sizes 8, 16 and 32 changed the error only modestly for the scalar
  base; subblock 8 did not create a free 2-bit representation.
- Shared-pattern dictionaries reduced CPU base terms but still required a
  dense residual or incurred large approximation error.

Therefore the first 27B sweep must not assume that a smaller residual bit count
is useful. It must jointly measure error, resident bytes and GPU work.

### 3. Highest-value tuning knobs

The compiler should enumerate these independently per `(layer, projection,
output_tile, input_group)`:

1. **Base granularity**: 8, 16, 32 values; optionally 64 only when the
   activation covariance supports it.
2. **Base objective**: mean, midrange, activation-weighted integer center, and
   two-center piecewise base.
3. **Residual representation**: exact signed 4-bit, signed 3-bit, signed 2-bit
   plus exception stream, and mixed per-group bits.
4. **Exception policy**: bitmap + packed signed correction, byte-aligned
   exception blocks, or exact fallback for high-outlier groups.
5. **Scale storage**: fp16 by default; fp32 only for groups whose measured
   merge error exceeds budget.
6. **Output tiling**: 512, 1024, 2048, 4096 rows; select from measured kernel
   occupancy and launch overhead.
7. **Resident window**: 1, 2, 4 and 8 layers; reserve a fixed VRAM budget for
   attention/KV and allocator fragmentation.
8. **Kernel shape**: one warp per output row, one CTA per output tile, and
   grouped GEMM for batched tokens. Decode and prefill should not share one
   tile choice.
9. **Prefetch depth**: 1, 2, 3, 4 tile stages; stop when L2 hit rate or
   registers degrade.
10. **Merge placement**: GPU fused epilogue preferred; CPU merge is a fallback
    only when it avoids a larger H2D transfer.

### 4. Mixed-bit residual is more promising than global 1/2/4-bit

A global 2-bit residual is not viable on the 2B evidence because too many
groups overflow. A global 4-bit residual is accurate but does not save much
space. The likely useful format is:

```text
per group:
  static base descriptor
  2-bit or 3-bit residual core
  exception bitmap
  packed 4/8-bit exception values
```

The compiler should choose the cheapest exact or error-bounded variant for each
group. Groups with high outlier density should use 4-bit or exact fallback;
regular groups can remain 2-bit. This turns the residual byte budget into a
distribution problem instead of a single global bit choice.

### 5. Static split and dynamic residual must be separated in accounting

Static artifacts are paid once at cold start. Dynamic quantities are paid per
token:

```text
static:
  base descriptors, residual weights, scales, exception indexes

dynamic:
  activation tile, activation sums, base output tile, residual output
```

The residual **weight** should be resident in VRAM. The residual **result** is
input-dependent and only needs short-lived buffers. A benchmark that counts
resident weight bytes as per-token H2D is measuring the wrong architecture.

### 6. Kernel optimization order

Do not start with more algebra. Use this order:

1. Verify exact code reconstruction on random and real Q4_K groups.
2. Upload residual weights once and prove zero residual-weight H2D per token.
3. Benchmark unpack-only, dot-only, and fused residual kernels.
4. Add shared-memory staging and async copy.
5. Fuse gate/up merge and SwiGLU.
6. Add down projection tiling and overlap.
7. Only then compare mixed-bit formats.

This isolates whether a regression comes from decomposition, packing, memory
layout, or kernel occupancy.

## Pre-27B experimental matrix

The first real checkpoint run should produce one JSON row per configuration:

```text
base_granularity: 8,16,32
base_objective: mean, activation_weighted, two_center
residual: 2bit+exceptions, 3bit+exceptions, 4bit_exact
scale: fp16, fp32_selective
tile_rows: 1024,2048,4096
resident_window: 1,4,8
```

For each row record:

```text
output_rel_l2
gate_rel_l2
up_rel_l2
fallback_fraction
resident_vram_bytes
dynamic_h2d_bytes_per_token
gpu_kernel_ms
gpu_wait_ms
cpu_base_ms
critical_ms
```

## Acceptance gates

Promote a configuration only if all are true:

1. Exact mode has zero integer reconstruction mismatches.
2. Approximate mode meets an explicitly recorded final-FFN error budget.
3. Residual weights are uploaded once per resident window, not once per token.
4. The GPU resident package plus workspace stays below the measured VRAM
   budget.
5. The fused kernel beats the direct Q4 baseline on the same shape, or it
   demonstrably reduces H2D/wait time in the full pipeline.
6. Any 20-token/s statement is based on measured steady-state generation, not
   a bandwidth-only lower bound.

## References

- T-MAC: CPU LUT kernels for low-bit LLMs,
  https://arxiv.org/abs/2407.00088
- T-MAC implementation,
  https://github.com/microsoft/T-MAC
- NVIDIA CUDA Best Practices, async global-to-shared copy and pipelining,
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- NVIDIA CUDA Programming Guide, pipeline synchronization,
  https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
