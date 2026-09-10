# Iteration 68: Controlled Base Reduction Tuning

Date: 2026-09-07

## Research and Hypothesis

Web-tool requests returned no content. Official sources were fetched directly
over HTTPS on this date:

- NVIDIA CUDA C Best Practices Guide, register pressure and occupancy:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- Triton fused softmax tutorial, compiled register/shared-memory metadata:
  https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- NVIDIA skills catalog checked without installing anything:
  https://raw.githubusercontent.com/NVIDIA/skills/main/skills.sh.json

Registers are shared by concurrent warps, but more occupancy alone does not
guarantee less runtime. Compiled register/spill counts are resource metadata,
not a measurement of achieved occupancy. CUDA event spans may include device
scheduling gaps, and must not be labelled active-SM time.

Hypothesis: the Q4 full-resident base/residual/SwiGLU kernel may benefit from
reducing the base coefficients in smaller group tiles instead of materializing
all 160 groups (rounded to 256 lanes). Test this single variable, retaining
all coefficients and residuals. This is the existing GPU-base comparison
path, not the CPU-base target and not a new claim about model residency.

## Protocol

- [x] Preserve iteration 67 correctness, traffic and unstable timing evidence.
- [x] Consult official sources before the experiment.
- [x] Test grouped base reduction with partial row/group tails and FP64 oracle.
- [x] Expose kernel register/spill metadata and keep the prior default (256).
- [x] Compile all variants before timed warmup.
- [x] Compare 8/16/32/64/128/256 in one process, randomized/interleaved.
- [x] Measure kernel graph spans and complete resident-FFN graph spans separately.
- [x] Record every sample, numerical errors, payload and memory tradeoff.
- [x] Repeat on another real Q4 layer; change defaults only with clear evidence.
- [x] Full regression, document decision, commit and push.

No paging, attention, KV cache, real-activation quality or generation claim.
No inference of hardware DMA/SM utilization from event timings alone.
