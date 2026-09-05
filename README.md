# FFN Residual Mesh

> **Trade compute for bandwidth.**

FFN Residual Mesh is an experimental runtime for running larger dense models on small-VRAM GPUs.

Its core idea is simple but deliberately aggressive:

> **Pre-expand the bulky FFN base terms during cold start, keep them in host/tablet memory, send only compact residual tiles to the center GPU, and merge them back before the original nonlinear expression.**

This is not a FLOPs-reduction trick. It is a **compute-for-bandwidth exchange**: the GPU does more regular arithmetic so it does not sit idle waiting for large weight streams over PCIe or device memory bandwidth.

## The Hardware Problem

Modern GPUs often have enormous peak throughput, but an FFN can still run at low **effective compute density**. The matrix units are ready to work; the workload is waiting for weight bytes, cache misses, PCIe transfers, synchronization, or a larger VRAM window. In this regime, the practical bottleneck is not theoretical FLOPs. It is the combination of:

- GPU arithmetic units not being continuously fed;
- limited VRAM capacity for large dense weights;
- limited H2D/device-memory bandwidth;
- host and device repeatedly moving bulky, highly repetitive FFN data.

FFN Residual Mesh makes this an explicit resource exchange:

| We spend | We get back |
|---|---|
| cold-start scan and compilation time | pre-expanded base artifacts in host/worker memory |
| CPU, tablet, or phone serial/vector arithmetic | less bulky FFN weight traffic to the GPU |
| additional regular GPU residual arithmetic | higher useful work per transferred byte |
| RAM and worker storage | freedom from keeping every FFN base term resident in VRAM |

The algorithm does not claim to create free compute. It moves work to the place where that work is cheaper: regular residual arithmetic stays on CUDA, while repetitive FFN base information is expanded ahead of time and kept outside the most expensive bandwidth path.

## Why Earlier Phone-Cluster Attempts Stalled

The old approach was usually: split a model layer across phones, then move large weight shards or full-layer tensors for every inference step. That makes network traffic scale with model size, so a phone pool spends its time transferring weights instead of computing.

FFN Residual Mesh changes the traffic pattern:

```text
cold start:  scan full FFN weights once -> compile resident base artifacts
runtime:    exchange descriptors + tiles -> compute residuals -> merge on GPU
```

The model weights are not re-sent as a dense layer on every step. The link carries only the active descriptor, base/residual tile, scale and protocol metadata. This turns a model-size transfer problem into a tile scheduling problem. It does not make bandwidth infinite: exact MiniMax H3 gate/up return is still large enough to require 10GbE, high-throughput USB, or a more compact exact artifact. The important change is that full-layer weight movement is no longer the default path.

## What We Claim

- A layered FFN split/merge algorithm that can be exact in the selected integer/quantized domain.
- A runtime contract that keeps the base term outside precious GPU memory and sends residual work to CUDA.
- A cold-start compiler model: scan the original weights once, build finite formulas/partial sums/residual artifacts, then avoid rescanning dense weight streams during normal execution.
- A practical path toward **Windows + NVIDIA CUDA 13 + ComfyUI + TeaCache + wired tablet workers**.
- A phone/tablet worker protocol with row sharding, checksums, deadlines, concurrent dispatch, and exact GPU fallback.
- A complete-FFN worker baseline: the device owns a full FFN layer and returns only the post-down hidden result.

The point is not to pretend that a phone replaces a GPU. The point is to make idle GPU arithmetic useful while the host and device stop fighting over the same bandwidth bottleneck.

## The Algorithm

For a gated FFN:

```text
g = W_gate x
u = W_up   x
y = W_down (SiLU(g) * u)
```

We compile each layer and projection into:

```text
W = W_base + W_residual
g = g_base + g_residual
u = u_base + u_residual
y = W_down (SiLU(g) * u)
```

The merge happens **before** SiLU and before the gated product. We do not illegally add two separately activated branches. In the integer domain, radix, centered-block, hierarchical, and finite-state formulas can preserve the missing cross terms exactly; if a path is approximate, its error and fallback policy are reported separately.

The runtime objective is:

```text
maximize   GPU work / transferred byte
minimize   GPU wait and synchronization
subject to exactness or an explicit error budget
```

## Why This Is Different

Generic distributed inference splits tensors across devices. FFN Residual Mesh targets a narrower systems problem:

1. **FFN-first**: attack the largest repeatable dense projection path before rewriting the whole model.
2. **Base/residual split**: the bulky, repetitive part is pre-expanded and resident in RAM or worker memory; the irregular part is computed densely on CUDA.
3. **Nonlinear-safe merge**: gate and up are reconstructed before SwiGLU, so the algebraic boundary is explicit.
4. **Bandwidth-first accounting**: H2D bytes, GPU wait, overlap, resident peak, and fallback rate matter more than headline FLOPs.
5. **Graceful fallback**: worker timeout, checksum failure, unsupported layer, or error-budget miss returns to the center GPU's exact FFN path.

## Current Evidence

The repository currently contains:

- Exact integer-domain FFN base/residual reconstruction tests.
- A native AVX2 CPU evaluator with software-prefetch distance sweeps and a
  direct packed-Q4 comparison. It is deliberately labeled as a benchmark, not
  as a claim that lookup always wins.
- CUDA layer-level CPU-base + GPU-residual + SwiGLU/down bridge experiments.
- MiniMax H3 ComfyUI analytical simulation using the local H3 tensor layout.
- TeaCache real/reuse-step accounting.
- A protocol-level phone/tablet loopback worker.
- Qwen3.8-27B-class memory and H2D budget scripts.

### MiniMax H3 / ComfyUI Simulation

Default workload: 832x480, 124 frames, 20 sampling steps, 8 TeaCache real steps, 12 reuse steps, 50 H3 blocks.

| Path | Link assumption | Result |
|---|---:|---:|
| Exact gate/up return | 1 Gb/s | ~839.8 MiB returned per real step; phone branch remains critical; ~116.7 s estimated total |
| Exact gate/up return | 10 Gb/s | phone branch hidden under the modeled GPU branch; ~114.2 s estimated total |
| Hidden approximation | 1 Gb/s | ~157.5 MiB per real step, but not algebraically exact |

This is an **analytical model, not an Android benchmark**. It supports the hardware decision: start with USB/10GbE tablets, not ordinary Wi-Fi phones, for the exact path.

### CPU Lookup Performance Note

The cold-expanded table is much larger than the source Q4 stream. On the local
i7-14650HX, Qwen3.5-2B layer 23, block-4, one token:

```text
direct packed-Q4 AVX2                 ~0.76 ms / projection
old table evaluator                   ~2.26 ms / projection
fused AVX2 table + software prefetch  ~1.27 ms / projection
```

At eight workers the fused table reaches roughly `0.23–0.27 ms` per projection,
close to the local direct-Q4 baseline (`~0.20–0.23 ms`). Block-2 remains slower
because it performs more table-address selections. These are warm local CPU
microbenchmarks; they exclude file I/O and are not a full-model or Android
result. The practical rule is: use pre-expanded tables only when their lookup
locality beats the native packed-Q4 kernel; otherwise keep the table as an exact
oracle and use the direct/worker path.

## Quick Start

Run the MiniMax H3/ComfyUI transport simulation:

```powershell
python scripts/simulate_comfyui_phone_ffn.py `
  --width 832 --height 480 --frames 124 `
  --steps 20 --tea-real-steps 8 `
  --network-gbps 10 --phone-return gate_up_exact `
  --out results/comfyui_h3_phone_exact_10gbps.json
```

Run the phone-cluster budget model:

```powershell
python scripts/simulate_phone_ffn_cluster.py `
  --network-gbps 10 `
  --out results/phone_cluster_budget.json
```

Run the full test suite:

```powershell
python -m unittest discover -s tests -p "test*.py" -v
python -m compileall -q scripts src tests
```

## ComfyUI Deployment Plan

The target deployment is a **custom model backend**, not a pile of RPC nodes:

```text
ComfyUI model wrapper
    -> TeaCache decides real/reuse step
    -> broadcast descriptors to wired tablet workers
    -> workers produce FFN base tiles
    -> CUDA computes residual tiles
    -> merge gate/up
    -> original SwiGLU, down, attention and decode path
```

The first hardware milestone is a Windows host with an RTX 4070-class GPU and one or more USB/10GbE-connected tablets. Phones come after the wired tablet path is measured. Any worker deadline miss falls back to exact center-GPU execution.

### Complete FFN Offload Baseline

The repository now includes a separate baseline in which a worker stores a complete FFN layer, computes `gate -> SiLU -> up -> down` locally, and returns only the hidden output. For Qwen3.5 2B batch-1 decode this is about 8 KiB of fp16 activation roundtrip per layer, far smaller than repeatedly moving the layer weights. The tradeoff is different from residual split: layer dependencies make the 24 round trips serial, so worker compute and per-layer latency matter more than link bandwidth. For long packed video sequences such as MiniMax H3, the activation boundary is much larger and must be measured separately.

## Repository Map

- `docs/math_principles_report.md`: full mathematical report and Qwen3.8-27B-class budget.
- `docs/weight_code_split_spec.md`: split/merge contract and runtime restrictions.
- `docs/comfyui_phone_cluster_design.md`: MiniMax H3 and ComfyUI worker architecture.
- `docs/release/v0.1.0-rc2.md`: release-ready algorithm explanation and claims.
- `docs/release/ffn_residual_mesh_math_note_bingqin_wang.md`: short mathematical derivation signed by Bingqin WANG, suitable for a GitHub Release attachment.
- `scripts/simulate_comfyui_phone_ffn.py`: H3/ComfyUI phone-link simulation.
- `scripts/simulate_phone_ffn_cluster.py`: distributed FFN base-worker budget model.
- `src/phone_ffn_loopback.py`: framed worker protocol, checksum, concurrency, deadline, fallback.
- `src/full_ffn_loopback.py`: complete FFN layer worker and direct numerical equivalence check.
- `tests/`: correctness and protocol tests.
- `docs/log/`: iteration records with assumptions and results.
- `scripts/simulate_full_ffn_phone_offload.py`: complete-FFN activation-boundary and layer-dependency budget model.

## Honest Boundary

This repository does **not** claim that a 27B model is already running end-to-end on an 8 GiB GPU, or that a phone cluster automatically accelerates every workload.

The current evidence is layered and explicit:

- Mathematics: tested for the selected exact integer split.
- CUDA: tested at layer level, not full-model throughput.
- ComfyUI/H3: analytical simulation and integration design.
- Android/tablet: protocol loopback only; real device throughput is the next milestone.

That boundary is part of the design. A result that cannot beat the GPU branch on real bytes, wait time, and overlap is not promoted as a default backend.

## Contributing Hardware

If you have an old Android phone, tablet, mini PC, or wired network adapter, useful contributions are:

- device model and RAM;
- sustained worker memory bandwidth;
- USB/10GbE payload throughput;
- tile latency and deadline miss rate;
- thermal behavior over a long run.

The project is intentionally structured so a contributor can measure a worker without owning the full model or changing the ComfyUI workflow.

## License

MIT. See [LICENSE](LICENSE).
