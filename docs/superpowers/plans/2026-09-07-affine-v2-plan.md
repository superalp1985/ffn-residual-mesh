# Iteration 67: Versioned Q5_K Resident Residuals

Continue the approved resident-residual plan, task 5. Do not change the
mathematical split or assume that signed residuals compress the source.

## Research

Checked on 2026-09-07:

- llama.cpp `dequantize_row_q5_K`, fetched from upstream master and compared
  with the vendored implementation. Q5_K uses the same scale/min groups as
  Q4_K, plus a separate high-bit plane.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c
- The installed gguf-py dequantizer remains the independent numerical oracle.
- Web search returned no usable content in this session; upstream raw source
  was retrieved directly. No claim about a newer release or fastest kernel.

## Design

For each group choose `b = ceil((min(q) + max(q)) / 2)`, write `r = q - b`.
Q5_K gives `r in [-16, 15]`. CPU stores and multiplies `alpha*b + beta`;
GPU multiplies `alpha*r`. Merge gate/up before SwiGLU.

V1 Q4 reading and compilation remain supported unchanged. V2 carries
per-projection source type, residual bitwidth, and packing descriptor.
Each Q5 group of 32 values uses 16 adjacent-pair nibble bytes followed by
4 high-bit bytes (two's-complement five-bit residuals, LSB-first bit order).
No clipping, requantization, or runtime lookup table.

Only the single-flight ResidentGateUp path gains Q5 support this round.
Q4-only tiled/graph paths must reject incompatible packages explicitly.

## Checklist

- [x] Finish iteration 66 regression and current-factory numerical verification.
- [x] Research upstream quantization layout before implementing.
- [x] CPU artifact tests: Q5/mixed roundtrip, full range, metadata checks.
- [x] V2 compiler/reader and independent reconstruction tests.
- [x] GPU tests: Q5/mixed grouped residual + CPU base + SwiGLU.
- [x] Grouped GPU high-plane decoding, V1 compatibility, unsupported-path guards.
- [x] Complete fixture FFN and real layers 24/25 with original Q5 down.
- [x] Full regression, byte/timing report, commit and push.

Measure current latency rather than copying iteration 66's earlier results.
The fresh iteration 66 run showed large CPU/submission gaps, not a repeat of
the previously recorded approximately 1 ms wall time.
