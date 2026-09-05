# FFN Weight-Code Split Mathematical Specification

Date: 2026-09-03

## 1. Scope

This document defines the mathematical object to optimize before any page-transfer or kernel benchmark.

The object is one `gate` or `up` projection of an existing Q4_K model. The split is compiled from fixed weights at cold start. It is not an activation VQ, an input-state lookup, or a model retraining procedure. The exact code split below is a mathematical reference: it does **not** qualify as a runtime main-term algorithm if evaluating its high/base term still requires scanning a dense per-weight code stream once per token.

The required properties are:

1. The full split reconstructs the Q4_K integer-domain projection exactly.
2. Each layer, output tile, and 32-value quantization group may use its own formula table.
3. Approximation, if enabled, is isolated to a declared residual representation or declared input-state remainder and has a computable local error bound.
4. `g` and `u` are merged before the original SiLU and down projection. No invalid addition across SwiGLU is allowed.

### 1.1 Resident artifact v1 boundary

The first Qwen3.8-27B artifact implements only exact `Q4_K` gate/up
projections. `IQ*`, `Q3_K`, `Q5_K`, and `Q6_K` tensors remain in the original
GGUF as exact fallback until each decoder has its own independent round-trip
test. This is a format boundary, not an approximation claim.

The v1 decomposition uses one integer base per 32-code group:

```text
q[i] = c + r[i],  r[i] in [-8, 7]
```

The base is selected from the group min/max so the signed 4-bit residual
never clips. The GPU-resident stream contains packed `r` plus `a`; the CPU
host artifact contains the pre-expanded coefficient `a*c + beta`. `beta` and
the original base descriptors are retained for verification and future
repacking. No runtime lookup table is required.

The byte ledger must report these separately:

```text
resident_weight_bytes          # one-time VRAM upload
host_base_coefficient_bytes    # static system-memory artifact
dynamic_h2d_bytes_per_token    # activation + base-result transfers
residual_weight_h2d_bytes      # must be zero after residency
fallback_bytes                 # original-format exact fallback
```

For a measured layer, a lower dynamic byte count is not by itself a speed
claim. Kernel span, exposed CPU submission gap, CPU base time, and a separate
raw-Q4 weight-stream baseline are required.

## 2. Q4_K Group Equation

For output row `j`, Q4_K group `g`, and element `i` in that 32-value group, the decoded weight is:

```text
w[j,g,i] = a[j,g] * q[j,g,i] - c[j,g]
```

where:

- `q` is the original unsigned 4-bit code in `[0, 15]`;
- `a = d * scale`;
- `c = dmin * min`;
- `a` and `c` are fixed values extracted during cold start.

For an integer activation representation `x[g,i] = sx[g] * z[g,i]`, define:

```text
S[g] = sum_i z[g,i]
P[j,g] = sum_i z[g,i] * q[j,g,i]
```

The projection is exactly:

```text
v[j] = sum_g sx[g] * (a[j,g] * P[j,g] - c[j,g] * S[g])
```

This equation already separates the Q4_K offset correction. Any later code split must preserve both `P` and the `c*S` term.

`INT8` activation quantization only changes `x` to `sx*z`; it is a separate, measurable error source. With a supplied integer `z`, the equations below are exact.

## 3. Primary Candidate: Centered Block Split

The first formula to analyze is not an activation lookup. It is an exact centered decomposition of a shared block base.

For one 32-value group, choose scalar bases `a` and `c` shared by all positions:

```text
x[i] = a + b[i]
q[i] = c + d[i]
```

The exact integer dot product is:

```text
sum_i x[i]*q[i]
  = 32*a*c + a*D + c*B + sum_i b[i]*d[i]

B = sum_i b[i]
D = sum_i d[i]
```

`D` is fixed and can be precomputed during cold start. `B` is one runtime group sum. Thus the first three terms are scalar or precomputed aggregate terms; only `b dot d` remains a 32-position fine residual dot product.

If rational/fixed-point means are retained exactly:

```text
a = mean_i x[i]
c = mean_i q[i]
```

then `B = D = 0` and the formula simplifies without approximation to:

```text
sum_i x[i]*q[i] = 32*a*c + sum_i b[i]*d[i]
```

This is the rigorous form of "keep the large base, compute only the jagged remainder." The cross terms have not been discarded: their group sums are zero by construction. With integer bases such as `floor(sum(x)/32)`, the two explicit aggregate corrections above retain exactness.

This requires `a` and `c` to be block-shared bases. Splitting each element into unrelated high and low digits does not make the three large terms cheap, because it leaves a dense multiplication in every term.

For Q4_K, substitute the centered code dot product into the original group equation:

```text
v[j,g] = sx[g] * {
    a_q4k[j,g] * (32*a_x[g]*c_q[j,g]
                   + a_x[g]*D[j,g]
                   + c_q[j,g]*B[g]
                   + sum_i b_x[g,i]*d_q[j,g,i])
    - c_q4k[j,g] * (32*a_x[g] + B[g])
}
```

Here `a_q4k` and `c_q4k` are Q4_K scale and offset, while `a_x`, `c_q`, `b_x`, and `d_q` are the centered activation/code components. The notation keeps the two unrelated meanings of "base" explicit.

If the fine code residual is replaced by `d_hat`, the only code-split error is:

```text
epsilon[j,g] = sx[g] * a_q4k[j,g] * b_x[g]^T * (d_q[j,g] - d_hat[j,g])
```

and therefore:

```text
|epsilon[j,g]| <= |sx[g] * a_q4k[j,g]|
                   * ||b_x[g]||_2
                   * ||d_q[j,g] - d_hat[j,g]||_2
```

This is a tighter selection metric than merely minimizing code-value error: it directly weights a residual by the actual within-block activation variation that will multiply it.

## 4. General Per-Partition Formula Table

For each `(layer, projection, output_tile, group)` choose a fixed integer table `C` and fixed code streams `h` and `r` such that:

```text
q[i] = C[h[i]] + r[i]
```

The table is allowed to be irregular. `C` need not be an affine function, and different groups may use different table sizes and values.

Define the static masks `I[k] = { i | h[i] = k }`. At runtime:

```text
T[k] = sum_{i in I[k]} z[i]
R    = sum_i z[i] * r[i]

P = sum_k C[k] * T[k] + R
```

Substituting `P` into the Q4_K group equation gives the exact merge rule:

```text
v[j] = sum_g sx[g] * {
    a[j,g] * (sum_k C[j,g,k] * T[j,g,k] + R[j,g])
    - c[j,g] * S[g]
}
```

The high/base term is the table-weighted aggregate `sum C*T`; the jagged term is `R`. The runtime does not select `C` from the activation. `h`, `r`, masks, and `C` are all fixed artifacts produced from weights at cold start.

## 5. Important Special Cases

### 5.1 Centered radix split

Choose a group base `b`, radix `B`, and a signed centered residual:

```text
q[i] = b + B*h[i] + r[i]
```

with `h[i] = round((q[i] - b) / B)` and `r[i] = q[i] - b - B*h[i]`.

Then:

```text
P = b*S + B*H + R
H = sum_i z[i]*h[i]
R = sum_i z[i]*r[i]
```

and:

```text
v[j] = sum_g sx[g] * {
    a[j,g] * (b[j,g]*S[g] + B*H[j,g] + R[j,g])
    - c[j,g]*S[g]
}
```

Compared with an unsigned low digit, centered `r` minimizes its maximum magnitude for a fixed radix. It is the direct generalization of splitting a long number into a stable leading portion and a small signed remainder.

### 5.2 Non-uniform weight-value table

Choose `K` centers `C[0..K-1]` from integer values, then assign each fixed `q[i]` to a center and retain its signed residual. This is more general than radix digits:

```text
q[i] = C[h[i]] + r[i]
```

For example, centers may be unevenly placed where the actual Q4 codes concentrate. This is a code table for fixed weights, not an activation codebook. It is appropriate when a group has a non-uniform code distribution and a radix split wastes residual range.

### 5.3 Hierarchical block base

The same idea can be nested over the 32 input positions:

```text
q[i] = b32 + b8[subblock(i)] + C[h[i]] + r[i]
```

The projection becomes:

```text
P = b32*S32 + sum_t b8[t]*S8[t] + sum_k C[k]*T[k] + R
```

All terms are exact when `R` is retained. This is the intended "bulky base plus jagged residual" form: coarse terms use a few aggregate activation sums; only the last term needs fine-grained work.

## 6. Low-Rank Generalization

The centered split is rank one with the constant vector as its basis. A per-block formula may use a small orthonormal basis `U` instead:

```text
x = U*alpha + b
q = U*gamma + d
U^T*b = 0
U^T*d = 0
```

Then the cross terms vanish exactly:

```text
x^T*q = alpha^T*gamma + b^T*d
```

This is the general mathematical condition behind the user's expansion: coarse components must occupy a small shared subspace, while the detailed residual is orthogonal to that subspace. The rank-one mean split is the cheapest first case; a layer/block-specific `U` is a later candidate if it materially reduces the energy of `b dot d`.

### 6.1 Shared scalar base is a special case, not a universal assumption

For activation and weight values with separate quantization scales:

```text
x[i] = sx * (s + b[i])
w[i] = sw * (s + d[i])
```

the same normalized scalar center `s` gives:

```text
x^T*w = sx*sw * (n*s*s + s*sum(b+d) + b^T*d)
```

If both residual sums are zero, only `n*s*s` and `b^T*d` remain. If the activation and weight means differ, forcing the same `s` does not create a contradiction: `s*(sum(b)+sum(d))` is one cheap aggregate correction. However, it can increase the fine residual energy. The cold-start compiler must therefore enumerate shared centers in the common normalized code domain instead of assuming that the two independent means are equal.

The more robust target is a shared rank-`r` basis `U`; equal scalar bases correspond to `r=1` and `U` equal to the constant vector. The basis, not literal equality of raw floating-point values, is the reusable regularity that can be shared between activation and weight blocks.

## 7. Approximation Contract

Let `r_hat` be the stored or computed approximation to `r`. The pre-activation error for a group is:

```text
e[j,g] = sx[g] * a[j,g] * sum_i z[g,i] * (r[j,g,i] - r_hat[j,g,i])
```

Therefore a cheap safe bound is:

```text
|e[j,g]| <= |sx[g] * a[j,g]| * ||z[g]||_1 * max_i |r[j,g,i] - r_hat[j,g,i]|
```

Group bounds add to a bound for `delta_g` or `delta_u`. The exact FFN difference is always evaluated as:

```text
delta_h = SiLU(g + delta_g) * (u + delta_u) - SiLU(g) * u
delta_y = Wd * delta_h
```

For routing, a conservative norm bound is:

```text
||delta_y||_2 <= ||Wd||_2 * (
    M_silu * ||delta_u||_2
    + (||u||_inf + ||delta_u||_inf) * L_silu * ||delta_g||_2
)
```

where `M_silu` and `L_silu` are bounds for SiLU and its derivative over the observed gate interval. A token whose bound exceeds the configured budget computes a fuller residual or takes the exact FFN path.

## 8. Cold-Start Selection Problem

There is no required universal formula. For each weight group, enumerate a small family of tables:

```text
K in {2, 4, 8}
C values in [0, 15]
radix B in {2, 4, 8}
block layout in {32, 8+8+8+8, 16+16}
residual format in {exact signed, 2-bit, 1-bit, omitted}
```

For a full residual, every candidate is exact in the Q4_K integer domain. Choose a candidate for later approximation by minimizing:

```text
J = E_z[(a * z^T * (r - r_hat))^2]
    + lambda_gpu * gpu_static_bytes(h, C)
    + lambda_cpu * cpu_runtime_bytes(r, a, c)
    + lambda_ops * arithmetic_ops
```

The expectation uses captured activation statistics for this layer and group. It does not train a predictor for unseen outputs. Its purpose is to decide which fixed arithmetic decomposition gives the cheapest residual under the permitted error budget.

With a covariance matrix `Sigma_z` for the 32 activation integers, the residual contribution is computed without sampling every candidate token:

```text
E[(z^T * delta_r)^2] = delta_r^T * Sigma_z * delta_r
```

This permits exhaustive per-group fitting over a finite table family before any kernel work.

For the centered split, the primary residual loss term is:

```text
E_z[(b^T * (d - d_hat))^2]
```

which is evaluated against within-block activation residuals `b`, not raw `x`. This makes the cold-start fitting target explicitly match the only dense term that remains after exact aggregation.

## 9. Formula-Table Artifact

Each compiled partition records:

```text
layer_id
projection                 # gate or up
output_tile
input_group
formula_kind               # centered_radix, value_table, hierarchical
q4k_scale_a, q4k_offset_c
base_values                # b / b32 / b8 / C
high_code_format           # bit width and packing of h
residual_format            # exact or declared lossy encoding of r
merge_scales_and_carry
execution_owner            # GPU base, CPU residual, or fallback
error_bound_parameters
```

This is the table requested by the project: it specifies what is calculated, how the fixed weight data are transformed at cold start, and how partial results are merged. It does not need to be common across layers.

## 10. Cold-Start / Runtime Access Contract

The access boundary is explicit:

```text
cold start: raw Q4_K -> full scan -> compile artifact in host RAM
runtime:    artifact only -> CPU base + GPU residual -> merge
fallback:   optional raw Q4_K access, counted separately
```

The cold-start compiler may read every original weight, decode every Q4_K block, fit formula tables, and materialize host-side arrays. Runtime correctness must be tested with the raw GGUF mapping unavailable; any accidental raw-weight access is a contract violation. This distinction is independent of whether the host evaluator itself uses a dense artifact or a structured aggregate formula.

For the stricter target in this project, merely replacing raw GGUF reads with a sequential scan of an expanded high-code artifact is also insufficient. A viable runtime base must be a compiled circuit or a partial-sum/formula table whose evaluation does not enumerate the original per-weight code stream. The exact high/low bit split remains a reference identity and artifact format, not proof that this stronger condition is satisfied.

## 11. Decision Before Systems Work

Do not measure PCIe paging or throughput until this specification yields at least one group/table family satisfying all three conditions:

1. Exact mode reconstructs Q4_K `gate/up` projections to numerical roundoff with the complete residual retained.
2. Any optional lossy residual mode meets its separately declared analytical and held-out FFN error budget.
3. The selected high/base representation has lower GPU-resident bytes than original Q4_K, while its complete residual representation has a plausible GPU transport and arithmetic form.

## 12. Error Provenance and Runtime Boundary

The compiler may scan the complete Q4_K matrix during cold start to produce a main-term formula, partial-sum table, or finite-state artifact. That scan is not an approximation and contributes no inference error. With the complete weight residual retained, the weight-code split is exact apart from normal floating-point accumulation order.

Runtime error can arise only if a declared term is made inexact:

```text
weight residual clipping:       R -> R_hat
input-state discretization:     x = x_state + epsilon_x
table/parameter quantization:   H -> H_hat
finite-precision accumulation:  arithmetic roundoff
```

For a compiled main term `H`, input-state discretization contributes:

```text
H*x = H*x_state + H*epsilon_x
```

`H*epsilon_x` is an activation remainder, not a weight-residual error. It must either be retained as a separately computed correction, bounded and accepted as approximation, or trigger fallback. After gate/up projections, their individual errors can be amplified by the original SwiGLU product, so reports must keep pre-activation and final FFN error separate.

Any candidate that evaluates `H*x_state` by scanning a dense `H` stream at runtime fails the runtime contract even if it is algebraically exact. It must instead access a bounded number of compiled entries or formula parameters per input block.
