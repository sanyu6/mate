# Experimental dispatch + shared GEMM1 + independent mHC

`mate.ops.moe.Fp4DispatchSharedMhc` horizontally fuses two independent CTA groups
in one MP32 kernel. The first group executes the existing FP4 dispatch and full
local MXFP8 shared GEMM1. The second executes mHC projection, RMS statistics,
and 20 Sinkhorn iterations. No routed GEMM, shared GEMM2, residual-post, or final
MoE combine is included.

This prototype explicitly supports local rows=24, hidden=5120, residual factor=4,
FP32 mHC weight [24,20480], and rms_eps=hc_eps=1e-6. Residual storage is padded
BF16 [32,20480]. Three prepared BF16 weight components preserve the FP32-weight
projection; preparation is outside the measured operator. The FP4 dispatch and
MXFP8 shared input contracts are identical to `Fp4DispatchSharedGemm`.

`num_ctas` and `gemm_ctas` control the dispatch/shared group. `mhc_ctas` controls
the independent group, which strides over 20 projection K partitions. Values
above 20 create idle mHC CTAs. Projection outputs are stored in caller-owned
FP32 `partial_mix [20,24,24]` and `partial_sq [20,24,3]`. The last publishing mHC
CTA computes all coefficients; it never waits for the dispatch group. Its
private int32 `completion [1]` starts at zero and is reset after completion.
The output backing tensor has 576 FP32 elements: pre [24,4], post [24,4], and
comb [24,4,4] in that order.

Scratch must not alias other tensors or be shared by overlapping invocations.
Use one dispatch buffer and scratch set per concurrent stream. Sequential eager
and graph calls reuse them. All ranks must use matching dispatch CTA counts.
Warm up the selected configuration before capture. The serial backend executes
these same computations as two kernel launches, using 256 threads for standalone
mHC. The fused kernel uses the shared-GEMM group's block size for every CTA.

The generated mHC projection was adapted to native MP32 TME stride loads and
MP32 BF16 SQMMA, including MP32 accumulator coordinates. It retains the original
three-term accumulation and sum-of-squares layout. It is a fixed-shape research
implementation, not a general mHC backend or an automatically selected default.

Validation in the accompanying experiment includes FP64 CPU coefficients,
byte-exact dispatch checks, serial/fused bitwise shared and coefficient outputs,
changed-input graph replay with skew and disabled routes, and direct graph-node
inspection proving one kernel launch. Timings exclude input preparation, weight
preparation, and API validation overhead by invoking the cached launcher directly.

## Optimized coefficient tactic

The opt-in Op now defaults to `mhc_tactic="optimized"`, dispatch/shared24 CTAs
and mHC20 CTAs. `mhc_tactic="baseline"` preserves the original coefficient
implementation for A/B. The optimized tactic exposes24 logical weight rows to
TME (the prepared32-row stride is unchanged), redundantly computes the same
RN reciprocal in coefficient lanes instead of broadcasting a selected lane's
reciprocal, and publishes projection results with one system fence per CTA
leader. All store threads rendezvous before this fence and before the completion
counter is incremented. Consumer cache invalidation, completion retirement,
20 Sinkhorn iterations and three BF16 weight terms are retained.

The source mHC P512 tactic accelerates residual-post work. This operator takes
an already prepared residual, so no P512 residual-post work was added. Its tested
N24 and redundant-reciprocal ideas transfer directly. Full lagged boundary time
is not comparable with this operator's independent coefficient time.

Low-level JIT research controls retain tactic0 (baseline), tactic1 (N24/RN),
tactic2 (two tokens/warp; slower), and tactic3 (accepted CTA-leader publication).
`k_parts=40/80` variants were validated but slower for this horizontal layout;
the public Op keeps20 partitions and does not select these research options.
