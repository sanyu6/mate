# Shared GEMM1 epilogue fusion

`Fp4DispatchSharedSwiGluMhc` extends the prepared-input horizontal operator with
SwiGLU and MXFP8 quantization in the shared GEMM1 epilogue. It returns dispatched
routed inputs, `(shared_fp8, shared_scale)`, and mHC coefficients. The FP8 pair
feeds existing shared GEMM2 directly. No routed GEMMs, shared GEMM2, mHC post,
or full-model execution is performed by this Op.

The experimental contract is b24/H5120/shared intermediate2304. FP4 dispatch
and mHC input/scratch requirements match `Fp4DispatchSharedMhc`. Shared output
is E4M3 [24,2304], with UE8M0 [24,72] MN-major scale. Both output buffers are
allocated by the caller. There is no global [24,4608] BF16 gate/up buffer.

Call `prepare_shared_swiglu_weight(weight, scale, feature_tile=128)` once for
immutable weights before capture. It permutes the E4M3 weight rows and their
MN-major UE8M0 scales without requantization. Each feature tile contains a gate
half followed by matching up rows: feature128 stores gate64/up64. The packed
object records its tile size to prevent use with mismatched GEMM geometry.

The epilogue rounds accumulator outputs to BF16, applies the existing gate
upper clamp/up symmetric clamp, computes SiLU*up, rounds that result to BF16,
then quantizes32-value groups with the existing scale-code and E4M3 rounding.
The accepted tile-local scratch implementation preserves both BF16 numerical
boundaries without materializing the intermediate tensor in global memory.
Consumer-only barriers protect scratch reuse while dispatch and TME producers
remain independent. Experimental register/amax-only variants are retained as
explicit low-level JIT tactics; they are not selected by the public Op.

The quantizer reference is compiled with fast-math. Division in this epilogue
uses explicit `__fdividef` to match that quantizer's operation selection without
changing compilation flags or arithmetic of the independent mHC branch.

Original159e796 geometry:24 dispatch CTAs,up to36 shared GEMM CTAs (bounded by common MP
count),10 mHC CTAs,feature128/token32/K128,two pipeline stages. The dispatch CTA
count must match across ranks. Each call consumes one dispatch epoch. Dispatch
state and mHC scratch must be private to the current stream; completion starts
at zero. Warm up the selected geometry before capture. A smaller common MP count
is legal but has no performance claim from the recorded four-GPU measurements.

`swiglu_limit=10.0` is the model-aligned default; negative values disable clamp.
Tests cover unclamped,zero-clamp,clamped,zero/small/large input and changing routes.
Correctness compares FP8 and scale bytes to the previous native shared GEMM1 plus
existing standalone SwiGLU/quantizer,and compares the resulting shared GEMM2
output exactly. This does not establish full-model accuracy or equality with a
different GEMM1 accumulation algorithm.

## Owner-local dispatch queue

The updated opt-in default uses36 dispatch/shared CTAs and20 independent mHC
CTAs on the documented b24/H5120 workload (CTA counts are bounded by common
MPs). There are36 paired feature128 GEMM tiles and20 projection K partitions,
so the selected geometry assigns one of each per corresponding CTA. This is a
shape-specific measured default,not a claim of universal optimality or fixed
CTA-to-MP placement. `dispatch_schedule="legacy"` retains the global-route
round-robin scheduler; pass dispatch_ctas=24,mhc_ctas=10 to reproduce the previous
allocation. Explicit counts remain available.

The new scheduler constructs a stable list of routes owned by this rank from
the already gathered routing IDs,then distributes that list over dispatch warps.
It does not deduplicate tokens or change payload,scales,counts,offsets,metadata,
route weighting,ready/done epochs,or source lifetime. The all-owned case uses
its identity route order. The shared routing workspace is specialized to the
public b24 contract; standalone dispatch retains its old shape support unless
its low-level compact option is explicitly selected.

Tactic0 is legacy,tactic1 restricts the input fence to dispatch participants,
tactic2 is the accepted owner queue,tactic3 is an experimental dedicated-CTA
layout,and tactic4 additionally precomputes slots using warp matching. Dedicated
CTAs and precomputed slots did not improve the combined target cases and are
not selected by the public Op. Geometry/resource/work-distribution results are
kept with the experiment report.
