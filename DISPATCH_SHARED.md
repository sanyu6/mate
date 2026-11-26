# FP4 dispatch + full local shared-expert MXFP8 GEMM1

`mate.ops.moe.Fp4DispatchSharedGemm` is an opt-in experimental MP32 operator.
It runs the existing optimized standalone FP4 dispatch and a local unsharded
shared gate/up GEMM in one kernel. It does not perform a routed GEMM, token
 deduplication, SiLU, shared activation quantization, GEMM2 or combine.

```python
import torch
from mate.ops.moe import Fp4DispatchBuffer, Fp4DispatchSharedGemm

# Collectively construct once on the local MUSA device, before capture.
buffer = Fp4DispatchBuffer(group, num_experts=384, hidden=5120, topk=6)
# Common preparation, separate from this operator:
buffer.prepare_input(hidden_bf16, route_ids, route_weights)
# shared_x/shared_scale are already prepared E4M3/UE8M0 from that same hidden.
out = torch.empty((24,4608), device=shared_x.device, dtype=torch.bfloat16)
dispatched, gate_up = Fp4DispatchSharedGemm(
    buffer, shared_x, shared_scale,
    shared_weight, shared_weight_scale, out,
)
```

Shared input and full gate/up weight are contiguous E4M3 `[rows,H]` and
`[N,H]`. Their uint8 UE8M0 scales are `[rows,H/32]` and `[N,H/32]` in MN-major
storage (the transpose is contiguous). No scale conversion, quantization or
weight repacking occurs inside the Op. Output is caller-owned contiguous BF16
`[rows,N]`, with the same gate-half/up-half order as the input weight rows.
Default measured geometry is local rows24, H5120, N4608, feature128/token32/K128,
2 TME stages, 24 dispatch CTAs and at most 24 compute CTAs. `num_ctas` and
`gemm_ctas` must be within the common MP count; the dispatch CTA count must
match across ranks. `gemm_ctas` defaults to min(tile tasks,dispatch CTAs,common
MPs). Alternate tested tiles/stages are exposed for experiments, not automatic
autotuning.

One CTA contains a 128-thread dispatch squad, a producer squad (one active
warp) and native SS FP8 MMA consumers. The weight occupies the MMA feature/M
axis, and local tokens occupy its N axis. Dispatch barriers are scoped to its
four warps; they do not block the TME/MMA roles. A full input system fence
precedes both roles, and the original dispatch release/acquire epochs plus
output-publication fence are retained. Unused producer-squad warps remain idle.
The returned dispatch views have exactly the existing compact/M32 layout and
lifetime; padding is undefined. Use source metadata to interpret each expert
row. Both outputs must be consumed or snapshotted before the next overwriting
call. One ordered stream and dedicated buffer are required. Reset all graphs
referencing the buffer before collective close().

`override_backend="mp32_serial"` invokes standalone dispatch followed by the
same native MXFP8 GEMM1, for fusion-only numerical/performance control. The
same Op/Impl contract applies. The fused output must be bitwise identical to
this native serial control. Neither native path is promised bitwise identical
to muDNN's FP32 accumulation order: dtype, FP8 operands, UE8M0 scale semantics
and BF16 output are preserved, but rare BF16 rounding differences are measured
separately. Do not treat operator tests as model-accuracy acceptance.

The unchanged shared tail can consume gate_up:

```python
q, sf = mudnn_mxfp8_quant(
    gate_up, fuse_silu_and_mul=True, swiglu_limit=model_swiglu_limit,
)
shared = mudnn_mxfp8_linear(q, down_weight, down_weight_scale, input_scale=sf)
```

Device/occupancy checks are cached per successful thread/device. TME parameters
are cached by validated pointers, rows and device in a bounded 32-entry
thread-local cache; shape, dtype, stride and device checks execute every call.
The parameters depend on addresses/layout, not tensor contents. Changed input,
weight and scale contents at the same pointers are covered by queued-graph
regression. JIT imports and builds remain lazy; importing mate.ops does not
query hardware.

Validation in the matching vendor environment:

```sh
python -m pytest tests/test_dispatch_shared_contract.py tests/test_fp4_dispatch_contract.py -q
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/test_dispatch_shared_gpu.py
```

GPU tests compare dispatch bytes to an independent CPU routing oracle, fused
GEMM1 exactly to the same native serial GEMM, and native BF16 GEMM output to an
independent dequantized CPU FP32 matmul. They include zero/negative activations,
changed shared weights/scales, skew/disabled routes and repeated graphs, both
scale layouts, rows1/24 and H256/5120. MuDNN differences and performance are
recorded separately by the external task benchmark. This does not connect the
operator to the full SGLang model or convert its pools to the current fixed
expert/row W4A4 input layout.
