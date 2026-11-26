# Standalone FP4 dispatch

Experimental MP32 decode dispatch extracted from MegaMoE `dd01268dd45898f293b87f63905bab50b14b058c`. It has no expert weights, routed GEMM, TME producer, activation or combine. The complete dispatch invocation publishes route descriptors/counts, pulls peer FP4 payload/scales from symmetric IPC memory, packs local expert inputs, and retires its invocation counters.

```python
import mate
from mate.ops.moe import Fp4DispatchBuffer

# Set the local MUSA device and initialize the process group first.
buffer = Fp4DispatchBuffer(group, num_experts=384, hidden=5120, topk=6)
buffer.prepare_input(hidden_bf16, expert_ids, route_weights)  # separate input preparation
out = mate.ops.moe.Fp4Dispatch(buffer, tokens=24)

# Already-quantized callers can use this instead of prepare_input():
# buffer.prepare_fp4(payload_u8, row_major_scales_u8, expert_ids, route_weights)
```

Construct/warm up outside capture. The dispatch call itself allocates no tensors. It launches one 128-thread-per-CTA kernel, with one CTA per configured resident worker (original backend uses the common MP count). Buffer construction reuses the MegaMoE IPC allocator and conservatively sized M256 pool capacity; it retains an unused combine region for layout compatibility, but allocates no GEMM weights or compute intermediate pools. Removing this storage overhead is outside this initial extraction.

Every rank must provide the same active token count in 1..128 and invoke in the same collective order. Each token's enabled expert IDs must be unique; -1 disables a route. There is no deduplication: a token routed to six experts produces six expert-input rows. A status value of 1 reports invalid expert IDs other than -1, without leaving the collective protocol. It is a device tensor; inspect it after synchronization when validating inputs. Duplicate valid IDs violate the input contract and are not checked by this initial version.

Use a dedicated buffer and one ordered stream. No overlapping use or mixing with the fused Stage1/Stage2 APIs is supported. The final cross-rank barrier protects source input lifetime and counter retirement before the next preparation/call, including back-to-back graph nodes. It is part of dispatch-only timing. Persistent barrier phases are never reset.

Outputs are views owned by `buffer`; a subsequent dispatch overwrites them. `out.clone()` snapshots all outputs on the current stream, including in a graph. Padding rows/scale holes are undefined. Reset graph instances before `buffer.close()`, which collectively destroys IPC state.

| Output | Layout / meaning |
|---|---|
| `payload` | uint8 `[P,H/2]`, contiguous packed E2M1, expert-major compact M32 pools |
| `scales` | uint8 `[H/32,P]` with `line_padded_scales=False`; or `[H/32,4P]` with the default True |
| `counts` | int32 `[E/EP]`, valid row count per local expert |
| `offsets` | int32 `[E/EP+1]`, prefix sum of `ceil(count/32)*32`; last value is used pool rows |
| `metadata` | int32 `[P,4]`: source rank, local token, top-k slot, FP32 route-weight bits |
| `status` | int32 `[1]`, asynchronous input-ID status |

For line-padded scales, logical pool row `r` maps to physical scale row `(r//32)*128+r%32`. This matches the validated MegaMoE line-padded SFA layout. Within each expert, source ranks are grouped in ascending rank order; order within a source group is unspecified. Use metadata rather than assuming source-token order. These are MegaMoE compact pools, not the SGLang fixed `[expert,global_rows,...]` transport layout. An independent GEMM consumer must consume these explicit counts/offsets and metadata; the existing fused Stage1/Stage2 are not plug-in consumers after this dispatch retires its workspace.

Validation commands in the matching torch_musa/MUTLASS environment:

```sh
MATE_MUSA_ARCH_LIST=3.2 python -m pytest tests/test_fp4_dispatch_contract.py -q
python -m torch.distributed.run --standalone --nproc-per-node=1 tests/test_fp4_dispatch_gpu.py
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/test_fp4_dispatch_gpu.py --bench --output /out/dispatch-ep4.json
```

The GPU driver compares payload/scale bytes and metadata against an independent CPU routing oracle for b=1/24/128, balanced/skew/disabled/invalid IDs, negative routing weights, both scale layouts, and changing inputs in queued graph invocations. Its optional b24/H5120 timing captures eight complete dispatch calls per replay, flushes 256 MiB outside timing and reports max-rank samples. Input staging/quantization is excluded from that dispatch-only timing. No profiler is used.

This first version preserves the count/tag/descriptor/pull protocol and removes GEMM-specific arrival/consumer machinery. It does not claim the same performance as the dispatch role inside the fused MegaMoE kernel: launch grid, resource occupancy, absence of concurrent GEMM and the standalone completion boundary differ.

## Optimized peer-pull backend

The default (`override_backend=None`, or explicit `"mp32_pull"`) uses the
optimized protocol. `override_backend="mp32"` retains the original extracted
count/descriptor protocol. The optional `num_ctas` argument tunes only the pull
backend, must be an integer in 1..common_mps and must match across ranks. The
measured b24 default is 24 CTAs (bounded by common_mps); batches <=4 use four,
and batches >32 use the common grid. Other batch sizes have correctness coverage,
not an optimization claim.

The pull kernel reads peer route IDs into CTA-local shared memory, counts local
experts with shared atomics, computes M32 offsets with a warp prefix scan and
pulls each owning route's FP4/scales directly. Rank-major source ordering and
all output layouts/values are preserved. There are no global expert count
atomics, descriptor pushes, grid barriers or expert-counter retirement. Per-CTA
ready/done epochs replace them. Epoch counters and peer flags occupy the
previously unused combine region, with 128-byte separation; this buffer remains
dedicated to dispatch-only and cannot share that region with routed output.

System-scope release/acquire, the initial and final full system fences, and
all peer completion checks are retained. Input may be overwritten only after
all local CTAs have observed all peers' done phases. Unsigned epochs use a
signed modular-distance check, including uint32 wrap. Status `2` additionally
reports a peer-handshake timeout; output data is invalid then, and all ranks
must discard/recreate the buffer before proceeding. A timed-out rank is not
silently replaced by a different collective implementation.

Host-side device/occupancy checks are cached per thread/device after successful
validation, and native FFI launchers are cached; tensor shape/type/device checks
still execute on every launch. No GEMM/activation/combine is added.
