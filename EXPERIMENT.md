# MP32 W4A4 decode: compatible numerical mode and opt-in optimization

The existing decoder remains the default. This change adds a numerical mode
compatible with the packed SGLang W4A4 routed path and a separately selectable
optimization. It applies to decode only; the prefill layout is unchanged.

The caller supplies prepared logical FP4 weights from
`transform_weights_for_fp4_fp4_mega_moe` and a live symmetric buffer created with
`enable_decode=True`. Given BF16 `hidden[local_rows, hidden]`, int32/int64
`topk_ids[local_rows, topk]` and FP32 `topk_weights`:

```python
sym_buffer.prepare_decode_input(hidden, topk_ids, topk_weights)
fp4_fp4_mega_moe_decode(
    output,
    w1,
    w2,
    sym_buffer,
    sglang_numerics=True,
    line_padded_scales=True,
    pack_b=True,
    double_buffered=True,
)
```

Warm up the input producer, both expert stages and all weight preparations
before graph capture. Finish Stage2 before overwriting the shared input/pools.
All ranks must use the same call order, equal active row count, and common
legal MP count. Decode supports 1..128 rows; the main performance validation
uses EP4, 24 rows/rank, H5120/I2304/E384/TopK6.

## Numerical modes

With `sglang_numerics=False`, gate/up BF16 rounding, BST4 activation rounding,
FP4 quantization, and final weighted combine retain the original MATE contract.
The same optimized data/compute path can be used with that contract.

With `sglang_numerics=True`, FC1 rounds gate/up to BF16, computes FP32 exp2
SwiGLU, applies the routing factor before E2M1/UE8M0 quantization, and FC2 sends
BF16 route outputs. Combine rounds each EP owner's partial to BF16 before
rank-ordered FP32 reduction and final BF16 rounding. This matches the packed
SGLang routed path's boundaries. Never mix Stage1 and Stage2 numerical modes.
The public two-stage entry prebuilds both specializations before any IPC launch.

## Optimized path

- Input preparation fuses BF16-to-FP4 quantization, row-major scale generation,
  int32-to-int64 routing IDs, and weight copies into symmetric input storage.
- `line_padded_scales=True` gives each logical M32 activation-scale block an
  independent 128-byte cache line. Consumers retain the row-arrival acquires
  and descriptor barriers; the invocation cache reset remains. Once a block
  is published, another block cannot partially modify its prefetched line.
  Only the repeated per-feature-chunk cache reset is removed.
- `pack_b=True` caches K256-contiguous weights, retains the online N permutation
  and register epilogues, and uses an independent weight-B TME descriptor.
- `double_buffered=True` uses an M32/N256/K256 tile, two TME stages, and a
  three-CTA launch-bounds target. Runtime occupancy selects 3, 2 or 1 CTA/MP;
  the cooperative grid never exceeds the queried resident capacity.

The line-padded SF scratch adds four times the compact SF-pool byte capacity
without replacing the existing prefill scratch. Packed-weight caches retain
source tensors and add packed storage per weight pair/backend. In-place weight
updates invalidate the prepared cache for future calls; recapture graphs after
changing weights. Account for these allocations during model integration.

## Validation and boundaries

```sh
python -m pytest tests/test_mega_moe_fp4_fp4_contract.py tests/test_mega_moe_fp4_fp4_decode_contract.py tests/test_mega_moe_fp4_fp4_decode_w4a4.py::test_owner_partial_rounding_is_part_of_the_contract -q
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/test_mega_moe_fp4_fp4_decode_w4a4.py
```

The independent CPU reference and queued graph regression cover balanced,
skewed, disabled and changed inputs/routes/weights. Separate same-process
benchmarks compare the original contract, its optimized counterpart, the
SGLang-compatible baseline, the compatible optimized path, and the frozen
SGLang implementation. Full output is checked after every timing round.

Performance artifacts, raw rank samples, seed hashes, launch parameters and
rejected experiments are maintained outside this source tree. Both single-call
and amortized graph timing are distinguished. No full-model TPOT or checkpoint
accuracy claim follows from these routed-operator measurements; router, shared
experts and mHC are outside the common measured boundary.
