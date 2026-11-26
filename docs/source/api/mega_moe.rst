.. _apimega_moe:

Mega MoE
========

Mega MoE is MATE's distributed FP8 Mixture-of-Experts execution path for
multi-rank expert-parallel workloads. Use it when your integration already has
token activations, top-k expert indices, and top-k expert weights prepared and
you need MATE to run the expert compute path across a process group.

For integrations that already target DeepGEMM-style distributed MoE
execution, prefer the ``deep-gemm`` wrapper first. Use the direct MATE APIs
below when wrapper coverage is not enough.

The examples here use the ``mate.deep_gemm`` aliases. The same entrypoints are
also available from ``mate.mega_moe``.

Use Mega MoE when
-----------------

- you are integrating a distributed expert-parallel MoE path across multiple
  ranks
- your expert weights are stored as grouped FP8 tensors
- you need the prefill two-layer expert execution path, not just one grouped
  GEMM call

Use the :doc:`GEMM APIs <gemm>` instead when you only need grouped or masked
expert GEMM operators, including the ``0.2.4`` W4A8 mixed-dtype GEMM path.

Typical Flow
------------

Mega MoE uses a shared runtime workspace that holds FP8 inputs, routing
metadata, arrival masks, and intermediate activations for the serial split
``Stage 1 -> Stage 2 -> Stage 3`` execution path.

1. Allocate the symmetric runtime buffer with
   ``get_symm_buffer_for_mega_moe``.
2. Transform logical FP8 expert weights with
   ``transform_weights_for_mega_moe``.
3. Launch the kernel with ``fp8_fp8_mega_moe``.

Current Limits
--------------

- Intranode process groups only, with up to ``8`` ranks.
- ``num_experts`` must be divisible by the process-group size.
- ``activation="swiglu"`` only.
- ``use_fp8_dispatch=True`` only.
- ``hidden`` must be divisible by ``256``.
- ``intermediate_hidden`` must be divisible by ``128`` and no greater than
  ``8192``.
- The symmetric-buffer token capacity is rounded up to a multiple of ``256``.
- ``activation_clamp`` is not supported by the split-prefill path.
- ``fp8_fp8_mega_moe`` writes into a contiguous ``torch.bfloat16`` output
  tensor.

API
---

.. py:function:: mate.deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, use_fp8_dispatch=True, activation="swiglu", prefill_pool_capacity_ratio=1.0, prefill_l1_ring=True)

   Allocate the shared runtime workspace used by Mega MoE execution.

   ``prefill_pool_capacity_ratio`` may reduce the 256-row-aligned Prefill L1
   and L2 activation pools relative to their worst-case capacity. The default
   ``1.0`` is safe for every valid routing distribution. Values below ``1.0``
   are an unsafe capacity hint: if the actual sum of per-expert padded rows
   exceeds the configured pool, device memory may be overwritten.

   ``prefill_l1_ring`` is on by default and recycles the routed Prefill L1
   activation pool through an automatically sized ring. Routing metadata stays
   at the worst-case size and results are unchanged; pass ``False`` to fall back
   to the worst-case L1 pool.

.. py:function:: mate.deep_gemm.get_token_alignment_for_mega_moe()

   Return the split-prefill token-capacity alignment, currently ``256``.

.. py:function:: mate.deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights=None)

   Transform grouped FP8 expert weights into the layout expected by the Mega
   MoE runtime.

.. py:function:: mate.deep_gemm.fp8_fp8_mega_moe(y, l1_weights, l2_weights, sym_buffer, cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32), activation="swiglu", activation_clamp=None, fast_math=True)

   Run serial split-prefill Stage 1, Stage 2, and Stage 3 and write the result directly to
   ``y``.

MP32 FP8/FP4 prefill
-------------------

``mate.fp8_fp4_mega_moe`` runs three prefill kernels with 40 MPs by default.
Use ``mate.mega_moe_fp8_fp4.get_symm_buffer_for_fp8_fp4_mega_moe`` to allocate
full pools; input scales are UE8M0 bytes of shape ``[tokens, H/32]``.
Prepare packed E2M1 weights with
``mate.mega_moe_fp8_fp4.transform_weights_for_fp8_fp4_mega_moe``.
Logical weight scales have shape ``[E, N, K/32]`` (uint8), or
``[E, N, K/128]`` (int32 packing four UE8M0 bytes).
The helper interleaves W1 gate32/up32 and transposes both scale tensors.
W2 remains in logical intermediate order. Ring recycling is disabled.
Routing weights are applied to BF16 GEMM2 partials by the final FP32 reduce.
All ranks must call collectively with the same token count and configuration.

.. autofunction:: mate.fp8_fp4_mega_moe
