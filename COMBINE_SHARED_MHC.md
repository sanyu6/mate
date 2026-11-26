# End-side fusion prototype contract

Baseline159e796, current prototype b24/H5120/I2304/EP4/topk6.
Input routed output is BF16 [pool_rows,5120] in the exact M32 expert-pool row order of dispatch. Metadata int32 [pool_rows,4] is source_rank/token/topk/float_weight_bits; counts and offsets select only valid rows. Padding is not consumed. Each CTA's communication squad builds an invocation-local inverse route table from those valid rows. `weighted=true` applies metadata weights; `false` means caller already applied routing weights. The latter must be selected when the routed GEMM path preweights inputs.

Numerical order: per-rank topk-ordered FP32 FMA; routed_scaling_factor; BF16 local partial; rank0..R-1 FP32 sum; BF16 routed result; owner-only shared BF16 add and BF16 rounding; existing mHC post FMA/BF16; HC collapse/BF16; existing RMSNorm/BF16; existing MXFP8 quantizer.

Communication uses a dedicated symmetric buffer (not the begin-side dispatch buffer). Control is at kCombine, payload at kCombine+65536. Every CTA owns the same hidden tiles on every rank, and publishes ready only after all squad writes and a system fence. Done handshake follows remote reads before the next invocation can reuse storage. No input-preparation/reorder kernel is excluded: inverse mapping is inside combine.

24 CTAs:128 communication threads,one active TME producer warp,and128 MMA threads. The complete384-thread CTA joins through cooperative grid.sync only in fused mode. Launch occupancy is checked and all24 CTAs must fit. Each CTA then owns one local token for shared add and the complete mHC post/combine/norm/quant tail. There is no unchecked grid-wide busy wait. Serial mode2 launches combine-only,shared-GEMM2-only,and dependent tail separately; mode1 combines the independent producers then launches tail. Mode0 is one cooperative launch.

The initial baseline is same-native-math serial control. Existing muDNN GEMM2 is a numerical diagnostic,not yet claimed bitwise equal to native GEMM2. The original SGLang mHC post/combine/norm/quant serves as an independent bit comparison oracle. Full-model execution is not covered.

Public entry: `mate.ops.moe.MoeCombineSharedMhc`. Output tensors/scratch are
caller-owned; shared_x/shared_scale are the begin-side FP8 activation pair.
Return value is `(updated_residual, normalized_bf16, (fp8, mn_major_scale))`.
The supplied status tensor is0 on success and1 on peer timeout; discard all
outputs when it is nonzero. Valid metadata keys must be unique per source
rank/token/topk slot. The routed pool capacity must match metadata capacity.
The operator does not include routed GEMM2 or mHC coefficient calculation;
those are required inputs. Changing a route distribution changes counts/offsets
and metadata but does not require a host synchronization or buffer reallocation.

## V2 candidate and real old-path comparison

V2 defaults to two_kernel:40 producer CTAs,512 communication threads per CTA,
128-thread producer squad and128 MMA threads (768 total),four GEMM stages,and
a dedicated two-epoch partial plane. The first kernel includes inverse-map
initialization,top-k combine,peer return,shared GEMM2,and owner-only shared add
into merged_out. The second invokes the installed SGLang MUSA post/combine/norm/
quant kernel with1024 threads andparts2. This backend requires the matching
SGLang kernels. Production model flags are unchanged.

The single-kernel fused backend remains a slower control. It adds shared and
reduced in post registers with the same BF16 rounding,leaving merged_out
untouched. Its tail uses producer dynamic shared after the cooperative join;
post_split is register-local and collapsed/quantized shared storage aliases.
Both backends retain shared_out/reduced_out global buffers; two_kernel also
retains merged_out write/read. No claim of eliminating those buffers is made.

Communication publication explicitly orders all writer-thread stores with
system_noflush before the named rendezvous,then the leader performs full system
fence before ready release. The receiver observes ready with acquire,its leader
invalidates cache,then all communication threads rendezvous with a compiler
memory clobber before normal packet loads. Named async arrival alone is not a
memory publication primitive. Producer completion and the single-kernel tail
handoff also retain their explicit publication/invalidation sequence.

Two-epoch reuse is valid only with a fixed CTA-owned packet partition and
protocol on a dedicated state. Do not alternate V1 single-plane and V2 ring2 on
the same state. The A/B harness uses separate V1/V2 symmetric buffers. Failed
pre-fix candidates (including optimistic73us measurements) are not accepted.

The actual old reference is SGLang custom combine/RS + muDNN shared GEMM2 +
original post boundary. Native GEMM2 can differ from muDNN by a few BF16 values;
V1/V2 retain exact equality,while old RS/post are checked exactly with common
native shared input. True muDNN-path differences remain separately reported.
