import argparse

import torch

import mate
from mate.mate_runtime import get_physical_num_mps
from mate.testing import quantize_mxfp4, quantize_mxfp8
from mate.testing.utils import bench_kineto, count_bytes


def pack_scale(scale: torch.Tensor, per_page: bool = False) -> torch.Tensor:
    packed = torch.zeros((*scale.shape[:-1], 4), dtype=torch.uint8, device="musa")
    packed[..., : scale.shape[-1]] = scale
    if per_page:
        packed = packed.reshape(scale.shape[0], -1, 4).transpose(1, 2).contiguous()
    else:
        packed = packed.reshape(-1, 4).transpose(0, 1).contiguous()
    return packed.view(torch.int32).reshape(scale.shape[:-1])


def benchmark(qkv_format, quantize, args: argparse.Namespace) -> None:
    pages_per_request = (args.context_len + args.page_size - 1) // args.page_size
    max_seqlen_kv = pages_per_request * args.page_size
    num_pages = args.batch_size * pages_per_request

    q, scale_q = quantize(
        torch.randn(
            (args.batch_size, args.next_n, args.num_heads, args.head_dim),
            dtype=torch.bfloat16,
            device="musa",
        )
    )
    kv, scale_kv = quantize(
        torch.randn(
            (num_pages, args.page_size, 1, args.head_dim),
            dtype=torch.bfloat16,
            device="musa",
        )
    )

    packed_head_dim = kv.shape[-1]
    fused_kv = torch.empty(
        (num_pages, args.page_size * (packed_head_dim + 4)),
        dtype=torch.uint8,
        device="musa",
    )
    fused_kv[:, : args.page_size * packed_head_dim] = kv.view(torch.uint8).reshape(
        num_pages, args.page_size * packed_head_dim
    )
    fused_kv[:, args.page_size * packed_head_dim :] = (
        pack_scale(scale_kv, per_page=True)
        .view(torch.uint8)
        .reshape(num_pages, args.page_size * 4)
    )
    fused_kv = fused_kv.view(num_pages, args.page_size, 1, packed_head_dim + 4)

    scale_q = pack_scale(scale_q)
    weights = torch.randn(
        (args.batch_size * args.next_n, args.num_heads),
        dtype=torch.float32,
        device="musa",
    )
    seqlens_kv = torch.full(
        (args.batch_size, args.next_n),
        args.context_len,
        dtype=torch.int32,
        device="musa",
    )
    page_table = torch.randperm(num_pages, dtype=torch.int32, device="musa").view(
        args.batch_size, pages_per_request
    )
    metadata = mate.deep_gemm.get_paged_mqa_logits_metadata(seqlens_kv, 512)

    def run() -> None:
        mate.deep_gemm.paged_mqa_logits(
            q,
            fused_kv,
            weights,
            seqlens_kv,
            page_table,
            metadata,
            max_seqlen_kv,
            scale_q=scale_q,
        )

    seconds = bench_kineto(
        run,
        "MP32PagedMQALogitsBlockScale",
        num_tests=args.num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )

    num_logits = args.batch_size * args.next_n * args.context_len
    flops = 2 * num_logits * args.num_heads * args.head_dim
    bytes_ = (
        count_bytes(q, scale_q, fused_kv, weights, seqlens_kv, page_table, metadata)
        + num_logits * torch.float32.itemsize
    )
    print(
        f"{qkv_format}: B={args.batch_size}, N={args.next_n}, "
        f"H={args.num_heads}, D={args.head_dim}, P={args.page_size}, "
        f"L={args.context_len}: "
        f"{seconds * 1e6:.2f} us, {flops / seconds / 1e12:.2f} TFLOP/s, "
        f"{bytes_ / seconds / 1e9:.2f} GB/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark paged block-scaled MQA logits."
    )
    parser.add_argument("--dtype", choices=("mxfp8", "mxfp4"), default="mxfp4")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--next-n", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=8192)
    parser.add_argument("--num-tests", type=int, default=30)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = get_physical_num_mps()

    torch.manual_seed(666)
    if args.dtype == "mxfp8":
        benchmark("MXFP8", quantize_mxfp8, args)
    else:
        benchmark("MXFP4", quantize_mxfp4, args)


if __name__ == "__main__":
    main()
