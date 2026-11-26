from __future__ import annotations

import os
import random

import pytest
import torch
import torch.nn.functional as F

from mate.arch import MP32_ARCH
from mate.gdn_prefill import chunk_gated_delta_rule
from mate.mate_runtime import get_arch
from mate.testing.operators.gdn import blockwise_gdn_prefill_reference


pytestmark = pytest.mark.usefixtures("disable_mudnn_tf32")

CHUNK_SIZE = 64
HEAD_SIZE = 128


def _reference_qk(
    q: torch.Tensor, k: torch.Tensor, use_qk_l2norm_in_kernel: bool, eps: float = 1e-6
):
    """q/k as the kernel sees them: with use_qk_l2norm_in_kernel the API l2-normalizes
    them as x * rsqrt(sum(x^2) + eps).  MP32 does this without touching the inputs
    (MP31 normalizes in place), so the reference normalizes its own fp32 copy on the
    inputs' device."""
    q, k = q.float(), k.float()
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + eps)
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + eps)
    return q, k


def exclusive_cumsum(seq_lens: list[int]) -> list[int]:
    out = [0]
    for seq_len in seq_lens:
        out.append(out[-1] + seq_len)
    return out


def _get_runtime_device() -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa")
    pytest.skip("MUSA is not available")


def _manual_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if device.type == "musa":
        torch.musa.manual_seed(seed)


def _synchronize(device: torch.device) -> None:
    if device.type == "musa":
        torch.musa.synchronize()


def _resolve_scale(scale: float | str, head_size: int) -> float:
    return head_size**-0.5 if scale == "auto" else float(scale)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
    is_output: bool,
) -> None:
    if dtype == torch.bfloat16:
        rtol = 5e-2 if is_output else 2e-2
        atol = 5e-2 if is_output else 2e-2
    else:
        rtol = 2e-4 if is_output else 1e-2
        atol = 2e-4 if is_output else 1e-2
    torch.testing.assert_close(
        actual.detach().float().cpu(),
        expected.detach().float().cpu(),
        rtol=rtol,
        atol=atol,
    )


def gen_qkv(
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    normalize_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_seq_len = sum(seq_lens)
    q_dim = num_qk_heads * head_size
    k_dim = num_qk_heads * head_size
    v_dim = num_v_heads * head_size
    fused_dim = q_dim + k_dim + v_dim

    mixed_qkv = torch.empty(total_seq_len, fused_dim, device=device, dtype=dtype)
    q_flat, k_flat, v_flat = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

    q_dense = torch.empty(
        total_seq_len, num_qk_heads, head_size, device=device, dtype=dtype
    ).uniform_(-0.25, 0.25)
    k_dense = torch.empty(
        total_seq_len, num_qk_heads, head_size, device=device, dtype=dtype
    ).uniform_(-0.25, 0.25)
    v_dense = torch.empty(
        total_seq_len, num_v_heads, head_size, device=device, dtype=dtype
    ).uniform_(-0.25, 0.25)
    if normalize_k:
        k_dense = F.normalize(k_dense.float(), p=2.0, dim=-1).to(dtype)

    q_flat.copy_(q_dense.reshape(total_seq_len, q_dim))
    k_flat.copy_(k_dense.reshape(total_seq_len, k_dim))
    v_flat.copy_(v_dense.reshape(total_seq_len, v_dim))

    q = q_flat.view(total_seq_len, num_qk_heads, head_size)
    k = k_flat.view(total_seq_len, num_qk_heads, head_size)
    v = v_flat.view(total_seq_len, num_v_heads, head_size)

    assert q.stride() == (fused_dim, head_size, 1)
    assert k.stride() == (fused_dim, head_size, 1)
    assert v.stride() == (fused_dim, head_size, 1)
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()
    return q, k, v


def gen_gates(
    total_seq_len: int,
    num_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.exp(
        -0.02 * torch.rand(total_seq_len, num_heads, device=device, dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(total_seq_len, num_heads, device=device, dtype=torch.float32)
    )
    return g, beta


def _to_kernel_gate_space(g: torch.Tensor, is_log_space: bool) -> torch.Tensor:
    return torch.log(g) if is_log_space else g


def _to_kernel_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: list[int],
    *,
    varlen: bool,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    if varlen:
        cu_seqlens = torch.tensor(
            exclusive_cumsum(seq_lens), device=device, dtype=torch.int32
        )
        return q, k, v, g, beta, cu_seqlens

    assert len(set(seq_lens)) == 1
    batch_size = len(seq_lens)
    num_tokens = seq_lens[0]
    return (
        q.reshape(batch_size, num_tokens, q.shape[1], q.shape[2]),
        k.reshape(batch_size, num_tokens, k.shape[1], k.shape[2]),
        v.reshape(batch_size, num_tokens, v.shape[1], v.shape[2]),
        g.reshape(batch_size, num_tokens, g.shape[1]),
        beta.reshape(batch_size, num_tokens, beta.shape[1]),
        None,
    )


def _reference_output_for_kernel_shape(
    output: torch.Tensor,
    *,
    seq_lens: list[int],
    varlen: bool,
) -> torch.Tensor:
    if varlen:
        return output
    batch_size = len(seq_lens)
    num_tokens = seq_lens[0]
    return output.reshape(batch_size, num_tokens, output.shape[1], output.shape[2])


def _test_prefill_kernel(
    *,
    dtype_name: str,
    num_qk_heads: int,
    num_v_heads: int,
    seq_lens: list[int],
    scale: float | str,
    use_initial_state: bool,
    varlen: bool,
    seed: int,
    is_log_space: bool = True,
    use_zero_alpha: bool = False,
    unit_alpha_stride: int = 0,
    use_qk_l2norm_in_kernel: bool = True,
) -> None:
    device = _get_runtime_device()
    _manual_seed(seed, device)

    dtype = getattr(torch, dtype_name)
    scale = _resolve_scale(scale, HEAD_SIZE)
    scale = 1.0
    q, k, v = gen_qkv(
        seq_lens=seq_lens,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        head_size=HEAD_SIZE,
        dtype=dtype,
        device=device,
        normalize_k=not use_qk_l2norm_in_kernel,
    )
    g, beta = gen_gates(sum(seq_lens), num_v_heads, device)
    if use_zero_alpha:
        g[7::17, :] = 0.0
    if unit_alpha_stride:
        g[::unit_alpha_stride, :] = 1.0
    initial_state = 0.05 * torch.randn(
        len(seq_lens),
        num_v_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        device=device,
        dtype=torch.float32,
    )
    ref_initial_state = (
        initial_state if use_initial_state else torch.zeros_like(initial_state)
    )
    kernel_initial_state = initial_state if use_initial_state else None

    q_in, k_in, v_in, g_in, beta_in, cu_seqlens = _to_kernel_inputs(
        q, k, v, g, beta, seq_lens, varlen=varlen, device=device
    )
    assert q_in.stride(-1) == k_in.stride(-1) == v_in.stride(-1) == 1
    assert not q_in.is_contiguous()
    assert not k_in.is_contiguous()
    assert not v_in.is_contiguous()
    kernel_g_in = _to_kernel_gate_space(g_in, is_log_space)
    actual_o, actual_state = chunk_gated_delta_rule(
        q=q_in,
        k=k_in,
        v=v_in,
        g=kernel_g_in,
        beta=beta_in,
        scale=scale,
        initial_state=kernel_initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_log_space=is_log_space,
    )
    _synchronize(device)

    ref_q, ref_k = _reference_qk(q, k, use_qk_l2norm_in_kernel)
    ref_o, ref_state = blockwise_gdn_prefill_reference(
        ref_q,
        ref_k,
        v.float(),
        g,
        beta,
        seq_lens,
        initial_state=ref_initial_state,
        chunk_size=CHUNK_SIZE,
        scale=scale,
    )
    ref_o = _reference_output_for_kernel_shape(ref_o, seq_lens=seq_lens, varlen=varlen)

    _assert_close(actual_o, ref_o, dtype=dtype, is_output=False)
    _assert_close(actual_state, ref_state, dtype=dtype, is_output=False)


def _concat_varlen_segments(
    x1: torch.Tensor,
    seq_lens1: list[int],
    x2: torch.Tensor,
    seq_lens2: list[int],
) -> torch.Tensor:
    cu_seqlens1 = exclusive_cumsum(seq_lens1)
    cu_seqlens2 = exclusive_cumsum(seq_lens2)
    pieces = []
    for seq_idx in range(len(seq_lens1)):
        pieces.append(x1[cu_seqlens1[seq_idx] : cu_seqlens1[seq_idx + 1]])
        pieces.append(x2[cu_seqlens2[seq_idx] : cu_seqlens2[seq_idx + 1]])
    return torch.cat(pieces, dim=0)


def _test_chunked_prefill(
    *,
    dtype_name: str,
    num_qk_heads: int,
    num_v_heads: int,
    seq_lens1: list[int],
    seq_lens2: list[int],
    scale: float | str,
    seed: int,
    is_log_space: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> None:
    assert len(seq_lens1) == len(seq_lens2)
    device = _get_runtime_device()
    _manual_seed(seed, device)

    dtype = getattr(torch, dtype_name)
    scale = _resolve_scale(scale, HEAD_SIZE)

    q1, k1, v1 = gen_qkv(
        seq_lens1,
        num_qk_heads,
        num_v_heads,
        HEAD_SIZE,
        dtype,
        device,
        normalize_k=not use_qk_l2norm_in_kernel,
    )
    q2, k2, v2 = gen_qkv(
        seq_lens2,
        num_qk_heads,
        num_v_heads,
        HEAD_SIZE,
        dtype,
        device,
        normalize_k=not use_qk_l2norm_in_kernel,
    )
    g1, beta1 = gen_gates(sum(seq_lens1), num_v_heads, device)
    g2, beta2 = gen_gates(sum(seq_lens2), num_v_heads, device)

    q1_in, k1_in, v1_in, g1_in, beta1_in, cu_seqlens1 = _to_kernel_inputs(
        q1, k1, v1, g1, beta1, seq_lens1, varlen=True, device=device
    )
    q2_in, k2_in, v2_in, g2_in, beta2_in, cu_seqlens2 = _to_kernel_inputs(
        q2, k2, v2, g2, beta2, seq_lens2, varlen=True, device=device
    )
    kernel_g1_in = _to_kernel_gate_space(g1_in, is_log_space)
    kernel_g2_in = _to_kernel_gate_space(g2_in, is_log_space)

    actual_o1, state1 = chunk_gated_delta_rule(
        q=q1_in,
        k=k1_in,
        v=v1_in,
        g=kernel_g1_in,
        beta=beta1_in,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens1,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_log_space=is_log_space,
    )
    actual_o2, state2 = chunk_gated_delta_rule(
        q=q2_in,
        k=k2_in,
        v=v2_in,
        g=kernel_g2_in,
        beta=beta2_in,
        scale=scale,
        initial_state=state1,
        output_final_state=True,
        cu_seqlens=cu_seqlens2,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_log_space=is_log_space,
    )
    _synchronize(device)

    q = _concat_varlen_segments(q1, seq_lens1, q2, seq_lens2)
    k = _concat_varlen_segments(k1, seq_lens1, k2, seq_lens2)
    v = _concat_varlen_segments(v1, seq_lens1, v2, seq_lens2)
    g = _concat_varlen_segments(g1, seq_lens1, g2, seq_lens2)
    beta = _concat_varlen_segments(beta1, seq_lens1, beta2, seq_lens2)
    seq_lens = [a + b for a, b in zip(seq_lens1, seq_lens2)]
    initial_state = torch.zeros(
        len(seq_lens),
        num_v_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        device=device,
        dtype=torch.float32,
    )
    ref_q, ref_k = _reference_qk(q, k, use_qk_l2norm_in_kernel)
    ref_o, ref_state = blockwise_gdn_prefill_reference(
        ref_q,
        ref_k,
        v.float(),
        g,
        beta,
        seq_lens,
        initial_state=initial_state,
        chunk_size=CHUNK_SIZE,
        scale=scale,
    )

    actual_o = _concat_varlen_segments(actual_o1, seq_lens1, actual_o2, seq_lens2)
    _assert_close(actual_o, ref_o, dtype=dtype, is_output=True)
    _assert_close(state2, ref_state, dtype=dtype, is_output=False)


PREFILL_CASES = [
    pytest.param([64], 1, 1, "auto", False, False, 0, id="full-64-h1-zero"),
    pytest.param([128, 128], 2, 2, 1.0, True, False, 1, id="batch2-128-h2-init"),
    pytest.param([128], 1, 2, "auto", True, False, 2, id="gva-full-128-init"),
]

VARLEN_PREFILL_CASES = [
    pytest.param([31], 1, 1, "auto", False, 3, id="varlen-tail-31-h1-zero"),
    pytest.param([64, 128], 2, 2, 1.0, True, 4, id="varlen-64-128-h2-init"),
    pytest.param([31, 63, 93], 1, 2, "auto", True, 5, id="gva-varlen-tails-init"),
]

CHUNKED_PREFILL_CASES = [
    pytest.param([64], [128], 1, 1, "auto", 6, id="chunked-full-h1"),
    pytest.param([31, 63], [33, 65], 1, 2, 1.0, 7, id="gva-chunked-tails"),
]


ZERO_LOG_GATE_CASES = [
    pytest.param([256], 1, 1, False, 1, 10, id="full-256-all"),
    pytest.param([256], 1, 2, False, 5, 11, id="gva-256-every5"),
    pytest.param([100, 37, 300], 1, 1, True, 1, 12, id="varlen-all"),
    pytest.param([100, 37, 300], 2, 2, True, 5, 13, id="varlen-every5-h2"),
]


# @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("dtype_name", ["float16"])
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "scale",
        "use_initial_state",
        "varlen",
        "seed_offset",
    ),
    PREFILL_CASES,
)
def test_gdn_prefill_matches_reference(
    dtype_name: str,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    scale: float | str,
    use_initial_state: bool,
    varlen: bool,
    seed_offset: int,
) -> None:
    seed = int(os.environ.get("SEED", "0")) + seed_offset
    _test_prefill_kernel(
        dtype_name=dtype_name,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        seq_lens=seq_lens,
        scale=scale,
        use_initial_state=use_initial_state,
        varlen=varlen,
        seed=seed,
    )


def test_gdn_prefill_alpha_space_matches_reference() -> None:
    seed = int(os.environ.get("SEED", "0")) + 8
    _test_prefill_kernel(
        dtype_name="float16",
        num_qk_heads=1,
        num_v_heads=1,
        seq_lens=[64],
        scale=1.0,
        use_initial_state=False,
        varlen=False,
        seed=seed,
        is_log_space=False,
    )


@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "varlen",
        "unit_alpha_stride",
        "seed_offset",
    ),
    ZERO_LOG_GATE_CASES,
)
def test_gdn_prefill_zero_log_gate_matches_reference(
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    varlen: bool,
    unit_alpha_stride: int,
    seed_offset: int,
) -> None:
    # log g == 0 (no decay) makes the in-chunk cumsum flat, so the causal mask must
    # come from token order, not from the sign of the rounded cumsum difference.
    seed = int(os.environ.get("SEED", "0")) + seed_offset
    _test_prefill_kernel(
        dtype_name="float16",
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        seq_lens=seq_lens,
        scale=1.0,
        use_initial_state=True,
        varlen=varlen,
        seed=seed,
        unit_alpha_stride=unit_alpha_stride,
    )


def test_gdn_prefill_log_space_neg_inf_matches_reference() -> None:
    seed = int(os.environ.get("SEED", "0")) + 9
    _test_prefill_kernel(
        dtype_name="float16",
        num_qk_heads=1,
        num_v_heads=1,
        seq_lens=[64],
        scale=1.0,
        use_initial_state=True,
        varlen=False,
        seed=seed,
        is_log_space=True,
        use_zero_alpha=True,
    )


# @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("dtype_name", ["float16"])
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "scale",
        "use_initial_state",
        "seed_offset",
    ),
    VARLEN_PREFILL_CASES,
)
def test_gdn_prefill_varlen_matches_reference(
    dtype_name: str,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    scale: float | str,
    use_initial_state: bool,
    seed_offset: int,
) -> None:
    seed = int(os.environ.get("SEED", "0")) + seed_offset
    _test_prefill_kernel(
        dtype_name=dtype_name,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        seq_lens=seq_lens,
        scale=scale,
        use_initial_state=use_initial_state,
        varlen=True,
        seed=seed,
    )


@pytest.mark.parametrize("dtype_name", ["float16"])
# @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    ("seq_lens1", "seq_lens2", "num_qk_heads", "num_v_heads", "scale", "seed_offset"),
    CHUNKED_PREFILL_CASES,
)
def test_gdn_prefill_chunked_matches_reference(
    dtype_name: str,
    seq_lens1: list[int],
    seq_lens2: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    scale: float | str,
    seed_offset: int,
) -> None:
    seed = int(os.environ.get("SEED", "0")) + seed_offset
    _test_chunked_prefill(
        dtype_name=dtype_name,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        seq_lens1=seq_lens1,
        seq_lens2=seq_lens2,
        scale=scale,
        seed=seed,
    )


@pytest.mark.parametrize("varlen", [True, False])
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("preallocated_state", [True, False])
def test_gdn_prefill_zero_tokens(
    varlen: bool, use_initial_state: bool, preallocated_state: bool
) -> None:
    """An input without tokens launches no kernel and passes the state through."""
    device = _get_runtime_device()
    _manual_seed(int(os.environ.get("SEED", "0")) + 703, device)
    num_seqs, num_qk_heads, num_v_heads = 3, 1, 2
    token_shape = (0,) if varlen else (num_seqs, 0)
    q = torch.empty(
        *token_shape, num_qk_heads, HEAD_SIZE, dtype=torch.float16, device=device
    )
    k = torch.empty_like(q)
    v = torch.empty(
        *token_shape, num_v_heads, HEAD_SIZE, dtype=torch.float16, device=device
    )
    g = torch.empty(*token_shape, num_v_heads, dtype=torch.float32, device=device)
    beta = torch.empty_like(g)
    cu_seqlens = (
        torch.zeros(num_seqs + 1, dtype=torch.int32, device=device) if varlen else None
    )
    initial_state = 0.05 * torch.randn(
        num_seqs, num_v_heads, HEAD_SIZE, HEAD_SIZE, device=device
    )
    output_state = (
        torch.full_like(initial_state, float("nan")) if preallocated_state else None
    )

    o, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=HEAD_SIZE**-0.5,
        initial_state=initial_state if use_initial_state else None,
        output_final_state=True,
        output_state=output_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    _synchronize(device)

    assert o.shape == v.shape
    expected = initial_state if use_initial_state else torch.zeros_like(initial_state)
    assert torch.equal(final_state, expected)
    if preallocated_state:
        assert final_state.data_ptr() == output_state.data_ptr()


@pytest.mark.parametrize("varlen", [True, False])
def test_gdn_prefill_mp32_state_checkpoints_zero_tokens(varlen: bool) -> None:
    device = _get_runtime_device()
    if get_arch(device) != MP32_ARCH:
        pytest.skip("state checkpoints are only implemented by the MP32 kernel")
    from mate.gdn_kernels.tilelang.mp32.gdn_prefill import fused_chunk_gdn_prefill

    num_seqs, num_v_heads = 2, 2
    batch = 1 if varlen else num_seqs
    q = torch.empty(batch, 0, 1, HEAD_SIZE, dtype=torch.float16, device=device)
    v = torch.empty(
        batch, 0, num_v_heads, HEAD_SIZE, dtype=torch.float16, device=device
    )
    g = torch.empty(batch, 0, num_v_heads, dtype=torch.float32, device=device)
    initial_state = 0.05 * torch.randn(
        num_seqs, num_v_heads, HEAD_SIZE, HEAD_SIZE, device=device
    )
    cu_seqlens = (
        torch.zeros(num_seqs + 1, dtype=torch.int32, device=device) if varlen else None
    )

    o, checkpoints, ht = fused_chunk_gdn_prefill(
        q,
        torch.empty_like(q),
        v,
        None,
        g,
        torch.empty_like(g),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        l2norm_eps=1e-6,
        checkpoint_every_n_tokens=CHUNK_SIZE,
    )
    _synchronize(device)

    assert o.shape == v.shape
    rows = num_seqs if varlen else 0
    assert checkpoints.shape == (rows, num_v_heads, HEAD_SIZE, HEAD_SIZE)
    assert torch.equal(ht, initial_state)


@pytest.mark.parametrize("every_n_tokens", [CHUNK_SIZE, 3 * CHUNK_SIZE])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    ("seq_lens", "varlen"),
    [([100, 64, 200, 0, 193], True), ([192, 192], False)],
)
def test_gdn_prefill_mp32_state_checkpoints(
    every_n_tokens: int, dtype_name: str, seq_lens: list[int], varlen: bool
) -> None:
    device = _get_runtime_device()
    if get_arch(device) != MP32_ARCH:
        pytest.skip("state checkpoints are only implemented by the MP32 kernel")
    from mate.gdn_kernels.tilelang.mp32.gdn_prefill import fused_chunk_gdn_prefill

    _manual_seed(int(os.environ.get("SEED", "0")) + 700, device)
    dtype = getattr(torch, dtype_name)
    num_qk_heads, num_v_heads, scale, eps = 2, 4, HEAD_SIZE**-0.5, 1e-6
    q, k, v = gen_qkv(seq_lens, num_qk_heads, num_v_heads, HEAD_SIZE, dtype, device)
    g, beta = gen_gates(sum(seq_lens), num_v_heads, device)
    initial_state = 0.05 * torch.randn(
        len(seq_lens), num_v_heads, HEAD_SIZE, HEAD_SIZE, device=device
    )
    q_in, k_in, v_in, g_in, beta_in, cu_seqlens = _to_kernel_inputs(
        q, k, v, g, beta, seq_lens, varlen=varlen, device=device
    )
    if varlen:
        q_in, k_in, v_in, g_in, beta_in = (
            x.unsqueeze(0) for x in (q_in, k_in, v_in, g_in, beta_in)
        )

    def run(every_n: int):
        return fused_chunk_gdn_prefill(
            q_in,
            k_in,
            v_in,
            None,
            torch.log(g_in),
            beta_in,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            l2norm_eps=eps,
            checkpoint_every_n_tokens=every_n,
        )

    o_ref_kernel, none, ht_ref_kernel = run(0)
    o, checkpoints, ht = run(every_n_tokens)
    _synchronize(device)

    # Storing checkpoints must not change o / final_state.
    assert none is None
    assert torch.equal(o, o_ref_kernel)
    assert torch.equal(ht, ht_ref_kernel)

    counts = [(n + every_n_tokens - 1) // every_n_tokens for n in seq_lens]
    total = sum(seq_lens)
    rows = max(total // every_n_tokens + len(seq_lens), 1) if varlen else sum(counts)
    assert checkpoints.shape == (rows, num_v_heads, HEAD_SIZE, HEAD_SIZE)
    assert checkpoints.dtype == dtype
    first = exclusive_cumsum(counts)
    stored = checkpoints[: first[-1]]

    # The first checkpoint of every non-empty sequence is the initial state.
    for seq_idx, row in enumerate(first[:-1]):
        if counts[seq_idx]:
            assert torch.equal(stored[row], initial_state[seq_idx].to(dtype))

    # Each checkpoint is the reference state before tokens 0, N, 2N, ...
    ref_q, ref_k = _reference_qk(q, k, True, eps)
    chunk_states: list[torch.Tensor] = []
    blockwise_gdn_prefill_reference(
        ref_q,
        ref_k,
        v.float(),
        g,
        beta,
        seq_lens,
        initial_state=initial_state,
        chunk_size=CHUNK_SIZE,
        scale=scale,
        chunk_states=chunk_states,
    )
    stride = every_n_tokens // CHUNK_SIZE
    chunks = [(n + CHUNK_SIZE - 1) // CHUNK_SIZE for n in seq_lens]
    picked = [
        chunk_states[c0 + c]
        for c0, n in zip(exclusive_cumsum(chunks), chunks)
        for c in range(0, n, stride)
    ]
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(stored.float(), torch.stack(picked), rtol=tol, atol=tol)

    if every_n_tokens != CHUNK_SIZE:
        # A coarser cadence stores exactly a subset of the per-chunk states.
        _, per_chunk, _ = run(CHUNK_SIZE)
        _synchronize(device)
        subset = [
            c0 + c
            for c0, n in zip(exclusive_cumsum(chunks), chunks)
            for c in range(0, n, stride)
        ]
        assert torch.equal(stored, per_chunk[subset])


def test_gdn_prefill_state_checkpoints_public_api() -> None:
    device = _get_runtime_device()
    _manual_seed(int(os.environ.get("SEED", "0")) + 704, device)
    seq_lens = [130, 64]
    q, k, v = gen_qkv(seq_lens, 1, 2, HEAD_SIZE, torch.float16, device)
    g, beta = gen_gates(sum(seq_lens), 2, device)
    cu_seqlens = torch.tensor(
        exclusive_cumsum(seq_lens), device=device, dtype=torch.int32
    )

    def call(every_n: int):
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=torch.log(g),
            beta=beta,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            checkpoint_every_n_tokens=every_n,
        )

    if get_arch(device) != MP32_ARCH:
        with pytest.raises(NotImplementedError):
            call(CHUNK_SIZE)
        return
    with pytest.raises(ValueError):
        call(CHUNK_SIZE + 1)
    o_ref, ht_ref = call(0)
    o, ht, checkpoints = call(2 * CHUNK_SIZE)
    _synchronize(device)
    assert torch.equal(o, o_ref)
    assert torch.equal(ht, ht_ref)
    assert checkpoints.shape == (sum(seq_lens) // 128 + 2, 2, HEAD_SIZE, HEAD_SIZE)
    assert checkpoints.dtype == q.dtype


def test_gdn_prefill_mp32_varlen_tail_ignores_memory_after_rnorm(monkeypatch) -> None:
    """Padded rows of the last tail chunk must not read past the rnorm buffer."""
    device = _get_runtime_device()
    if get_arch(device) != MP32_ARCH:
        pytest.skip("q/k norm folding is only implemented by the MP32 kernel")
    import mate.gdn_kernels.tilelang.mp32.gdn_prefill as mp32_prefill

    _manual_seed(int(os.environ.get("SEED", "0")) + 702, device)
    seq_lens = [100, 37, 300]
    num_qk_heads, num_v_heads = 2, 2
    q, k, v = gen_qkv(
        seq_lens, num_qk_heads, num_v_heads, HEAD_SIZE, torch.float16, device
    )
    g, beta = gen_gates(sum(seq_lens), num_v_heads, device)
    initial_state = 0.05 * torch.randn(
        len(seq_lens), num_v_heads, HEAD_SIZE, HEAD_SIZE, device=device
    )
    cu_seqlens = torch.tensor(
        exclusive_cumsum(seq_lens), device=device, dtype=torch.int32
    )

    def run():
        return mp32_prefill.fused_chunk_gdn_prefill(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            None,
            torch.log(g).unsqueeze(0),
            beta.unsqueeze(0),
            scale=HEAD_SIZE**-0.5,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            l2norm_eps=1e-6,
        )

    o_ref, _, ht_ref = run()
    rnorm = mp32_prefill.gdn_qk_rnorm

    def rnorm_followed_by_nan(q, k, eps, *args, **kwargs):
        rn = rnorm(q, k, eps, *args, **kwargs)
        padded = torch.full(
            (rn.numel() + 4096,), float("nan"), dtype=rn.dtype, device=rn.device
        )
        padded[: rn.numel()] = rn.reshape(-1)
        return padded[: rn.numel()].view(rn.shape)

    monkeypatch.setattr(mp32_prefill, "gdn_qk_rnorm", rnorm_followed_by_nan)
    o, _, ht = run()
    _synchronize(device)

    assert torch.isfinite(o).all() and torch.isfinite(ht).all()
    assert torch.equal(o, o_ref)
    assert torch.equal(ht, ht_ref)


def _mp32_prefill_from_seq(
    q, k, v, g, beta, initial_state, seq_len: int, start_seq: int, **kwargs
):
    """Run sequences ``start_seq..`` of an equal-length varlen batch on MP32."""
    from mate.gdn_kernels.tilelang.mp32.gdn_prefill import fused_chunk_gdn_prefill

    total = q.shape[1]
    tokens = slice(start_seq * seq_len, total)
    return fused_chunk_gdn_prefill(
        q[:, tokens],
        k[:, tokens],
        v[:, tokens],
        None,
        g[:, tokens],
        beta[:, tokens],
        scale=HEAD_SIZE**-0.5,
        initial_state=initial_state[start_seq:],
        output_final_state=True,
        cu_seqlens=torch.arange(
            0,
            total - start_seq * seq_len + 1,
            seq_len,
            dtype=torch.int32,
            device=q.device,
        ),
        l2norm_eps=1e-6,
        **kwargs,
    )


def test_gdn_prefill_mp32_state_checkpoints_over_2gib() -> None:
    """Checkpoints past 2**31 bytes must be stored with 64-bit addresses."""
    device = _get_runtime_device()
    if get_arch(device) != MP32_ARCH:
        pytest.skip("state checkpoints are only implemented by the MP32 kernel")

    _manual_seed(int(os.environ.get("SEED", "0")) + 701, device)
    # 4 seqs x 258 chunks x 64 heads x 128 x 128 x fp16 = 2.16 GB > 2**31 B.
    num_seqs, chunks_per_seq, num_qk_heads, num_v_heads = 4, 258, 16, 64
    seq_len = chunks_per_seq * CHUNK_SIZE
    total = num_seqs * seq_len
    q = torch.randn(
        1, total, num_qk_heads, HEAD_SIZE, dtype=torch.float16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total, num_v_heads, HEAD_SIZE, dtype=torch.float16, device=device
    )
    g = -torch.rand(1, total, num_v_heads, device=device)
    beta = torch.sigmoid(torch.randn(1, total, num_v_heads, device=device))
    initial_state = 0.05 * torch.randn(
        num_seqs, num_v_heads, HEAD_SIZE, HEAD_SIZE, device=device
    )

    def run(start_seq: int):
        return _mp32_prefill_from_seq(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            seq_len,
            start_seq,
            checkpoint_every_n_tokens=CHUNK_SIZE,
        )

    o, h, ht = run(0)
    used = num_seqs * chunks_per_seq
    assert h[:used].numel() * h.element_size() > 2**31
    # Sequences are independent, so the last one alone (small h) must match bitwise.
    o_last, h_last, ht_last = run(num_seqs - 1)
    _synchronize(device)

    assert torch.equal(o[:, -seq_len:], o_last)
    assert torch.equal(ht[-1:], ht_last)
    assert torch.equal(h[used - chunks_per_seq : used], h_last[:chunks_per_seq])
    for seq_idx in range(num_seqs):
        assert torch.equal(
            h[seq_idx * chunks_per_seq], initial_state[seq_idx].to(torch.float16)
        )


def test_gdn_prefill_mp32_fused_qkv_over_2gib() -> None:
    """q/k/v split from one fused qkv that spans more than 2**31 bytes."""
    device = _get_runtime_device()
    if get_arch(device) != MP32_ARCH:
        pytest.skip("the read-only rnorm kernel is only used on MP32")

    _manual_seed(int(os.environ.get("SEED", "0")) + 702, device)
    # Qwen3.5-397B TP1 heads: 6 x 16384 tokens x (16 + 16 + 64) x 128 x fp16 = 2.42 GB.
    num_seqs, seq_len, num_qk_heads, num_v_heads = 6, 16384, 16, 64
    total = num_seqs * seq_len
    mixed_qkv = torch.randn(
        1,
        total,
        (2 * num_qk_heads + num_v_heads) * HEAD_SIZE,
        dtype=torch.float16,
        device=device,
    )
    assert mixed_qkv.numel() * mixed_qkv.element_size() > 2**31
    q, k, v = torch.split(
        mixed_qkv,
        [num_qk_heads * HEAD_SIZE, num_qk_heads * HEAD_SIZE, num_v_heads * HEAD_SIZE],
        dim=-1,
    )
    q = q.view(1, total, num_qk_heads, HEAD_SIZE)
    k = k.view(1, total, num_qk_heads, HEAD_SIZE)
    v = v.view(1, total, num_v_heads, HEAD_SIZE)
    g = -torch.rand(1, total, num_v_heads, device=device)
    beta = torch.sigmoid(torch.randn(1, total, num_v_heads, device=device))
    initial_state = 0.05 * torch.randn(
        num_seqs, num_v_heads, HEAD_SIZE, HEAD_SIZE, device=device
    )

    o, _, ht = _mp32_prefill_from_seq(q, k, v, g, beta, initial_state, seq_len, 0)
    # Sequences are independent; the last one alone spans far less than 2**31 bytes.
    o_last, _, ht_last = _mp32_prefill_from_seq(
        q, k, v, g, beta, initial_state, seq_len, num_seqs - 1
    )
    _synchronize(device)

    assert torch.isfinite(o).all() and torch.isfinite(ht).all()
    assert torch.equal(o[:, -seq_len:], o_last)
    assert torch.equal(ht[-1:], ht_last)


def test_gdn_prefill_mp32_states_over_2gib() -> None:
    """Initial / final states past 2**31 bytes must use 64-bit addresses."""
    device = _get_runtime_device()
    if get_arch(device) != MP32_ARCH:
        pytest.skip("covers the MP32 kernel's state addressing")

    _manual_seed(int(os.environ.get("SEED", "0")) + 703, device)
    # 516 seqs x 64 heads x 128 x 128 x fp32 = 2.16 GB > 2**31 B.
    num_seqs, seq_len, num_qk_heads, num_v_heads = 516, CHUNK_SIZE, 16, 64
    total = num_seqs * seq_len
    q = torch.randn(
        1, total, num_qk_heads, HEAD_SIZE, dtype=torch.float16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total, num_v_heads, HEAD_SIZE, dtype=torch.float16, device=device
    )
    g = -torch.rand(1, total, num_v_heads, device=device)
    beta = torch.sigmoid(torch.randn(1, total, num_v_heads, device=device))
    initial_state = 0.05 * torch.randn(
        num_seqs, num_v_heads, HEAD_SIZE, HEAD_SIZE, device=device
    )
    assert initial_state.numel() * initial_state.element_size() > 2**31

    o, _, ht = _mp32_prefill_from_seq(q, k, v, g, beta, initial_state, seq_len, 0)
    o_last, _, ht_last = _mp32_prefill_from_seq(
        q, k, v, g, beta, initial_state, seq_len, num_seqs - 1
    )
    _synchronize(device)

    assert torch.isfinite(o).all() and torch.isfinite(ht).all()
    assert torch.equal(o[:, -seq_len:], o_last)
    assert torch.equal(ht[-1:], ht_last)
