import functools

import torch

from mate.mate_runtime import get_arch


@functools.cache
def _get_transform_mxfp_sf_kernel(dtype: str, gran_mn: int, gran_k: int, target="musa"):
    """General positive-stride path, including unaligned input and MN tails."""
    import tilelang
    import tilelang.language as T

    groups, src_mn, src_q = T.dynamic("groups, src_mn, src_q")
    mn, q = T.dynamic("mn, q")
    stride_g, stride_mn, stride_q = T.dynamic("stride_g, stride_mn, stride_q")
    block_mn, block_q = 128, 16

    @tilelang.jit(
        target=target if target == "musa" else {"kind": "musa", "arch": target}
    )
    def transform(
        sf: T.StridedTensor[
            (groups, src_mn, src_q), (stride_g, stride_mn, stride_q), dtype
        ],
        out: T.Tensor[(groups, q, mn), T.uint8],
    ):
        with T.Kernel(
            T.ceildiv(mn, block_mn), T.ceildiv(q, block_q), groups, threads=128
        ) as (bm, bq, g):
            row = bm * block_mn + T.get_thread_binding()
            for j in T.serial(block_q):
                col = bq * block_q + j
                if row < mn and col < q:
                    src_row = row // gran_mn
                    src_col = col // (gran_k // 32)
                    if dtype == "int32":
                        word = T.cast(sf[g, src_row, src_col // 4], T.uint32)
                        out[g, col, row] = T.cast(
                            (word >> ((src_col % 4) * 8)) & 255, T.uint8
                        )
                    elif dtype == "float32":
                        bits = T.reinterpret(sf[g, src_row, src_col], T.uint32)
                        out[g, col, row] = T.cast((bits >> 23) & 255, T.uint8)
                    else:
                        out[g, col, row] = sf[g, src_row, src_col]

    return transform


@functools.cache
def _vector_kernel(
    dtype,
    mn,
    q,
    gran_mn,
    gran_k,
    stride_mn,
    stride_q,
    vector=16,
    threads=128,
    block_mn=128,
    target="musa",
):
    """Cast contiguous MN vectors and reuse packed/broadcast source scales."""
    import tilelang
    import tilelang.language as T

    groups, stride_g = T.dynamic("groups, stride_g")
    packing = 4 if dtype == "int32" else 1
    repeat = gran_k // 32
    src_mn = (mn + gran_mn - 1) // gran_mn
    src_q = (q + repeat * packing - 1) // (repeat * packing)
    block_src_q = threads // (block_mn // vector)
    load_vector = min(vector, 16 if dtype == "uint8" else 4)

    @tilelang.jit(
        target=target if target == "musa" else {"kind": "musa", "arch": target},
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        },
    )
    def mxfp_sf_vector(
        sf: T.StridedTensor[
            (groups, src_mn, src_q), (stride_g, stride_mn, stride_q), dtype
        ],
        out: T.Tensor[(groups, q, mn), T.uint8],
    ):
        with T.Kernel(
            T.ceildiv(mn, block_mn),
            T.ceildiv(src_q, block_src_q),
            groups,
            threads=threads,
        ) as (bm, bq, g):
            tx = T.get_thread_binding()
            row = bm * block_mn + (tx % (block_mn // vector)) * vector
            col = bq * block_src_q + tx // (block_mn // vector)
            values = T.alloc_local((vector,), dtype)
            codes = T.alloc_local((vector,), T.uint8)
            if gran_mn == 1:
                # The selector checks alignment, including the group stride.
                T.assume(stride_g % load_vector == 0)
            if row < mn and col < src_q:
                if gran_mn == 1:
                    for chunk in T.unroll(vector // load_vector):
                        for v in T.vectorized(load_vector):
                            values[chunk * load_vector + v] = sf[
                                g, row + chunk * load_vector + v, col
                            ]
                else:
                    value = sf[g, row // gran_mn, col]
                    # Scalar broadcasts avoid a uint8x16 constructor bug in
                    # older MUSA codegen; the output still uses wide stores.
                    for v in T.unroll(vector):
                        values[v] = value
                for byte in T.unroll(packing):
                    for v in T.unroll(vector):
                        if dtype == "float32":
                            codes[v] = T.cast(
                                (T.reinterpret(values[v], T.uint32) >> 23) & 255,
                                T.uint8,
                            )
                        elif dtype == "int32":
                            codes[v] = T.cast(
                                (T.cast(values[v], T.uint32) >> (8 * byte)) & 255,
                                T.uint8,
                            )
                        else:
                            codes[v] = values[v]
                    for rep in T.unroll(repeat):
                        out_col = (col * packing + byte) * repeat + rep
                        if out_col < q:
                            for v in T.vectorized(vector):
                                out[g, out_col, row + v] = codes[v]

    return mxfp_sf_vector


@functools.cache
def _transpose_kernel(
    dtype,
    mn,
    q,
    gran_k,
    stride_mn,
    stride_q,
    vector=4,
    threads=256,
    block_mn=128,
    block_q=32,
    target="musa",
):
    """Coalesced K loads, packed shared transpose, and contiguous MN stores."""
    import tilelang
    import tilelang.language as T

    groups, stride_g = T.dynamic("groups, stride_g")
    packing = 4 if dtype == "int32" else 1
    repeat = gran_k // 32
    logical_src_q = (q + repeat - 1) // repeat
    src_q = (logical_src_q + packing - 1) // packing
    block_packed_q = block_q // 4

    @tilelang.jit(
        target=target if target == "musa" else {"kind": "musa", "arch": target},
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        },
    )
    def mxfp_sf_transpose(
        sf: T.StridedTensor[
            (groups, mn, src_q), (stride_g, stride_mn, stride_q), dtype
        ],
        out: T.Tensor[(groups, q, mn), T.uint8],
    ):
        with T.Kernel(
            T.ceildiv(mn, block_mn),
            T.ceildiv(logical_src_q, block_q),
            groups,
            threads=threads,
        ) as (bm, bq, g):
            T.assume(stride_g % (16 if dtype == "uint8" else 4) == 0)
            tx = T.get_thread_binding()
            # Four K codes per word; the extra word separates shared banks
            # when the store phase reads across MN.
            packed = T.alloc_shared((block_mn, block_packed_q + 1), T.uint32)
            loaded = T.alloc_local((4,), dtype)
            words = T.alloc_local((vector,), T.uint32)
            codes = T.alloc_local((vector,), T.uint8)
            for i in T.unroll(T.ceildiv(block_mn * block_packed_q, threads)):
                idx = i * threads + tx
                row = idx // block_packed_q
                col = idx % block_packed_q
                word = T.alloc_var(T.uint32, init=0)
                if idx < block_mn * block_packed_q and bm * block_mn + row < mn:
                    if dtype == "int32":
                        if bq * block_packed_q + col < src_q:
                            word = T.cast(
                                sf[g, bm * block_mn + row, bq * block_packed_q + col],
                                T.uint32,
                            )
                    else:
                        if (bq * block_packed_q + col) * 4 + 3 < src_q:
                            for v in T.vectorized(4):
                                loaded[v] = sf[
                                    g,
                                    bm * block_mn + row,
                                    (bq * block_packed_q + col) * 4 + v,
                                ]
                        else:
                            for v in T.unroll(4):
                                loaded[v] = 0
                                if (bq * block_packed_q + col) * 4 + v < src_q:
                                    loaded[v] = sf[
                                        g,
                                        bm * block_mn + row,
                                        (bq * block_packed_q + col) * 4 + v,
                                    ]
                        for v in T.unroll(4):
                            if dtype == "float32":
                                word = word | (
                                    ((T.reinterpret(loaded[v], T.uint32) >> 23) & 255)
                                    << (8 * v)
                                )
                            else:
                                word = word | (T.cast(loaded[v], T.uint32) << (8 * v))
                if idx < block_mn * block_packed_q:
                    packed[row, col] = word
            T.sync_threads()
            for i in T.unroll(
                T.ceildiv((block_mn // vector) * block_packed_q, threads)
            ):
                idx = i * threads + tx
                row = (idx % (block_mn // vector)) * vector
                col = idx // (block_mn // vector)
                if (
                    idx < (block_mn // vector) * block_packed_q
                    and bm * block_mn + row < mn
                ):
                    for v in T.unroll(vector):
                        words[v] = packed[row + v, col]
                    for byte in T.unroll(4):
                        for v in T.vectorized(vector):
                            codes[v] = T.cast((words[v] >> (8 * byte)) & 255, T.uint8)
                        for rep in T.unroll(repeat):
                            out_col = (
                                (bq * block_packed_q + col) * 4 + byte
                            ) * repeat + rep
                            if out_col < q:
                                for v in T.vectorized(vector):
                                    out[g, out_col, bm * block_mn + row + v] = codes[v]

    return mxfp_sf_transpose


def _select_transform_mxfp_sf_kernel(sf, mn, k, recipe):
    dtype = {torch.float32: "float32", torch.int32: "int32", torch.uint8: "uint8"}[
        sf.dtype
    ]
    gran_mn, gran_k = recipe
    q = (k + 31) // 32
    stride_mn, stride_q = sf.stride()[-2:]
    arch = get_arch(sf.device)
    target = f"mp_{arch.major}{arch.minor}"
    if mn % 4 == 0:
        vector = 16 if mn % 16 == 0 else 4
        if gran_mn == 128:
            return _vector_kernel(
                dtype,
                mn,
                q,
                gran_mn,
                gran_k,
                stride_mn,
                stride_q,
                vector=vector,
                target=target,
            )
        if stride_mn == 1:
            # UINT8 MN padding can permit only 4-byte loads. FP32/INT32
            # loads always require 16-byte alignment.
            for width in (vector, 4) if vector != 4 else (4,):
                alignment = width if dtype == "uint8" else 16
                if sf.data_ptr() % alignment == 0 and all(
                    s * sf.element_size() % alignment == 0
                    for s in sf.stride()[:-2] + (stride_q,)
                ):
                    return _vector_kernel(
                        dtype,
                        mn,
                        q,
                        gran_mn,
                        gran_k,
                        stride_mn,
                        stride_q,
                        vector=width,
                        target=target,
                    )
        if (
            stride_q == 1
            and sf.data_ptr() % 16 == 0
            and all(s * sf.element_size() % 16 == 0 for s in sf.stride()[:-1])
        ):
            if dtype == "float32":
                width, threads, block_q = (
                    (16, 128, 32) if gran_k == 32 else (8, 256, 32)
                )
            else:
                width, threads, block_q = (8 if gran_k == 32 else 16), 256, 64
            width = width if mn % width == 0 else 4
            return _transpose_kernel(
                dtype,
                mn,
                q,
                gran_k,
                stride_mn,
                stride_q,
                vector=width,
                threads=threads,
                block_q=block_q,
                target=target,
            )
    return _get_transform_mxfp_sf_kernel(dtype, gran_mn, gran_k, target)


def run_transform_mxfp_sf(
    sf: torch.Tensor, mn: int, k: int, recipe: tuple[int, int]
) -> torch.Tensor:
    groups = sf.shape[0] if sf.ndim == 3 else 1
    q = (k + 31) // 32
    # The kernel writes contiguous [G,Q,MN] storage; expose logical [G,MN,Q].
    storage = torch.empty((groups, q, mn), dtype=torch.uint8, device=sf.device)
    shape = (groups, mn, q) if sf.ndim == 3 else (mn, q)
    strides = (mn * q, 1, mn) if sf.ndim == 3 else (1, mn)
    out = storage.as_strided(shape, strides)
    if out.numel() == 0:
        return out
    kernel = _select_transform_mxfp_sf_kernel(sf, mn, k, recipe)
    kernel(sf if sf.ndim == 3 else sf.unsqueeze(0), storage)
    return out
