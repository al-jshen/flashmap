# coding=utf-8
"""Triton kernels of the FlashMap sglang draft (see :mod:`.model`).

They reproduce the rounding of the reference modules they replace: the RMSNorm
multiplies the bf16 input by its bf16-rounded scale ``weight * rsqrt(mean(x^2) +
eps)``, and the interleaved-pair RoPE rotates in float32 and rounds back to
bf16. The ops are registered as custom ops so torch.compile keeps them whole
inside the draft step it compiles.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_self_attention_kernel(
    qkv_ptr,
    q_norm_ptr,
    k_norm_ptr,
    freqs_ptr,
    out_ptr,
    eps,
    sm_scale,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # One program per (block, head): the block's T queries attend to its T keys.
    b = tl.program_id(0) // H
    h = tl.program_id(0) % H
    HALF: tl.constexpr = D // 2
    t = tl.arange(0, T)[:, None]
    i = tl.arange(0, HALF)[None, :]
    d = tl.arange(0, D)[None, :]
    # qkv is (blocks, T, 3, H, D); RoPE pairs dims (2i, 2i + 1).
    row = qkv_ptr + (b * T + t) * (3 * H * D) + h * D
    freqs = freqs_ptr + (b * T + t) * D + 2 * i  # (blocks, T, D / 2, [cos, sin])
    cos = tl.load(freqs)
    sin = tl.load(freqs + 1)

    q_even = tl.load(row + 2 * i).to(tl.float32)
    q_odd = tl.load(row + 2 * i + 1).to(tl.float32)
    k_even = tl.load(row + H * D + 2 * i).to(tl.float32)
    k_odd = tl.load(row + H * D + 2 * i + 1).to(tl.float32)

    # RMSNorm: x * bf16(weight * rsqrt(mean(x^2) + eps)), in bf16.
    q_rms = tl.rsqrt((tl.sum(q_even * q_even, 1) + tl.sum(q_odd * q_odd, 1)) / D + eps)[
        :, None
    ]
    k_rms = tl.rsqrt((tl.sum(k_even * k_even, 1) + tl.sum(k_odd * k_odd, 1)) / D + eps)[
        :, None
    ]
    q_even = _scale(q_even, tl.load(q_norm_ptr + 2 * i) * q_rms)
    q_odd = _scale(q_odd, tl.load(q_norm_ptr + 2 * i + 1) * q_rms)
    k_even = _scale(k_even, tl.load(k_norm_ptr + 2 * i) * k_rms)
    k_odd = _scale(k_odd, tl.load(k_norm_ptr + 2 * i + 1) * k_rms)

    # RoPE, rounded to bf16 as the attention receives it.
    q_rot_even = (q_even * cos - q_odd * sin).to(tl.bfloat16)
    q_rot_odd = (q_odd * cos + q_even * sin).to(tl.bfloat16)
    k_rot_even = (k_even * cos - k_odd * sin).to(tl.bfloat16)
    k_rot_odd = (k_odd * cos + k_even * sin).to(tl.bfloat16)

    scores = tl.dot(q_rot_even, tl.trans(k_rot_even)) + tl.dot(
        q_rot_odd, tl.trans(k_rot_odd)
    )
    scores = scores * sm_scale
    probs = tl.exp(scores - tl.max(scores, 1)[:, None])
    denom = tl.sum(probs, 1)[:, None]
    values = tl.load(row + 2 * H * D + d)
    out = tl.dot(probs.to(tl.bfloat16), values) / denom
    tl.store(out_ptr + (b * T + t) * (H * D) + h * D + d, out.to(tl.bfloat16))


@triton.jit
def _scale(x, scale):
    """``x`` (bf16 values in float32) times the bf16-rounded ``scale``, in bf16."""
    return (x * scale.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)


@torch.library.custom_op("flashmap_sglang::block_self_attention", mutates_args=())
def block_self_attention(
    qkv: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Self-attention within each draft block, from its projected ``qkv
    (blocks, T, 3, heads, head_dim)``: q/k RMSNorms with float32 weights, RoPE
    by ``freqs (blocks, T, head_dim / 2, [cos, sin])``, then unmasked softmax
    attention. Returns ``(blocks, T, heads, head_dim)``."""
    blocks, block_size, _, heads, head_dim = qkv.shape
    assert qkv.is_contiguous() and freqs.is_contiguous()
    assert freqs.dtype == torch.float32
    out = torch.empty(
        (blocks, block_size, heads, head_dim), dtype=qkv.dtype, device=qkv.device
    )
    _block_self_attention_kernel[(blocks * heads,)](
        qkv,
        q_norm,
        k_norm,
        freqs,
        out,
        eps,
        head_dim**-0.5,
        T=block_size,
        H=heads,
        D=head_dim,
    )
    return out


@block_self_attention.register_fake
def _(qkv, q_norm, k_norm, freqs, eps):
    blocks, block_size, _, heads, head_dim = qkv.shape
    return qkv.new_empty((blocks, block_size, heads, head_dim))


@triton.jit
def _rms_norm_kernel(x_ptr, weight_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, 0) / n_cols + eps)
    scale = tl.load(weight_ptr + cols, mask=mask, other=0.0) * rms
    scale = scale.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (x * scale).to(tl.bfloat16), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm of a contiguous bf16 ``x (..., C)`` with a float32 ``weight``,
    in one launch."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert weight.dtype == torch.float32
    out = torch.empty_like(x)
    n_cols = x.shape[-1]
    if x.numel():
        _rms_norm_kernel[(x.numel() // n_cols,)](
            x, weight, out, n_cols, eps, BLOCK=triton.next_power_of_2(n_cols)
        )
    return out


@triton.jit
def _rms_norm_rope_kernel(
    x_ptr, weight_ptr, freqs_ptr, out_ptr, eps, H: tl.constexpr, D: tl.constexpr
):
    # One program per (token, head) of a contiguous (tokens, H, D) input.
    token = tl.program_id(0) // H
    head = tl.program_id(0) % H
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    row = x_ptr + (token * H + head) * D
    even = tl.load(row + 2 * i).to(tl.float32)
    odd = tl.load(row + 2 * i + 1).to(tl.float32)
    rms = tl.rsqrt((tl.sum(even * even, 0) + tl.sum(odd * odd, 0)) / D + eps)
    even = _scale(even, tl.load(weight_ptr + 2 * i) * rms)
    odd = _scale(odd, tl.load(weight_ptr + 2 * i + 1) * rms)
    cos = tl.load(freqs_ptr + token * D + 2 * i)
    sin = tl.load(freqs_ptr + token * D + 2 * i + 1)
    out = out_ptr + (token * H + head) * D
    tl.store(out + 2 * i, (even * cos - odd * sin).to(tl.bfloat16))
    tl.store(out + 2 * i + 1, (odd * cos + even * sin).to(tl.bfloat16))


@torch.library.custom_op("flashmap_sglang::rms_norm_rope", mutates_args=())
def rms_norm_rope(
    x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float
) -> torch.Tensor:
    """RoPE of RMSNorm(x) for ``x (..., T, heads, head_dim)`` and ``freqs (...,
    T, head_dim / 2, [cos, sin])``, in one launch."""
    *_, heads, head_dim = x.shape
    assert x.is_contiguous() and freqs.is_contiguous()
    assert freqs.dtype == torch.float32
    out = torch.empty_like(x)
    _rms_norm_rope_kernel[(x.numel() // head_dim,)](
        x, weight, freqs, out, eps, H=heads, D=head_dim
    )
    return out


@rms_norm_rope.register_fake
def _(x, weight, freqs, eps):
    return torch.empty_like(x)


__all__ = ["block_self_attention", "rms_norm", "rms_norm_rope"]


@triton.jit
def _sample_tiles_kernel(
    logits_ptr,
    temperatures_ptr,
    greedy_ptr,
    seed_ptr,
    best_ptr,
    best_index_ptr,
    V,
    n_tiles,
    BLOCK_V: tl.constexpr,
):
    # One program per (row, tile of the vocabulary): the tile's best key and
    # its first index. A greedy row's key is its logit; a sampled row's is its
    # logit over the temperature plus Gumbel noise from in-kernel Philox
    # (Gumbel-max: the argmax is a sample of softmax(logits / T)).
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    key = tl.load(logits_ptr + row * V + offs, mask=mask, other=float("-inf"))
    key = key.to(tl.float32)
    if tl.load(greedy_ptr + row) == 0:
        uniform = tl.rand(tl.load(seed_ptr), row * V + offs)
        key = key / tl.load(temperatures_ptr + row) - tl.log(-tl.log(uniform))
    key = tl.where(mask, key, float("-inf"))
    best = tl.max(key, axis=0)
    index = tl.min(tl.where(key == best, offs, V), axis=0)
    tl.store(best_ptr + row * n_tiles + tile, best)
    tl.store(best_index_ptr + row * n_tiles + tile, index)


def sample_rows(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy: torch.Tensor,
    seed: torch.Tensor,
    block_v: int = 4096,
) -> torch.Tensor:
    """A token per row of contiguous ``logits (rows, V)``: the argmax for
    ``greedy`` rows, else a sample of ``softmax(logits / temperature)``, with
    Philox noise seeded by the int64 scalar tensor ``seed`` (advance it between
    calls, in-graph, for fresh noise). Reads the logits once; no noise buffer."""
    rows, vocab = logits.shape
    assert logits.is_contiguous()
    n_tiles = triton.cdiv(vocab, block_v)
    best = torch.empty((rows, n_tiles), dtype=torch.float32, device=logits.device)
    best_index = torch.empty((rows, n_tiles), dtype=torch.int64, device=logits.device)
    _sample_tiles_kernel[(rows, n_tiles)](
        logits, temperatures, greedy, seed, best, best_index, vocab, n_tiles,
        BLOCK_V=block_v,
    )  # fmt: skip
    return best_index.gather(1, best.argmax(dim=1, keepdim=True)).squeeze(1)


__all__ += ["sample_rows"]


# The largest e4m3 magnitude.
_FP8_MAX = 448.0


def quantize_weight_fp8(weight: torch.Tensor):
    """``weight (N, K)`` -> ``(e4m3 weight (N, K), float32 scale (1, N))``, one
    scale per output channel."""
    scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
    quantized = (weight.float() / scale).to(torch.float8_e4m3fn)
    return quantized, scale.t().contiguous()


def fp8_linear(x, weight, scale, bias=None) -> torch.Tensor:
    """``x @ W^T + b`` with ``W`` from :func:`quantize_weight_fp8` and ``x``
    quantized per row on the fly (dynamic per-token scales): one FP8 GEMM with
    row-wise scales (Hopper), bf16 out. Elementwise parts fuse under compile."""
    rows = x.reshape(-1, x.shape[-1])
    x_scale = rows.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
    quantized = (rows.float() / x_scale).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(
        quantized, weight.t(), scale_a=x_scale, scale_b=scale, out_dtype=torch.bfloat16
    )
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], -1)


__all__ += ["fp8_linear", "quantize_weight_fp8"]


@triton.jit
def _kv_norm_rope_write_fp8_kernel(
    kv_ptr,
    meta_ptr,
    knw_ptr,
    cos_sin_ptr,
    pos_ptr,
    loc_ptr,
    commit_lens_ptr,
    locs_row_width,
    k_scale,
    v_scale,
    KV: tl.constexpr,
    D: tl.constexpr,
    NH: tl.constexpr,
    L: tl.constexpr,
    EPS: tl.constexpr,
    HAS_COMMIT_LENS: tl.constexpr,
):
    # sglang's fused_kv_norm_rope_write (interleaved RoPE), storing e4m3 with
    # static per-tensor scales instead of bf16.
    t = tl.program_id(0).to(tl.int64)
    layer = tl.program_id(1).to(tl.int64)
    if HAS_COMMIT_LENS:
        row_b = t // locs_row_width
        if t - row_b * locs_row_width >= tl.load(commit_lens_ptr + row_b).to(tl.int64):
            return
    loc = tl.load(loc_ptr + t).to(tl.int64)
    if loc < 0:
        return
    pos = tl.load(pos_ptr + t).to(tl.int64)
    HALF: tl.constexpr = D // 2
    half = tl.arange(0, HALF)
    d = tl.arange(0, D)
    even, odd = 2 * half, 2 * half + 1
    cos = tl.load(cos_sin_ptr + pos * D + half).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * D + HALF + half).to(tl.float32)
    w_even = tl.load(knw_ptr + layer * D + even).to(tl.float32)
    w_odd = tl.load(knw_ptr + layer * D + odd).to(tl.float32)
    k_buf = tl.load(meta_ptr + layer * 4 + 0).to(tl.pointer_type(tl.float8e4nv))
    v_buf = tl.load(meta_ptr + layer * 4 + 1).to(tl.pointer_type(tl.float8e4nv))
    k_stride = tl.load(meta_ptr + layer * 4 + 2)
    v_stride = tl.load(meta_ptr + layer * 4 + 3)
    row = kv_ptr + t * (L * 2 * KV) + layer * (2 * KV)
    for h in tl.static_range(NH):
        k = tl.load(row + h * D + d).to(tl.float32)
        inv = 1.0 / tl.sqrt(tl.sum(k * k, 0) / D + EPS)
        k1 = (tl.load(row + h * D + even).to(tl.float32) * inv * w_even).to(tl.bfloat16)
        k2 = (tl.load(row + h * D + odd).to(tl.float32) * inv * w_odd).to(tl.bfloat16)
        k1, k2 = k1.to(tl.float32), k2.to(tl.float32)
        o1 = (k1 * cos - k2 * sin) / k_scale
        o2 = (k2 * cos + k1 * sin) / k_scale
        tl.store(k_buf + loc * k_stride + h * D + even, o1.to(tl.float8e4nv))
        tl.store(k_buf + loc * k_stride + h * D + odd, o2.to(tl.float8e4nv))
        v = tl.load(row + KV + h * D + d).to(tl.float32) / v_scale
        tl.store(v_buf + loc * v_stride + h * D + d, v.to(tl.float8e4nv))


def kv_norm_rope_write_fp8(
    kv, meta, k_norm_weights, cos_sin, positions, locs, num_layers, kv_size,
    head_dim, eps, *, k_scale=1.0, v_scale=1.0, commit_lens=None, locs_row_width=None,
):  # fmt: skip
    """``fused_kv_norm_rope_write`` (same arguments; interleaved RoPE) for an
    e4m3 pool: keys k-normed and rotated, both divided by their static scale."""
    if kv.shape[0] == 0:
        return
    has_commit = commit_lens is not None
    _kv_norm_rope_write_fp8_kernel[(kv.shape[0], num_layers)](
        kv, meta, k_norm_weights, cos_sin,
        positions.to(torch.int64).contiguous(), locs.to(torch.int64).contiguous(),
        commit_lens.contiguous() if has_commit else locs, locs_row_width or 1,
        k_scale, v_scale,
        KV=kv_size, D=head_dim, NH=kv_size // head_dim, L=num_layers, EPS=eps,
        HAS_COMMIT_LENS=has_commit,
    )  # fmt: skip


__all__ += ["kv_norm_rope_write_fp8"]


@triton.jit
def _tree_commit_kernel(
    accept_ptr,  # (bs, D1) int32: global node ids of the accepted path, root first; -1 past it
    num_correct_ptr,  # (bs,) int32: accepted drafts
    window_ptr,  # (bs, W) int64: the verify window's KV slots, node order
    predict_ptr,  # (bs * W,) int32: the target's token after each node on the path
    req_pool_ptr,  # (bs,) int64
    prefix_ptr,  # (bs,) int64: committed length before this step
    req_to_token_ptr,  # (rows, row_stride) int32
    row_stride,
    out_tokens_ptr,  # (bs, D1) int64
    bonus_ptr,  # (bs,) int64
    commit_ptr,  # (bs,) int32
    new_seq_lens_ptr,  # (bs,) int64
    inject_loc_ptr,  # (bs * W,) int64
    W: tl.constexpr,
    D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per request: the tree path's commit in one launch.
    b = tl.program_id(0)
    n = tl.load(num_correct_ptr + b)
    j = tl.arange(0, BLOCK)  # path positions
    i = tl.arange(0, BLOCK)  # tree nodes
    on_path_j = j <= n
    node = tl.load(accept_ptr + b * D1 + j, mask=(j < D1) & on_path_j, other=0) - b * W
    node = tl.where(on_path_j & (j < D1), node, -1)
    # Which nodes the path takes (node i at path position j).
    hit = (node[:, None] == i[None, :]) & (node[:, None] >= 0)
    on_path = tl.sum(hit.to(tl.int32), axis=0) > 0
    in_tree = i < W
    window = tl.load(window_ptr + b * W + i, mask=in_tree, other=0)
    # Reorder the window's req_to_token entries: the path first (where its KV
    # is), then every other node in index order.
    off_path = (on_path == 0) & in_tree
    rank = tl.cumsum(off_path.to(tl.int32), axis=0) - 1
    row = tl.load(req_pool_ptr + b)
    prefix = tl.load(prefix_ptr + b)
    base = req_to_token_ptr + row * row_stride + prefix
    path_slot = tl.load(window_ptr + b * W + node, mask=node >= 0, other=0)
    tl.store(base + j, path_slot.to(tl.int32), mask=node >= 0)
    tl.store(base + n + 1 + rank, window.to(tl.int32), mask=off_path)
    # The path's tokens (the drafts, then the target's next one) and the bonus.
    token = tl.load(predict_ptr + b * W + node, mask=node >= 0, other=0).to(tl.int64)
    tl.store(out_tokens_ptr + b * D1 + j, token, mask=j < D1)
    tl.store(bonus_ptr + b, tl.sum(tl.where(j == n, token, 0), axis=0))
    tl.store(commit_ptr + b, n + 1)
    tl.store(new_seq_lens_ptr + b, prefix + n + 1)
    # Draft context goes where the path's KV is; other rows to dummy slot 0.
    tl.store(inject_loc_ptr + b * W + i, tl.where(on_path, window, 0), mask=in_tree)


def tree_commit(
    accept_index,
    num_correct,
    window,
    predict,
    req_pool_indices,
    prefix_lens,
    req_to_token,
):
    """The tree path's commit after verify_tree_greedy, one launch for the batch.
    Reorders each request's verify window in req_to_token so the accepted path
    comes first (no KV moves) and returns (out_tokens (bs, D1) int64, bonus,
    commit_lens int32, new_seq_lens, inject_locs (bs * W) int64)."""
    bs, d1 = accept_index.shape
    width = window.shape[1]
    device = accept_index.device
    out_tokens = torch.empty((bs, d1), dtype=torch.int64, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    commit = torch.empty((bs,), dtype=torch.int32, device=device)
    new_seq_lens = torch.empty((bs,), dtype=prefix_lens.dtype, device=device)
    inject_locs = torch.empty((bs * width,), dtype=torch.int64, device=device)
    _tree_commit_kernel[(bs,)](
        accept_index,
        num_correct,
        window,
        predict,
        req_pool_indices,
        prefix_lens,
        req_to_token,
        req_to_token.stride(0),
        out_tokens,
        bonus,
        commit,
        new_seq_lens,
        inject_locs,
        W=width,
        D1=d1,
        BLOCK=triton.next_power_of_2(max(width, d1)),
    )
    return out_tokens, bonus, commit, new_seq_lens, inject_locs


__all__ += ["tree_commit"]
