# coding=utf-8
"""Split-KV attention for speculative verification on sglang's Triton backend.

A verify step attends L query tokens per request (a draft chain or tree plus
its root) to the request's cached prefix and to each other. sglang's Triton
backend runs it as a prefill (extend_attention_fwd): one program per (request,
query head) walks the whole prefix serially, its 128-row query tile holding
only L = 17 real rows. At batch size 1 that is 32 programs on a 132-SM GPU, and
the time grows with the context; a tree's custom mask costs a further ~27%.

Here one program per (request, KV head, row block, split) holds all G query
heads that share the KV head (G * L rows, in blocks of up to 128 for wide
trees), so each prefix key and value is read once per KV head (and row block)
rather than once per query head, and the prefix is split across programs
(as flash-decoding does) when the batch alone cannot fill the GPU. One more
split per (request, KV head) attends the L tokens to each other, under the
tree's mask (sglang's layout: L rows of prefix + L columns per request, the
prefix all visible; or, with ``qlen_mask``, each request's L x L block alone,
EAGLE's QLEN_ONLY layout, which skips writing the prefix columns) or causally.
A second kernel merges the splits by their
log-sum-exp. Only the shapes are read on the host, so it is graph-capturable.

``SplitKVVerify`` takes extend_attention_fwd's arguments and returns False,
having done nothing, for a case it does not handle (the caller then runs
extend_attention_fwd); tree_worker.py installs it.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _verify_partial_kernel(
    q_ptr,
    k_ext_ptr,
    v_ext_ptr,
    k_buf_ptr,
    v_buf_ptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    part_out_ptr,
    part_lse_ptr,
    sm_scale,
    k_scale,
    v_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_kbt,
    stride_kbh,
    stride_vbt,
    stride_vbh,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    M: tl.constexpr,
    RB: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_CHUNK: tl.constexpr,
    USE_MASK: tl.constexpr,
    QLEN_MASK: tl.constexpr,
):
    # Program (b, h * RB + rb, s): request b, KV head h, rows rb * M .. + M;
    # splits s < S cover the prefix, split S the L verify tokens. Row r is
    # query head h * G + r // L at token r % L of the request.
    b = tl.program_id(0)
    h = tl.program_id(1) // RB
    rb = tl.program_id(1) % RB
    s = tl.program_id(2)
    rows = rb * M + tl.arange(0, M)
    row_ok = rows < G * L
    g = rows // L
    t = rows % L
    d = tl.arange(0, BLOCK_D)
    dv = tl.arange(0, BLOCK_DV)
    q = tl.load(
        q_ptr
        + (b * L + t)[:, None] * stride_qt
        + (h * G + g)[:, None] * stride_qh
        + d[None, :],
        mask=row_ok[:, None] & (d[None, :] < D),
        other=0.0,
    )
    kv_start = tl.load(kv_indptr + b)
    prefix = tl.load(kv_indptr + b + 1) - kv_start
    m_i = tl.full([M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, BLOCK_DV], dtype=tl.float32)
    live = s >= 0  # a split that holds keys
    if s < S:
        chunk = tl.cdiv(tl.maximum(tl.cdiv(prefix, S), MIN_CHUNK), BLOCK_N) * BLOCK_N
        lo = s * chunk
        hi = tl.minimum(lo + chunk, prefix)
        live = hi > lo
        for start in range(lo, hi, BLOCK_N):
            n = start + tl.arange(0, BLOCK_N)
            n_ok = n < hi
            loc = tl.load(kv_indices + kv_start + n, mask=n_ok, other=0)
            k = tl.load(
                k_buf_ptr + loc[None, :] * stride_kbt + h * stride_kbh + d[:, None],
                mask=n_ok[None, :] & (d[:, None] < D),
                other=0.0,
            )
            qk = tl.dot(q, k) * (sm_scale * k_scale)
            qk = tl.where(n_ok[None, :], qk, float("-inf"))
            v = tl.load(
                v_buf_ptr + loc[:, None] * stride_vbt + h * stride_vbh + dv[None, :],
                mask=n_ok[:, None] & (dv[None, :] < DV),
                other=0.0,
            )
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        acc = acc * v_scale
    else:
        # Names differ from the prefix branch's: Triton requires a name assigned
        # in both branches of a runtime `if` to have one type.
        j = tl.arange(0, BLOCK_L)
        j_ok = j < L
        k_ext = tl.load(
            k_ext_ptr + (b * L + j)[None, :] * stride_kt + h * stride_kh + d[:, None],
            mask=j_ok[None, :] & (d[:, None] < D),
            other=0.0,
        )
        qk_ext = tl.dot(q, k_ext) * sm_scale
        if USE_MASK:
            if QLEN_MASK:
                # Request b's L x L block, row t for verify token t.
                offs = b * L * L + t[:, None] * L + j[None, :]
            else:
                # Row t of request b's mask: prefix + L entries, the last L for the verify tokens.
                offs = (
                    tl.load(mask_indptr + b)
                    + t[:, None] * (prefix + L)
                    + prefix
                    + j[None, :]
                )
            allowed = (
                tl.load(mask_ptr + offs, mask=row_ok[:, None] & j_ok[None, :], other=0)
                != 0
            )
        else:
            allowed = j[None, :] <= t[:, None]
        allowed = allowed & row_ok[:, None] & j_ok[None, :]
        qk_ext = tl.where(allowed, qk_ext, float("-inf"))
        v_ext = tl.load(
            v_ext_ptr + (b * L + j)[:, None] * stride_vt + h * stride_vh + dv[None, :],
            mask=j_ok[:, None] & (dv[None, :] < DV),
            other=0.0,
        )
        m_i = tl.max(qk_ext, 1)
        p_ext = tl.exp(qk_ext - tl.where(m_i == float("-inf"), 0.0, m_i)[:, None])
        l_i = tl.sum(p_ext, 1)
        acc = tl.dot(p_ext.to(v_ext.dtype), v_ext)
    part = (b * H_KV + h) * (S + 1) + s
    lse = tl.where(l_i > 0, m_i + tl.log(l_i), float("-inf"))
    tl.store(part_lse_ptr + part * (RB * M) + rows, lse, mask=row_ok)
    if live:
        out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(
            part_out_ptr + (part * (RB * M) + rows)[:, None] * BLOCK_DV + dv[None, :],
            out,
            mask=row_ok[:, None],
        )


@triton.jit
def _verify_merge_kernel(
    part_out_ptr,
    part_lse_ptr,
    o_ptr,
    stride_ot,
    stride_oh,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Program (b, query head): the splits' outputs for its L rows (of ROWS per
    # partial), weighted by their share of the softmax mass.
    b = tl.program_id(0)
    head = tl.program_id(1)
    h = head // G
    t = tl.arange(0, BLOCK_L)
    t_ok = t < L
    rows = (head % G) * L + t
    dv = tl.arange(0, BLOCK_DV)
    m = tl.full([BLOCK_L], float("-inf"), dtype=tl.float32)
    den = tl.zeros([BLOCK_L], dtype=tl.float32)
    acc = tl.zeros([BLOCK_L, BLOCK_DV], dtype=tl.float32)
    for s in tl.static_range(S + 1):
        part = (b * H_KV + h) * (S + 1) + s
        lse = tl.load(part_lse_ptr + part * ROWS + rows, mask=t_ok, other=float("-inf"))
        live = lse > float("-inf")
        out = tl.load(
            part_out_ptr + (part * ROWS + rows)[:, None] * BLOCK_DV + dv[None, :],
            mask=live[:, None],
            other=0.0,
        )
        m_new = tl.maximum(m, lse)
        base = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m - base)
        w = tl.exp(lse - base)
        acc = acc * alpha[:, None] + out * w[:, None]
        den = den * alpha + w
        m = m_new
    out = acc / tl.where(den > 0, den, 1.0)[:, None]
    tl.store(
        o_ptr + (b * L + t)[:, None] * stride_ot + head * stride_oh + dv[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=t_ok[:, None] & (dv[None, :] < DV),
    )


def row_blocks(rows: int) -> tuple:
    """(rows per program, programs) covering ``rows`` query rows: blocks of 64
    or 128, whichever pads less (ties to 128, fewer programs re-reading K/V)."""
    best = None
    for size in (128, 64):
        size = min(size, max(16, triton.next_power_of_2(rows)))
        blocks = triton.cdiv(rows, size)
        if best is None or size * blocks < best[0] * best[1]:
            best = (size, blocks)
    return best


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


class SplitKVVerify:
    """extend_attention_fwd for a verify batch (L tokens per request), split
    over the prefix. Scratch for the splits' partial outputs grows with the
    batch, outside graph capture (a capture runs warm-up passes first); a
    replaced buffer is kept, as captured graphs still point at it."""

    MAX_SCRATCH_BYTES = 512 << 20
    # Custom masks arrive as each request's L x L block (EAGLE's QLEN_ONLY) rather
    # than sglang's full prefix + L rows. Whoever builds the masks sets this
    # before graphs are captured; extend_attention_fwd cannot read that layout,
    # so a batch this kernel cannot take then raises instead of falling back.
    qlen_mask = False

    def __init__(self):
        # Splits per (request, KV head): enough programs for `waves` waves of
        # the GPU's SMs, at most MAX; MIN_CHUNK keys at least per split, so
        # short prefixes use fewer of them.
        self.waves = _env_int("FLASHMAP_SPLITKV_WAVES", 2)
        self.max_splits = _env_int("FLASHMAP_SPLITKV_MAX_SPLITS", 16)
        self.min_chunk = _env_int("FLASHMAP_SPLITKV_MIN_CHUNK", 64)
        self.block_n = _env_int("FLASHMAP_SPLITKV_BLOCK_N", 64)
        self.num_warps = _env_int("FLASHMAP_SPLITKV_WARPS", 8)
        self.part_out = None
        self.part_lse = None
        self._retired = []
        self._sms = None

    def splits(self, bs: int, h_kv: int, device) -> int:
        if self._sms is None:
            self._sms = torch.cuda.get_device_properties(device).multi_processor_count
        want = max(1, self.waves * self._sms // (bs * h_kv))
        return min(self.max_splits, 1 << (want.bit_length() - 1))

    def _scratch(self, parts: int, m: int, block_dv: int, device) -> bool:
        if (
            self.part_out is not None
            and self.part_out.shape[0] >= parts
            and self.part_out.shape[1:] == (m, block_dv)
        ):
            return True
        if (
            torch.cuda.is_current_stream_capturing()
            or parts * m * block_dv * 4 > self.MAX_SCRATCH_BYTES
        ):
            return False
        if self.part_out is not None:
            self._retired.append((self.part_out, self.part_lse))
        self.part_out = torch.empty(
            parts, m, block_dv, dtype=torch.float32, device=device
        )
        self.part_lse = torch.empty(parts, m, dtype=torch.float32, device=device)
        return True

    def __call__(
        self,
        q,
        k,
        v,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        k_scale=1.0,
        v_scale=1.0,
        sm_scale=None,
        logit_cap=0.0,
        sliding_window_size=-1,
        sinks=None,
        window_kv_offsets=None,
        xai_temperature_len=-1,
        max_bs=None,
        **unused,
    ) -> bool:
        del window_kv_offsets, max_bs, unused
        handled = self._run(
            q,
            k,
            v,
            o,
            k_buffer,
            v_buffer,
            kv_indptr,
            kv_indices,
            custom_mask,
            is_causal,
            mask_indptr,
            qo_indptr,
            max_len_extend,
            k_scale,
            v_scale,
            sm_scale,
            logit_cap,
            sliding_window_size,
            sinks,
            xai_temperature_len,
        )
        if not handled and custom_mask is not None and self.qlen_mask:
            raise RuntimeError(
                "split-KV verify declined a batch whose mask only it can read (QLEN_ONLY layout)"
            )
        return handled

    def _run(
        self,
        q,
        k,
        v,
        o,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        qo_indptr,
        max_len_extend,
        k_scale,
        v_scale,
        sm_scale,
        logit_cap,
        sliding_window_size,
        sinks,
        xai_temperature_len,
    ) -> bool:
        if (logit_cap or 0) > 0 or sinks is not None or (sliding_window_size or -1) > 0:
            return False
        if (xai_temperature_len or -1) > 0 or (custom_mask is None and not is_causal):
            return False
        if not (
            isinstance(k_scale, (int, float)) and isinstance(v_scale, (int, float))
        ):
            return False
        bs = qo_indptr.shape[0] - 1
        length = int(max_len_extend)
        h_q, d = q.shape[1], q.shape[2]
        h_kv, dv = k_buffer.shape[1], v_buffer.shape[-1]
        if (
            bs < 1
            or q.shape[0] != bs * length
            or h_q % h_kv
            or k_buffer.dtype != q.dtype
        ):
            return False
        group = h_q // h_kv
        m, blocks = row_blocks(group * length)
        k = k.view(-1, h_kv, d)
        v = v.view(-1, h_kv, dv)
        splits = self.splits(bs, h_kv, q.device)
        block_dv = triton.next_power_of_2(dv)
        if not self._scratch(bs * h_kv * (splits + 1), m * blocks, block_dv, q.device):
            return False
        block_l = max(16, triton.next_power_of_2(length))
        sm_scale = sm_scale or d**-0.5
        use_mask = custom_mask is not None
        _verify_partial_kernel[(bs, h_kv * blocks, splits + 1)](
            q,
            k,
            v,
            k_buffer,
            v_buffer,
            kv_indptr,
            kv_indices,
            custom_mask if use_mask else q,
            mask_indptr if use_mask else kv_indptr,
            self.part_out,
            self.part_lse,
            sm_scale,
            float(k_scale),
            float(v_scale),
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            H_KV=h_kv,
            G=group,
            L=length,
            S=splits,
            D=d,
            DV=dv,
            BLOCK_D=triton.next_power_of_2(d),
            BLOCK_DV=block_dv,
            M=m,
            RB=blocks,
            BLOCK_L=block_l,
            BLOCK_N=self.block_n,
            MIN_CHUNK=self.min_chunk,
            USE_MASK=use_mask,
            QLEN_MASK=self.qlen_mask,
            num_warps=self.num_warps,
            num_stages=1,
        )
        o = o.view(-1, h_q, dv)
        _verify_merge_kernel[(bs, h_q)](
            self.part_out,
            self.part_lse,
            o,
            o.stride(0),
            o.stride(1),
            H_KV=h_kv,
            G=group,
            L=length,
            S=splits,
            DV=dv,
            BLOCK_DV=block_dv,
            ROWS=m * blocks,
            BLOCK_L=block_l,
            num_warps=4,
            num_stages=1,
        )
        return True
