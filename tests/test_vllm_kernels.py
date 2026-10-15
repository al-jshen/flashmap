"""GPU parity tests for FlashMap's vLLM Triton kernels against PyTorch."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from flashmap.vllm.kernels import (
    block_self_attention,
    rms_norm,
    rms_norm_rope,
    rope_table,
    scatter_rows,
    seeded_normal,
    write_context_kv,
)

DEVICE = "cuda"


def _rms_ref(x, weight, eps):
    scale = weight.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * scale.to(torch.bfloat16).float()).to(torch.bfloat16)


def _rope_ref(x, cos_sin, positions):
    freqs = cos_sin[positions]  # (tokens, D/2, 2)
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cos, sin = freqs[:, None, :, 0], freqs[:, None, :, 1]
    out = torch.stack(
        [
            pairs[..., 0] * cos - pairs[..., 1] * sin,
            pairs[..., 1] * cos + pairs[..., 0] * sin,
        ],
        dim=-1,
    )
    return out.flatten(-2).to(torch.bfloat16)


def _bf16(*shape, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(*shape, generator=generator, device=DEVICE).to(torch.bfloat16)


def test_rms_norm_strided():
    x = _bf16(7, 2, 96)[:, 1]  # strided rows
    weight = 1 + 0.1 * _bf16(96, seed=1)
    torch.testing.assert_close(rms_norm(x, weight, 1e-6), _rms_ref(x, weight, 1e-6))


def test_rms_norm_rope_strided_positions():
    heads, head_dim = 4, 128
    x = _bf16(5, heads, head_dim)
    weight = 1 + 0.1 * _bf16(head_dim, seed=1)
    cos_sin = rope_table(64, head_dim, 10000.0, DEVICE)
    positions = torch.arange(0, 50, device=DEVICE)[3::10]  # strided, odd offset
    expected = _rope_ref(_rms_ref(x, weight, 1e-5), cos_sin, positions)
    actual = rms_norm_rope(x, weight, cos_sin, positions, 1e-5)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_block_self_attention():
    blocks, block, heads, head_dim = 3, 16, 2, 128
    qkv = _bf16(blocks, block, 3, heads, head_dim)
    q_norm = 1 + 0.1 * _bf16(head_dim, seed=1)
    k_norm = 1 + 0.1 * _bf16(head_dim, seed=2)
    cos_sin = rope_table(128, head_dim, 10000.0, DEVICE)
    positions = (torch.arange(blocks * block, device=DEVICE) + 5).view(blocks, block)
    positions = positions.flatten()
    flat = qkv.view(blocks * block, 3, heads, head_dim)
    q = _rope_ref(_rms_ref(flat[:, 0], q_norm, 1e-5), cos_sin, positions)
    k = _rope_ref(_rms_ref(flat[:, 1], k_norm, 1e-5), cos_sin, positions)
    v = flat[:, 2]

    def blockwise(t):
        return t.view(blocks, block, heads, head_dim).transpose(1, 2)

    expected = torch.nn.functional.scaled_dot_product_attention(
        blockwise(q).float(), blockwise(k).float(), blockwise(v).float()
    )
    expected = expected.transpose(1, 2).reshape(blocks * block, heads * head_dim)
    actual = block_self_attention(qkv, q_norm, k_norm, cos_sin, positions, 1e-5)
    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=3e-2)


def test_write_context_kv():
    layers, heads, head_dim, block_size, num_blocks = 2, 2, 128, 4, 6
    tokens = 5
    kv = _bf16(tokens, layers, 2, heads, head_dim)
    positions = torch.tensor([0, 9, 3, 17, 2], device=DEVICE)
    slots = torch.tensor([1, 6, -1, 13, 22], device=DEVICE)
    k_norm = (1 + 0.1 * _bf16(layers, head_dim, seed=3)).float()
    cos_sin = rope_table(32, head_dim, 10000.0, DEVICE)
    # vLLM FlashAttention layout: (blocks, heads, block_size, 2 * head_dim).
    caches = [
        torch.zeros(
            num_blocks,
            heads,
            block_size,
            2 * head_dim,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        for _ in range(layers)
    ]
    key, value = caches[0].transpose(1, 2).split(head_dim, dim=-1)
    strides = (key.stride(0), key.stride(2), key.stride(1), head_dim)
    ptrs = torch.tensor([c.data_ptr() for c in caches], device=DEVICE)
    write_context_kv(
        kv, positions, slots, k_norm, cos_sin, ptrs, strides, block_size, 1e-5
    )
    for layer, cache in enumerate(caches):
        for i in range(tokens):
            slot = int(slots[i])
            if slot < 0:
                continue
            entry = cache[slot // block_size, :, slot % block_size]  # (heads, 2D)
            expected_k = _rope_ref(
                _rms_ref(kv[i, layer, 0][None], k_norm[layer], 1e-5),
                cos_sin,
                positions[i : i + 1],
            )[0]
            torch.testing.assert_close(
                entry[:, :head_dim], expected_k, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(entry[:, head_dim:], kv[i, layer, 1])
    assert int(caches[0].count_nonzero()) == 4 * heads * 2 * head_dim  # slot -1 skipped


def test_seeded_normal_is_keyed_by_seed_and_position():
    seeds = torch.tensor([11, 22, 11], device=DEVICE)
    rows = torch.tensor([0, 1, 2, -1], dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([5, 5, 5, 5], device=DEVICE)
    z = seeded_normal(seeds, rows, positions, 4096)
    torch.testing.assert_close(z[0], z[2])  # same seed and position
    assert not torch.equal(z[0], z[1])
    again = seeded_normal(seeds, rows, positions + 1, 4096)
    assert not torch.equal(z[0], again[0])
    assert abs(float(z.mean())) < 0.05
    assert abs(float(z.std()) - 1) < 0.05


def test_scatter_rows():
    dst = torch.full((3, 2, 10), float("-inf"), device=DEVICE)
    src = torch.arange(8, dtype=torch.float32, device=DEVICE).view(2, 4)
    req = torch.tensor([2, -1], dtype=torch.int32, device=DEVICE)
    col = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
    ids = torch.tensor([0, 3, 5, 9], device=DEVICE)
    scatter_rows(src, req, col, ids, dst)
    assert torch.equal(dst[2, 1, ids], src[0])
    assert torch.isinf(dst).sum() == 3 * 2 * 10 - 4


def test_write_context_kv_fp8():
    layers, heads, head_dim, block_size, num_blocks = 2, 2, 128, 4, 3
    kv = _bf16(3, layers, 2, heads, head_dim)
    positions = torch.tensor([0, 5, 7], device=DEVICE)
    slots = torch.tensor([0, 5, 11], device=DEVICE)
    k_norm = (1 + 0.1 * _bf16(layers, head_dim, seed=3)).float()
    cos_sin = rope_table(16, head_dim, 10000.0, DEVICE)
    scales = torch.tensor([[0.5, 2.0], [1.0, 1.0]], device=DEVICE)
    caches = [
        torch.zeros(
            num_blocks,
            heads,
            block_size,
            2 * head_dim,
            device=DEVICE,
            dtype=torch.uint8,
        )
        for _ in range(layers)
    ]
    key, _ = caches[0].transpose(1, 2).split(head_dim, dim=-1)
    strides = (key.stride(0), key.stride(2), key.stride(1), head_dim)
    ptrs = torch.tensor([c.data_ptr() for c in caches], device=DEVICE)
    write_context_kv(
        kv, positions, slots, k_norm, cos_sin, ptrs, strides, block_size, 1e-5, scales
    )
    for layer, cache in enumerate(caches):
        cache = cache.view(torch.float8_e4m3fn)
        for i, slot in enumerate(slots.tolist()):
            entry = cache[slot // block_size, :, slot % block_size].float()
            expected_k = _rope_ref(
                _rms_ref(kv[i, layer, 0][None], k_norm[layer], 1e-5),
                cos_sin,
                positions[i : i + 1],
            )[0].float()
            k_scale, v_scale = scales[layer].tolist()
            torch.testing.assert_close(
                entry[:, :head_dim] * k_scale, expected_k, atol=0.1, rtol=0.07
            )
            torch.testing.assert_close(
                entry[:, head_dim:] * v_scale,
                kv[i, layer, 1].float(),
                atol=0.1,
                rtol=0.07,
            )
