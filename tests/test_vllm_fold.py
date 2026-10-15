"""CPU tests for the vLLM plugin's load-time folds (flashmap.vllm.fold)."""

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from flashmap.vllm.fold import (
    fold_head,
    fold_noise,
    noise_factor,
    quantize_fp8_per_channel,
    split_argmax,
)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * weight * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def test_noise_factor_reproduces_embedding_covariance():
    generator = torch.Generator().manual_seed(0)
    embedding = torch.randn(50, 6, generator=generator, dtype=torch.float64)
    factor, eigvals = noise_factor(embedding.float())
    gram = embedding.T @ embedding
    torch.testing.assert_close(factor.T @ factor, gram, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        factor @ factor.T, torch.diag(eigvals), rtol=1e-4, atol=1e-4
    )


def test_fold_noise_matches_noise_norm_then_x_embed():
    generator = torch.Generator().manual_seed(1)
    embedding = torch.randn(40, 8, generator=generator)
    norm_weight = 1 + 0.1 * torch.randn(8, generator=generator)
    x_embed_weight = torch.randn(5, 8, generator=generator)
    eps = 1e-6
    folded, eig_over_h = fold_noise(embedding, norm_weight, x_embed_weight)
    factor, _ = noise_factor(embedding)

    z = torch.randn(7, 8, generator=generator)
    noise = (z.double() @ factor).float()  # x0 @ E, drawn in the eigenbasis
    expected = F.linear(_rms(noise, norm_weight, eps), x_embed_weight)
    rms = torch.rsqrt((z * z) @ eig_over_h + eps)
    actual = rms[:, None] * F.linear(z, folded)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("restricted", [False, True])
def test_fold_head_matches_both_heads(restricted):
    generator = torch.Generator().manual_seed(2)
    lm_head = torch.randn(30, 6, generator=generator)
    head_weight = torch.randn(6, 4, generator=generator)
    head_bias = torch.randn(6, generator=generator)
    rows = torch.tensor([1, 4, 7, 29]) if restricted else None
    weight, bias = fold_head(lm_head, head_weight, head_bias, rows, chunk=3)

    hidden = torch.randn(5, 4, generator=generator)
    expected = F.linear(F.linear(hidden, head_weight, head_bias), lm_head)
    if restricted:
        expected = expected[:, rows]
    torch.testing.assert_close(
        F.linear(hidden, weight, bias), expected, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("vocab", [1024, 151936, 1000, 7])
def test_split_argmax_matches_argmax(vocab):
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(9, vocab, generator=generator).to(torch.bfloat16)
    # Ties resolve to the first index, as argmax does.
    logits[0, :] = 0
    logits[1, vocab // 2 :] = logits[1].max()
    torch.testing.assert_close(split_argmax(logits), logits.argmax(dim=-1))


def test_quantize_fp8_per_channel():
    generator = torch.Generator().manual_seed(4)
    weight = (
        torch.randn(32, 64, generator=generator) * torch.linspace(0.01, 10, 32)[:, None]
    )
    w_q, scale = quantize_fp8_per_channel(weight)
    assert w_q.shape == (64, 32)
    assert w_q.dtype == torch.float8_e4m3fn
    assert scale.shape == (32, 1)
    restored = w_q.t().float() * scale
    # e4m3 keeps 3 mantissa bits: a relative error of at most 2^-4 per entry.
    torch.testing.assert_close(
        restored, weight, rtol=0.07, atol=1e-3 * scale.max().item()
    )
