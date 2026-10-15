"""Load-time folds of FlashMap's one-jump draft (pure PyTorch, no vLLM).

``fold_noise``: a one-jump draft embeds its starting noise ``x0 ~ N(0, I_V)`` as
``x_embed(noise_norm(x0 @ E))`` (plus a constant). ``x0 @ E`` is Gaussian with
covariance ``E^T E = Q Lambda Q^T``, so it is drawn as ``z @ A`` from
``z ~ N(0, I_H)`` with ``A = sqrt(Lambda) Q^T``; since ``A A^T = Lambda`` the
noise norm's mean square is ``sum(lambda z^2) / H``, elementwise in ``z``, and
``A``, the noise norm's weight and ``x_embed``'s weight fold into one matrix:

    x_embed(noise_norm(z @ A)) = rsqrt(z^2 @ lambda / H + eps) * (z @ M^T) + b,
    M = W_x diag(w) A^T.

``fold_head``: the DiT's output head (``D -> H`` with bias) and the target's LM
head (``H -> V``) have no nonlinearity between them, so the draft reads one
``(V, D)`` matrix and bias instead of both.
"""

import math

import torch

__all__ = [
    "fold_head",
    "fold_noise",
    "noise_factor",
    "quantize_fp8_per_channel",
    "split_argmax",
    "timestep_embedding",
]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """Sinusoidal embedding of ``t``: ``cat([cos(t f), sin(t f)])`` in float32."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


@torch.no_grad()
def noise_factor(
    embedding: torch.Tensor, chunk: int = 16384
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(A, lambda)`` in float64 for ``embedding (V, H)``: ``A = sqrt(Lambda)
    Q^T`` with ``E^T E = Q Lambda Q^T``, so ``A^T A = E^T E`` and ``A A^T =
    Lambda``."""
    hidden = embedding.shape[1]
    gram = torch.zeros(hidden, hidden, device=embedding.device, dtype=torch.float32)
    for rows in embedding.split(chunk):
        gram += rows.float().T @ rows.float()
    eigvals, eigvecs = torch.linalg.eigh(gram.double())
    eigvals = eigvals.clamp_min(0)
    return eigvals.sqrt()[:, None] * eigvecs.T, eigvals


@torch.no_grad()
def fold_noise(
    embedding: torch.Tensor,
    noise_norm_weight: torch.Tensor,
    x_embed_weight: torch.Tensor,
    chunk: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(M, lambda / H)`` for ``embedding (V, H)``: ``M (D, H)`` in the
    embedding's dtype, as ``F.linear`` takes it, and ``lambda / H (H,)`` in
    float32 (see the module docstring)."""
    factor, eigvals = noise_factor(embedding, chunk)
    weight = noise_norm_weight.double()
    folded = x_embed_weight.double() @ (factor * weight[None, :]).T
    return folded.to(embedding.dtype), (eigvals / embedding.shape[1]).float()


@torch.no_grad()
def fold_head(
    lm_head: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor,
    rows: torch.Tensor | None = None,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lm_head[rows] @ head`` as one ``(len(rows), D)`` weight and bias in the
    LM head's dtype, folded ``chunk`` rows at a time in float32."""
    count = lm_head.shape[0] if rows is None else rows.numel()
    weight = torch.empty(
        count, head_weight.shape[1], dtype=lm_head.dtype, device=lm_head.device
    )
    bias = torch.empty(count, dtype=lm_head.dtype, device=lm_head.device)
    head_weight, head_bias = head_weight.float(), head_bias.float()
    for start in range(0, count, chunk):
        part = (
            lm_head[start : start + chunk]
            if rows is None
            else lm_head[rows[start : start + chunk]]
        ).float()
        weight[start : start + chunk] = (part @ head_weight).to(lm_head.dtype)
        bias[start : start + chunk] = (part @ head_bias).to(lm_head.dtype)
    return weight, bias


@torch.no_grad()
def quantize_fp8_per_channel(
    weight: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(w_q, scale)`` for a ``(N, K)`` weight: ``w_q (K, N)`` in ``dtype`` (a
    column-major view, as scaled-mm kernels take it) and ``scale (N, 1)`` in
    float32 with ``weight ~= (w_q.T * scale)``, one scale per output channel."""
    finfo = torch.finfo(dtype)
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / finfo.max
    w_q = (w / scale).clamp(finfo.min, finfo.max).to(dtype)
    return w_q.t(), scale.contiguous()


def split_argmax(logits: torch.Tensor, chunks: int = 128) -> torch.Tensor:
    """``logits.argmax(-1)`` for ``(rows, vocab)`` logits, reduced over vocabulary
    chunks first so that a few rows still fill the GPU. Ties resolve to the
    first index, as ``argmax`` does."""
    rows, vocab = logits.shape
    while chunks > 1 and vocab % chunks:
        chunks //= 2
    if chunks == 1:
        return logits.argmax(dim=-1)
    values, index = logits.view(rows, chunks, vocab // chunks).max(dim=-1)
    best = values.argmax(dim=-1, keepdim=True)
    return (best * (vocab // chunks) + index.gather(1, best)).squeeze(1)
