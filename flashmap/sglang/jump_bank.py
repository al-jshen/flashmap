# coding=utf-8
"""Multi-jump FlashMap drafting without vocabulary-sized matmuls.

A flow map's jump ``n >= 1`` starts from the vocabulary-sized state

    x_n = alpha_n z + sum_m beta_nm delta_m,   z ~ N(0, T^2 I_V),

(``z`` the draft's starting noise at temperature ``T``, ``delta_m`` the earlier
jumps' predictions, ``alpha_n = 1 - t_n`` and ``beta_nm`` from the jump update
``x <- ((1 - t) x + (t - s) delta) / (1 - s)``) and embeds it as

    state_norm(softmax(e_n x_n) @ E) + gate * noise_norm(x_n @ E),
    e_n = t_n / (1 - t_n)^2.

Done literally that is two ``(V, H)`` matmuls per jump, plus a third for the
first jump's noise and a full-vocabulary head for each ``delta_m``. Off the
predictions' support ``S`` (their top-k tokens) the posterior's logits are
``a_n z_v`` with ``a_n = t_n / (1 - t_n)``; on it ``a_n z_v + c_v`` with ``c_v =
e_n sum_m beta_nm delta_m,v >= 0``. So, exactly,

    softmax(e_n x_n) @ E = (W + sum_{v in S} r_v E_v) / (Z + sum_{v in S} r_v),
        W = sum_v e^{a_n z_v} E_v,  Z = sum_v e^{a_n z_v},  r_v = e^{a_n z_v} (e^{c_v} - 1),
    x_n @ E = alpha_n z @ E + sum_m beta_nm delta_m @ E.

Everything over the whole vocabulary depends on ``z`` alone: ``z @ E`` and, per
jump, ``W / Z`` and ``log Z``. :class:`JumpNoiseBank` precomputes them for ``M``
draws of ``z``; a draft takes a bank row per slot and then reads only the
embedding rows of the support. The approximations:

1. ``delta_m`` comes from the draft-vocabulary head, cut to its top-k;
2. on ``S``, ``z_v`` is drawn afresh (a hash of the slot's seed and the token,
   so a token keeps its value from jump to jump) instead of read from the
   row, whose sums keep their own term for ``v`` (one of ``V`` terms);
3. the bank is finite: a slot's noise is one of ``M`` i.i.d. draws.

Banked top-16 2-jump drafts match exact 2-jump drafts in accepted length on
every domain we evaluated.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

_MASK32 = 0xFFFFFFFF


def _hash32(x: torch.Tensor) -> torch.Tensor:
    """An integer hash of int64 values in [0, 2^32) (two multiply-xorshift
    rounds; every product stays below 2^59, so int64 arithmetic is exact)."""
    x = x ^ (x >> 16)
    x = (x * 0x45D9F3B) & _MASK32
    x = x ^ (x >> 16)
    x = (x * 0x45D9F3B) & _MASK32
    return x ^ (x >> 16)


def token_normals(seeds: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """A standard normal per (slot seed, token id): ``seeds (R,)`` and ``ids (R,
    L)`` int64 -> float32 ``(R, L)``, a function of the pair only (Box-Muller on
    two hashed uniforms)."""
    base = _hash32((ids & _MASK32) ^ 0x9E3779B9)
    h1 = _hash32(base ^ (seeds.unsqueeze(-1) & _MASK32))
    h2 = _hash32(h1 ^ 0x68E31DA4)
    u1 = (h1.float() + 1.0) / 4294967297.0
    u2 = h2.float() / 4294967296.0
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)


def jump_coefficients(grid):
    """Per jump ``n``: ``(alpha_n, [beta_n1, ..., beta_nn])``, the state's
    coefficients of ``z`` and of each earlier prediction."""
    alpha, betas, out = 1.0, [], [(1.0, [])]
    for n in range(len(grid) - 2):
        start, end = float(grid[n]), float(grid[n + 1])
        keep = (1.0 - end) / (1.0 - start)
        alpha = alpha * keep
        betas = [b * keep for b in betas] + [(end - start) / (1.0 - start)]
        out.append((alpha, list(betas)))
    return out


def top_predictions(
    logits: torch.Tensor, topk: int, hot_ids: Optional[torch.Tensor] = None
):
    """A jump's prediction over ``logits``' vocabulary (all tokens, or the sorted
    ``hot_ids`` the logits cover), cut to its ``topk``: ``(token ids, probs)``."""
    probs, index = logits.float().softmax(dim=-1).topk(topk, dim=-1)
    return (index if hot_ids is None else hot_ids[index]), probs


class JumpNoiseBank:
    """``size`` draws of a multi-jump draft's starting noise with what its jumps
    need from them over the whole vocabulary (see the module docstring).

    ``draft`` is the sglang :class:`FlashMapDraftModel`; its ``embedding`` (the
    target's token embedding), ``state_norm``/``noise_norm`` and
    ``noise_gate_at`` are read."""

    def __init__(
        self,
        draft,
        grid,
        *,
        size: int = 8192,
        temperature: float = 1.0,
        seed: int = 0,
        chunk: int = 128,
    ):
        embedding = draft.embedding
        device, dtype = embedding.device, embedding.dtype
        steps = len(grid) - 1
        self.grid = [float(t) for t in grid]
        self.size, self.temperature, self.steps = int(size), float(temperature), steps
        self.coefficients = jump_coefficients(self.grid)
        vocab, hidden = embedding.shape
        times = self.grid[1:steps]
        self.evidence = [t / (1.0 - t) ** 2 for t in times]  # e_n
        # a_n = e_n alpha_n (= t_n / (1 - t_n)); z carries the temperature.
        self.slopes = [
            e * self.coefficients[n + 1][0] for n, e in enumerate(self.evidence)
        ]
        self.noise_embed = torch.empty(self.size, hidden, dtype=dtype, device=device)
        self.mean_embed = torch.empty(
            steps - 1, self.size, hidden, dtype=dtype, device=device
        )
        self.log_mass = torch.empty(
            steps - 1, self.size, dtype=torch.float32, device=device
        )
        generator = torch.Generator(device=device).manual_seed(int(seed))
        with torch.no_grad():
            for start in range(0, self.size, chunk):
                rows = min(chunk, self.size - start)
                z = self.temperature * torch.randn(
                    rows, vocab, generator=generator, device=device
                )
                self.noise_embed[start : start + rows] = z.to(dtype) @ embedding
                for n, slope in enumerate(self.slopes):
                    logits = slope * z
                    self.log_mass[n, start : start + rows] = logits.logsumexp(-1)
                    self.mean_embed[n, start : start + rows] = (
                        logits.softmax(dim=-1).to(dtype) @ embedding
                    )

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.noise_embed, self.mean_embed, self.log_mass)
        )

    def draw(self, slots: int, device):
        """A bank row and a seed per slot."""
        rows = torch.randint(self.size, (slots,), device=device)
        seeds = torch.randint(2**31 - 1, (slots,), device=device)
        return rows, seeds

    def first_input(self, draft, rows: torch.Tensor, uniform: torch.Tensor):
        """The embedded starting state of slots ``rows`` at ``t = 0``: the
        uniform posterior's embedding ``uniform (H,)`` plus the gated noise."""
        gate = draft.noise_gate_at(self.grid[0], self.grid[1])
        out = uniform.expand(rows.numel(), -1)
        return out + gate * draft.noise_norm(self.noise_embed[rows])

    def jump_input(self, draft, n: int, rows, seeds, supports) -> torch.Tensor:
        """The embedded state of jump ``n >= 1`` for slots ``rows (R,)``;
        ``supports``: per earlier prediction ``m``, its top tokens ``(ids (R, k),
        probs (R, k))``. Returns ``(R, H)`` in the embedding's dtype."""
        embedding = draft.embedding
        alpha, betas = self.coefficients[n]
        ids = torch.cat([ids for ids, _ in supports], dim=1)  # (R, L)
        # Each slot's contribution to the state x_n at its token: beta_nm delta_m.
        linear = torch.cat(
            [beta * probs.float() for beta, (_, probs) in zip(betas, supports)], dim=1
        )
        same = ids.unsqueeze(-1) == ids.unsqueeze(-2)  # (R, L, L)
        # c_v sums every prediction's weight on v; a token listed twice counts once.
        shift = self.evidence[n - 1] * (same.float() @ linear.unsqueeze(-1)).squeeze(-1)
        width = ids.shape[1]
        earlier = torch.ones(width, width, dtype=torch.bool, device=ids.device).tril(-1)
        first = ~(same & earlier).any(dim=-1)
        z = self.temperature * token_normals(seeds, ids)
        log_mass = self.log_mass[n - 1][rows].unsqueeze(-1)
        # r_v / Z
        weight = (
            torch.exp(self.slopes[n - 1] * z - log_mass) * torch.expm1(shift) * first
        )
        # Both sums over the support's embedding rows in one batched matmul
        # (bf16 operands, float32 accumulation): (R, 2, L) @ (R, L, H).
        sums = torch.bmm(
            torch.stack([weight, linear], dim=1).to(embedding.dtype), embedding[ids]
        ).float()
        posterior = (self.mean_embed[n - 1][rows].float() + sums[:, 0]) / (
            1.0 + weight.sum(dim=1, keepdim=True)
        )
        out = draft.state_norm(posterior.to(embedding.dtype))
        gate = draft.noise_gate_at(self.grid[n], self.grid[n + 1])
        noise = alpha * self.noise_embed[rows].float() + sums[:, 1]
        return out + gate * draft.noise_norm(noise.to(embedding.dtype))


__all__ = ["JumpNoiseBank", "jump_coefficients", "token_normals", "top_predictions"]
