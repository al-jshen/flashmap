"""FlashMap: speculative decoding with a one-step flow-map drafter.

FlashMap drafts the ``K`` tokens after the last verified token (the anchor) in one
network evaluation, like DFlash, but samples them jointly rather than one slot at a
time. A block is a point ``x`` in ``R^{K x V}`` (``V`` the target's vocabulary)
moving along the path ``x_s = (1 - s) x0 + s onehot(y)`` from Gaussian noise ``x0``
at ``s = 0`` to the one-hot tokens ``y`` at ``s = 1``. The drafter is a *flow map*
(Flow Map Language Models, arXiv:2602.16813): a network that jumps along this path
from time ``s`` to time ``t``, distilled from a flow trained on it. Its output is a
distribution over the vocabulary for every slot. Drafting with one step is a single
jump from ``s = 0`` to ``t = 1``:

    x0 ~ N(0, I)  ->  logits = map(x0; 0 -> 1)  ->  y_k ~ softmax(logits_k) per slot.

Given the noise the slots are independent draws, so ``softmax(logits)`` is an exact
proposal distribution for lossless speculative (rejection) sampling, while across
noise draws the block is a joint sample: the noise decides which mode the whole block
follows. More steps jump over a grid ``0 = t_0 < t_1 < ... < t_n = 1``, moving the
state to ``x <- ((1 - t) x + (t - s) softmax(logits)) / (1 - s)`` between jumps.

The network is a small DiT. It reads the block's state through the target's token
embedding ``E`` (the noise enters as ``x0 @ E``), conditions on the jump's start and
end times through adaptive LayerNorms, self-attends across the block's slots, and
cross-attends to the target's hidden states at earlier positions (from a few of its
layers) and to the anchor's embedding. Its output goes through the target's lm_head.
The target's embedding and lm_head are not part of the drafter's weights.

``FlashMapDraftModel.spec_generate`` runs lossless speculative decoding with a
transformers target model.
"""

from dataclasses import dataclass, field
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache, PreTrainedModel

from .configuration_flashmap import FlashMapConfig


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary embedding of ``x (B, T, heads, head_dim)`` at ``positions (B, T)``,
    rotating interleaved channel pairs ``(x[2i], x[2i + 1])``, in float32."""
    head_dim = x.shape[-1]
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim)
    angles = positions.float()[..., None, None] * inv_freq
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2).type_as(x)


class FlashMapRMSNorm(nn.Module):
    """RMSNorm whose scale ``weight / rms(x)`` is rounded to the input's dtype before
    it multiplies the input (the order the released weights were trained with)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * (self.weight.float() * rms).to(x.dtype)


class FlashMapTimeEmbedding(nn.Module):
    """Sinusoidal features of a time in [0, 1], then an MLP."""

    def __init__(self, config: FlashMapConfig):
        super().__init__()
        self.dim, self.scale = config.time_embed_dim, config.time_scale
        self.linear_1 = nn.Linear(config.time_embed_dim, config.cond_dim)
        self.linear_2 = nn.Linear(config.cond_dim, config.cond_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = (t.float() * self.scale)[..., None] * freqs
        emb = torch.cat([args.cos(), args.sin()], dim=-1).to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(emb)))


class FlashMapAdaLayerNorm(nn.Module):
    """LayerNorm modulated by the time conditioning (AdaLN-zero): returns
    ``norm(x) * (1 + scale) + shift`` and the gate of the residual branch."""

    def __init__(self, config: FlashMapConfig):
        super().__init__()
        self.linear = nn.Linear(config.cond_dim, 3 * config.hidden_size)
        self.eps = config.layer_norm_eps

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift, gate = self.linear(F.silu(cond)).chunk(3, dim=-1)
        return F.layer_norm(x, x.shape[-1:], eps=self.eps) * (1 + scale) + shift, gate


class FlashMapAttention(nn.Module):
    """Multi-head attention with RMS-normed queries and keys and rotary positions.

    Keys and values come from ``project_kv`` so that cross-attention can project its
    context once and cache it."""

    def __init__(self, config: FlashMapConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = FlashMapRMSNorm(self.head_dim, config.qk_norm_eps)
        self.k_norm = FlashMapRMSNorm(self.head_dim, config.qk_norm_eps)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:2], self.num_heads, self.head_dim)

    def project_kv(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values ``(B, heads, T, head_dim)`` of ``x (B, T, dim)``."""
        keys = apply_rope(self.k_norm(self._heads(self.k_proj(x))), positions, self.rope_theta)
        return keys.transpose(1, 2), self._heads(self.v_proj(x)).transpose(1, 2)

    def forward(self, x, positions, keys, values) -> torch.Tensor:
        queries = apply_rope(self.q_norm(self._heads(self.q_proj(x))), positions, self.rope_theta)
        out = F.scaled_dot_product_attention(queries.transpose(1, 2), keys, values)
        return self.o_proj(out.transpose(1, 2).flatten(2))


class FlashMapMLP(nn.Module):
    def __init__(self, config: FlashMapConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class FlashMapLayer(nn.Module):
    """A DiT block: self-attention across the block's slots, cross-attention to the
    context, and an MLP, each behind a time-modulated LayerNorm and a gated residual."""

    def __init__(self, config: FlashMapConfig):
        super().__init__()
        self.self_attn_adaln = FlashMapAdaLayerNorm(config)
        self.self_attn = FlashMapAttention(config)
        self.cross_attn_adaln = FlashMapAdaLayerNorm(config)
        self.cross_attn = FlashMapAttention(config)
        self.mlp_adaln = FlashMapAdaLayerNorm(config)
        self.mlp = FlashMapMLP(config)

    def forward(self, x, cond, positions, context: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h, gate = self.self_attn_adaln(x, cond)
        x = x + gate * self.self_attn(h, positions, *self.self_attn.project_kv(h, positions))
        h, gate = self.cross_attn_adaln(x, cond)
        x = x + gate * self.cross_attn(h, positions, *context)
        h, gate = self.mlp_adaln(x, cond)
        return x + gate * self.mlp(h)


class FlashMapCache:
    """Cross-attention keys and values of the drafter's context, for every DiT layer.

    Positions ``[0, length)`` hold the target's hidden states at the tokens verified
    so far. Position ``length`` belongs to the anchor, the last token, whose hidden
    state the target has not computed yet: drafting writes the anchor's embedding
    there, and the next ``extend_context`` overwrites it with its hidden state.
    """

    def __init__(self, config: FlashMapConfig, batch_size: int, max_length: int, device=None, dtype=None):
        head_dim = config.hidden_size // config.num_attention_heads
        shape = (config.num_hidden_layers, 2, batch_size, config.num_attention_heads, max_length, head_dim)
        self.kv = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def write(self, layer: int, start: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        end = start + keys.shape[2]
        if end > self.kv.shape[4]:
            raise ValueError(f"FlashMapCache holds {self.kv.shape[4]} positions, writing up to {end}")
        self.kv[layer, 0, :, :, start:end] = keys
        self.kv[layer, 1, :, :, start:end] = values

    def context(self, end: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(kv[0, :, :, :end], kv[1, :, :, :end]) for kv in self.kv]


@dataclass
class FlashMapGenerateOutput:
    """``sequences``: prompt and generated tokens, ``(1, length)``.
    ``acceptance_lengths``: tokens committed by each target forward pass (accepted
    draft tokens plus the target's own token)."""

    sequences: torch.LongTensor
    num_new_tokens: int
    acceptance_lengths: list = field(default_factory=list)

    @property
    def acceptance_length(self) -> float:
        """Mean tokens per target forward pass (the paper's gamma)."""
        return sum(self.acceptance_lengths) / max(len(self.acceptance_lengths), 1)


class FlashMapDraftModel(PreTrainedModel):
    config_class = FlashMapConfig
    base_model_prefix = "flashmap"
    _no_split_modules = ["FlashMapLayer"]

    def __init__(self, config: FlashMapConfig):
        super().__init__(config)
        dim, target_dim = config.hidden_size, config.target_hidden_size
        # Context: the target's hidden states at target_layer_ids, and the anchor.
        self.fc = nn.Linear(len(config.target_layer_ids) * target_dim, dim, bias=False)
        self.hidden_norm = FlashMapRMSNorm(dim, config.rms_norm_eps)
        self.anchor_proj = nn.Linear(target_dim, dim)
        self.anchor_norm = FlashMapRMSNorm(dim, config.rms_norm_eps)
        # State featurization (see embed_state).
        self.state_norm = FlashMapRMSNorm(target_dim, config.rms_norm_eps)
        self.noise_norm = FlashMapRMSNorm(target_dim, config.rms_norm_eps)
        self.noise_gate = nn.Parameter(torch.ones(1))
        # The DiT.
        self.x_embed = nn.Linear(target_dim, dim)
        self.time_embed = FlashMapTimeEmbedding(config)  # the jump's start time s
        self.jump_embed = FlashMapTimeEmbedding(config)  # the jump's end time t
        self.layers = nn.ModuleList(FlashMapLayer(config) for _ in range(config.num_hidden_layers))
        self.final_adaln = FlashMapAdaLayerNorm(config)
        self.head = nn.Linear(dim, target_dim)
        self._noise_constants = None
        self.post_init()

    def forward(self, inputs_embeds, s: float, t: float, position_ids, context) -> torch.Tensor:
        """The jump from time ``s`` to ``t``: the featurized state ``inputs_embeds (B, K,
        target_hidden)`` of blocks at ``position_ids (B, K)`` -> hidden states ``(B, K,
        target_hidden)`` whose target lm_head logits are the distribution it lands on.
        ``context`` holds each layer's cross-attention keys and values."""
        times = torch.tensor([s, t], device=inputs_embeds.device)
        cond = (self.time_embed(times[:1]) + self.jump_embed(times[1:]))[None]
        h = self.x_embed(inputs_embeds)
        for layer, layer_context in zip(self.layers, context):
            h = layer(h, cond, position_ids, layer_context)
        h, _ = self.final_adaln(h, cond)
        return self.head(h)

    # ---- context ------------------------------------------------------------------

    def select_target_hidden(self, hidden_states) -> torch.Tensor:
        """The drafter's features from a target forward's ``hidden_states`` tuple: the
        outputs of layers ``target_layer_ids``, concatenated."""
        return torch.cat([hidden_states[i + 1] for i in self.config.target_layer_ids], dim=-1)

    @torch.no_grad()
    def extend_context(self, cache: FlashMapCache, target_hidden: torch.Tensor) -> None:
        """Append the target's features ``(B, n, ...)`` at the next ``n`` verified positions."""
        start, n = cache.length, target_hidden.shape[1]
        context = self.hidden_norm(self.fc(target_hidden.to(self.dtype)))
        positions = torch.arange(start, start + n, device=context.device).expand(context.shape[0], -1)
        for i, layer in enumerate(self.layers):
            cache.write(i, start, *layer.cross_attn.project_kv(context, positions))
        cache.length += n

    # ---- drafting -----------------------------------------------------------------

    def embed_state(self, state, s: float, t: float, embedding: torch.Tensor) -> torch.Tensor:
        """DiT input for a vocabulary-sized state ``(B, K, V)`` at time ``s`` jumping to ``t``.

        The state is read through the target's token embedding twice: as the posterior
        over clean tokens it implies, ``softmax(s x / (1 - s)^2)``, and linearly, weighted
        by the jump. At ``s = 0`` the posterior is uniform whatever the noise, so the
        linear term is how a jump from noise sees which noise it starts from."""
        s = torch.tensor(s, device=state.device)
        posterior = (state.float() * (s / (1.0 - s).clamp_min(1e-3) ** 2)).softmax(dim=-1)
        posterior_embeds = posterior.to(embedding.dtype) @ embedding
        return self.mix_state(posterior_embeds, state.to(embedding.dtype) @ embedding, s, t)

    def mix_state(self, posterior_embeds, state_embeds, s, t) -> torch.Tensor:
        """``embed_state`` from the state's two projections onto the embedding."""
        s = torch.as_tensor(s, device=state_embeds.device)
        gate = (self.noise_gate.float() * (t - s)).to(self.dtype)
        posterior_term = self.state_norm(posterior_embeds.to(self.dtype))
        return posterior_term + gate * self.noise_norm(state_embeds.to(self.dtype))

    @torch.no_grad()
    def noise_constants(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What a one-step draft needs of the target's embedding ``E (V, H)``: the
        embedded uniform posterior, and ``L`` with ``L L^T = E^T E``.

        A one-step draft reads its noise only as ``x0 @ E`` with ``x0 ~ N(0, I_V)``,
        a Gaussian with covariance ``E^T E``; ``z @ L^T`` with ``z ~ N(0, I_H)`` has the
        same distribution and needs no vocabulary-sized noise."""
        key = (embedding.data_ptr(), embedding.device)
        if self._noise_constants is None or self._noise_constants[0] != key:
            gram = sum(rows.double().T @ rows.double() for rows in embedding.split(16384))
            factor = torch.linalg.cholesky(gram).float()
            vocab_size = embedding.shape[0]
            uniform = torch.full((1, vocab_size), 1.0 / vocab_size, device=embedding.device)
            self._noise_constants = (key, uniform.to(embedding.dtype) @ embedding, factor)
        return self._noise_constants[1:]

    @torch.no_grad()
    def draft(
        self,
        anchor_ids: torch.LongTensor,
        cache: FlashMapCache,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        num_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draft logits ``(B, block_size, V)`` for the block after ``anchor_ids (B,)``.

        ``cache`` holds the target's features at every position before the anchor.
        ``embed_tokens`` and ``lm_head`` are the target's. ``noise (B, block_size, V)``
        is the starting state ``x0``; by default it is drawn, and a one-step draft draws
        only its embedding (see ``noise_constants``). Given the noise, the block's
        tokens are independent draws from ``softmax(logits)``, one per slot."""
        num_steps = num_steps or self.config.num_steps
        grid = self.config.jump_grids[str(num_steps)]
        batch_size, block_size = anchor_ids.shape[0], self.config.block_size
        embedding, device = embed_tokens.weight, anchor_ids.device

        # The anchor's embedding enters the context at the anchor's position.
        anchor = cache.length
        anchor_context = self.anchor_norm(self.anchor_proj(embed_tokens(anchor_ids[:, None]).to(self.dtype)))
        anchor_positions = torch.full((batch_size, 1), anchor, device=device)
        for i, layer in enumerate(self.layers):
            cache.write(i, anchor, *layer.cross_attn.project_kv(anchor_context, anchor_positions))
        context = cache.context(anchor + 1)
        positions = anchor + 1 + torch.arange(block_size, device=device).expand(batch_size, -1)

        state = noise
        if state is None and num_steps == 1:
            uniform_embeds, factor = self.noise_constants(embedding)
            z = torch.randn(batch_size, block_size, factor.shape[0], device=device, generator=generator)
            inputs = self.mix_state(uniform_embeds, z @ factor.T, 0.0, grid[1])
        elif state is None:
            shape = (batch_size, block_size, embedding.shape[0])
            state = torch.randn(shape, device=device, generator=generator)

        for s, t in zip(grid[:-1], grid[1:]):
            if state is not None:
                inputs = self.embed_state(state, s, t, embedding)
            hidden = self(inputs, s, t, positions, context)
            logits = lm_head(hidden.to(lm_head.weight.dtype))
            if t < 1.0:
                # move to where the jump lands (eq. 19 of arXiv:2602.16813)
                s_, t_ = torch.tensor([s, t], device=device).unbind()
                state = ((1.0 - t_) * state + (t_ - s_) * logits.float().softmax(dim=-1)) / (1.0 - s_)
        return logits

    # ---- speculative decoding -----------------------------------------------------

    @torch.no_grad()
    def spec_generate(
        self,
        target: PreTrainedModel,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_steps: Optional[int] = None,
        stop_token_ids: Optional[list] = None,
        generator: Optional[torch.Generator] = None,
    ) -> FlashMapGenerateOutput:
        """Lossless speculative decoding of ``target`` (a transformers causal LM) from
        the prompt ``input_ids (1, P)``.

        Every round drafts a block after the last token, has the target score it in one
        forward pass, and keeps the longest prefix the target accepts plus one token of
        the target's own. At ``temperature`` 0 the output is the target's greedy
        decoding; above it, tokens are distributed exactly as the target samples them
        (rejection sampling against the draft's per-slot probabilities). Generation stops
        after ``max_new_tokens`` or at a token in ``stop_token_ids`` (default: the
        target's ``eos_token_id``)."""
        if input_ids.shape[0] != 1:
            raise ValueError("spec_generate decodes one sequence at a time")
        if stop_token_ids is None:
            eos = target.generation_config.eos_token_id
            stop_token_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        stop = torch.tensor(stop_token_ids, device=input_ids.device, dtype=torch.long)
        embed_tokens, lm_head = target.get_input_embeddings(), target.get_output_embeddings()
        block_size, prompt_length = self.config.block_size, input_ids.shape[1]

        def sample(logits):  # (1, n, V) -> tokens (1, n) and their distributions
            if temperature <= 0:
                return logits.argmax(dim=-1), None
            probs = (logits.float() / temperature).softmax(dim=-1)
            return torch.multinomial(probs[0], 1, generator=generator).T, probs

        target_cache = DynamicCache()
        out = target(
            input_ids, past_key_values=target_cache, use_cache=True, output_hidden_states=True, logits_to_keep=1
        )
        tokens = sample(out.logits[:, -1:])[0]
        max_length = prompt_length + max_new_tokens + block_size + 1  # the last round may overshoot
        cache = FlashMapCache(self.config, 1, max_length, device=input_ids.device, dtype=self.dtype)
        self.extend_context(cache, self.select_target_hidden(out.hidden_states))
        acceptance_lengths = []

        while tokens.shape[1] < max_new_tokens and not torch.isin(tokens[0, -1], stop):
            anchor = tokens[:, -1]
            draft_logits = self.draft(anchor, cache, embed_tokens, lm_head, num_steps, generator=generator)
            draft, draft_probs = sample(draft_logits)
            position = prompt_length + tokens.shape[1] - 1  # the anchor's
            out = target(
                torch.cat([anchor[:, None], draft], dim=1),
                position_ids=torch.arange(position, position + block_size + 1, device=draft.device)[None],
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            if temperature <= 0:
                target_tokens = out.logits.argmax(dim=-1)
                accepted = int((draft == target_tokens[:, :-1]).cumprod(dim=1).sum())
                bonus = target_tokens[:, accepted]
            else:
                target_probs = (out.logits.float() / temperature).softmax(dim=-1)[0]
                p = target_probs[:-1].gather(-1, draft[0, :, None])[:, 0]
                q = draft_probs[0].gather(-1, draft[0, :, None])[:, 0]
                u = torch.rand(q.shape, device=q.device, generator=generator)
                accepted = int((u * q < p).cumprod(dim=0).sum())
                residual = target_probs[accepted]
                if accepted < block_size:  # resample the rejected slot from max(p - q, 0)
                    residual = (residual - draft_probs[0, accepted]).clamp_min(0)
                    if residual.sum() <= 0:
                        residual = target_probs[accepted]
                bonus = torch.multinomial(residual, 1, generator=generator)
            new_tokens = torch.cat([draft[:, :accepted], bonus.view(1, 1)], dim=1)
            hits = torch.isin(new_tokens[0], stop).nonzero()
            if len(hits):
                new_tokens = new_tokens[:, : int(hits[0]) + 1]
            tokens = torch.cat([tokens, new_tokens], dim=1)
            acceptance_lengths.append(new_tokens.shape[1])
            # keep the anchor and the accepted drafts; the bonus token is the next anchor
            target_cache.crop(position + accepted + 1)
            self.extend_context(cache, self.select_target_hidden(out.hidden_states)[:, : accepted + 1])

        tokens = tokens[:, :max_new_tokens]
        return FlashMapGenerateOutput(
            sequences=torch.cat([input_ids, tokens], dim=1),
            num_new_tokens=tokens.shape[1],
            acceptance_lengths=acceptance_lengths,
        )


__all__ = ["FlashMapCache", "FlashMapDraftModel", "FlashMapGenerateOutput"]
