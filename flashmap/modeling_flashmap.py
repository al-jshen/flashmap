"""FlashMap: speculative decoding with a flow-map drafter.

FlashMap drafts the ``K`` tokens after the last verified token (the anchor) in one
network evaluation, like DFlash, but from a generative model that can couple the
slots of a block. A block is a point ``x`` in ``R^{K x V}`` (``V`` the target's
vocabulary) moving along the path ``x_s = (1 - s) x0 + s onehot(y)`` from Gaussian
noise ``x0`` at ``s = 0`` to the one-hot tokens ``y`` at ``s = 1``. The drafter is a
*flow map* (Flow Map Language Models, arXiv:2602.16813): a network that jumps along
this path from time ``s`` to time ``t``, distilled from a flow trained on it. Its
output is a distribution over the vocabulary for every slot. Drafting with one step
is a single jump from ``s = 0`` to ``t = 1``:

    x0 ~ N(0, I)  ->  logits = map(x0; 0 -> 1)  ->  y_k ~ softmax(logits_k) per slot.

Given the noise the slots are independent draws, so ``softmax(logits)`` is an exact
proposal distribution for lossless speculative (rejection) sampling. Across noise
draws the slots can be correlated, since the same noise conditions all of them; one
jump captures this joint structure only partly, and more jumps capture more. More
steps jump over a grid ``0 = t_0 < t_1 < ... < t_n = 1``, moving the state to
``x <- ((1 - t) x + (t - s) softmax(logits)) / (1 - s)`` between jumps.

The network is a small DiT. It reads the block's state through the target's token
embedding ``E`` (the noise enters as ``x0 @ E``), conditions on the jump's start and
end times through adaptive LayerNorms, self-attends across the block's slots, and
cross-attends to the target's hidden states at earlier positions (from a few of its
layers) and to the anchor's embedding. Its output goes through the target's lm_head.
The target's embedding and lm_head are not part of the drafter's weights.

``FlashMapDraftModel.spec_generate`` runs lossless speculative decoding with a
transformers target model. This module is inference-only.
"""

import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache, PreTrainedModel

from .configuration_flashmap import FlashMapConfig


def rope_tables(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos`` and ``sin`` of the rotary angles at ``positions (B, T)``, each
    ``(B, T, 1, head_dim // 2)`` in float32; computed once and shared by all layers."""
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    angles = positions.float()[..., None, None] * inv_freq
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate the interleaved channel pairs ``(x[2i], x[2i + 1])`` of ``x (B, T, heads,
    head_dim)`` by the angles of ``rope_tables``, in float32."""
    cos, sin = rope
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
    """LayerNorm modulated by the time conditioning (AdaLN-zero)."""

    def __init__(self, config: FlashMapConfig):
        super().__init__()
        self.linear = nn.Linear(config.cond_dim, 3 * config.hidden_size)
        self.eps = config.layer_norm_eps

    def modulation(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``(scale, shift, gate)`` for the time conditioning ``cond``."""
        return self.linear(F.silu(cond)).chunk(3, dim=-1)

    def forward(self, x: torch.Tensor, modulation) -> tuple[torch.Tensor, torch.Tensor]:
        """``norm(x) * (1 + scale) + shift``, and the gate of the residual branch."""
        scale, shift, gate = modulation
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
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = FlashMapRMSNorm(self.head_dim, config.qk_norm_eps)
        self.k_norm = FlashMapRMSNorm(self.head_dim, config.qk_norm_eps)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:2], self.num_heads, self.head_dim)

    def project_kv(self, x: torch.Tensor, rope) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values ``(B, heads, T, head_dim)`` of ``x (B, T, dim)``."""
        keys = apply_rope(self.k_norm(self._heads(self.k_proj(x))), rope)
        return keys.transpose(1, 2), self._heads(self.v_proj(x)).transpose(1, 2)

    def forward(self, x, rope, keys, values) -> torch.Tensor:
        queries = apply_rope(self.q_norm(self._heads(self.q_proj(x))), rope)
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

    def forward(self, x, modulations, rope, context: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self_mod, cross_mod, mlp_mod = modulations
        h, gate = self.self_attn_adaln(x, self_mod)
        x = x + gate * self.self_attn(h, rope, *self.self_attn.project_kv(h, rope))
        h, gate = self.cross_attn_adaln(x, cross_mod)
        x = x + gate * self.cross_attn(h, rope, *context)
        h, gate = self.mlp_adaln(x, mlp_mod)
        return x + gate * self.mlp(h)


class FlashMapCache:
    """Cross-attention keys and values of the drafter's context, for every DiT layer.

    Positions ``[0, length)`` hold the target's hidden states at the tokens verified
    so far. Position ``length`` belongs to the anchor, the last token, whose hidden
    state the target has not computed yet: drafting writes the anchor's embedding
    there, and the next ``extend_context`` overwrites it with its hidden state.
    Create it on the drafter's device and dtype.
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
    draft tokens plus the target's own token). The last pass counts in full even
    where it runs past ``max_new_tokens``, as in the evaluation of the paper."""

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
    _supports_sdpa = True

    def __init__(self, config: FlashMapConfig):
        super().__init__(config)
        dim, target_dim = config.hidden_size, config.target_hidden_size
        self.head_dim = dim // config.num_attention_heads
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
        self._clear_caches()
        self.post_init()

    # Values computed from the weights once and reused by every draft: the AdaLN
    # modulations of each jump, and constants of the target's embedding. Moving or
    # reloading the weights clears them.

    def _clear_caches(self) -> None:
        self._time_cache, self._target_cache = {}, {}

    def _apply(self, fn, *args, **kwargs):
        self._clear_caches()
        return super()._apply(fn, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self._clear_caches()
        return super().load_state_dict(*args, **kwargs)

    @torch.no_grad()
    def time_conditioning(self, s: float, t: float):
        """The AdaLN modulations of the jump from ``s`` to ``t``, for every norm of the
        DiT: per layer (self-attention, cross-attention, MLP), then the final norm. They
        depend on the two times alone, so each jump computes them once."""
        if (s, t) not in self._time_cache:
            times = torch.tensor([s, t], device=self.device)
            cond = (self.time_embed(times[:1]) + self.jump_embed(times[1:]))[None]
            layers = [
                [norm.modulation(cond) for norm in (block.self_attn_adaln, block.cross_attn_adaln, block.mlp_adaln)]
                for block in self.layers
            ]
            self._time_cache[(s, t)] = (layers, self.final_adaln.modulation(cond))
        return self._time_cache[(s, t)]

    def forward(self, inputs_embeds, s: float, t: float, position_ids, context) -> torch.Tensor:
        """The jump from time ``s`` to ``t``: the featurized state ``inputs_embeds (B, K,
        target_hidden)`` of blocks at ``position_ids (B, K)`` -> hidden states ``(B, K,
        target_hidden)`` whose target lm_head logits are the distribution it lands on.
        ``context`` holds each layer's cross-attention keys and values."""
        layer_modulations, final_modulation = self.time_conditioning(s, t)
        rope = rope_tables(position_ids, self.head_dim, self.config.rope_theta)
        h = self.x_embed(inputs_embeds)
        for layer, modulations, layer_context in zip(self.layers, layer_modulations, context):
            h = layer(h, modulations, rope, layer_context)
        h, _ = self.final_adaln(h, final_modulation)
        return self.head(h)

    # ---- context ------------------------------------------------------------------

    def select_target_hidden(self, hidden_states) -> torch.Tensor:
        """The drafter's features from a target forward's ``hidden_states`` tuple: the
        outputs of layers ``target_layer_ids``, concatenated."""
        return torch.cat([hidden_states[i + 1] for i in self.config.target_layer_ids], dim=-1)

    @contextmanager
    def target_features(self, target: PreTrainedModel):
        """Forward hooks that keep the outputs of the target's layers ``target_layer_ids``.
        Yields a function returning them, concatenated, for the target's last forward.
        Unlike ``output_hidden_states``, this does not keep every layer's output."""
        layers, outputs = target.get_decoder().layers, {}

        def keep(i):
            def hook(module, args, output):
                outputs[i] = output[0] if isinstance(output, tuple) else output

            return hook

        hooks = [layers[i].register_forward_hook(keep(i)) for i in self.config.target_layer_ids]
        try:
            yield lambda: torch.cat([outputs[i] for i in self.config.target_layer_ids], dim=-1)
        finally:
            for hook in hooks:
                hook.remove()

    @torch.no_grad()
    def extend_context(self, cache: FlashMapCache, target_hidden: torch.Tensor) -> None:
        """Append the target's features ``(B, n, ...)`` at the next ``n`` verified positions."""
        start, n = cache.length, target_hidden.shape[1]
        context = self.hidden_norm(self.fc(target_hidden.to(self.dtype)))
        positions = torch.arange(start, start + n, device=context.device).expand(context.shape[0], -1)
        rope = rope_tables(positions, self.head_dim, self.config.rope_theta)
        for i, layer in enumerate(self.layers):
            cache.write(i, start, *layer.cross_attn.project_kv(context, rope))
        cache.length += n

    # ---- drafting -----------------------------------------------------------------

    def embed_state(self, state, s: float, t: float, embedding: torch.Tensor) -> torch.Tensor:
        """DiT input for a vocabulary-sized state ``(B, K, V)`` at time ``s`` jumping to ``t``.

        The state is read through the target's token embedding twice: as the posterior
        over clean tokens it implies, ``softmax(s x / (1 - s)^2)``, and linearly, weighted
        by the jump. At ``s = 0`` the posterior is uniform whatever the noise, so the
        linear term is how a jump from noise sees which noise it starts from."""
        if s == 0:
            posterior_embeds = self.target_constants(embedding)[0]
        else:
            posterior = (state.float() * (s / max(1.0 - s, 1e-3) ** 2)).softmax(dim=-1)
            posterior_embeds = posterior.to(embedding.dtype) @ embedding
        return self.mix_state(posterior_embeds, state.to(embedding.dtype) @ embedding, s, t)

    def mix_state(self, posterior_embeds, state_embeds, s: float, t: float) -> torch.Tensor:
        """``embed_state`` from the state's two projections onto the embedding."""
        # the noise term's weight: noise_gate times the jump ("jump") or alone ("constant")
        jump = 1.0 if self.config.noise_gate == "constant" else t - s
        gate = (self.noise_gate.float() * jump).to(self.dtype)
        posterior_term = self.state_norm(posterior_embeds.to(self.dtype))
        return posterior_term + gate * self.noise_norm(state_embeds.to(self.dtype))

    @torch.no_grad()
    def target_constants(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What drafting needs of the target's embedding ``E (V, H)``: the embedded
        uniform posterior (the posterior of any state at ``s = 0``), and ``L`` with
        ``L L^T = E^T E``.

        A one-step draft reads its noise only as ``x0 @ E`` with ``x0 ~ N(0, I_V)``,
        a Gaussian with covariance ``E^T E``; ``z @ L^T`` with ``z ~ N(0, I_H)`` has the
        same distribution and needs no vocabulary-sized noise."""
        key = (embedding.data_ptr(), tuple(embedding.shape), embedding.device, embedding.dtype)
        if key not in self._target_cache:
            vocab_size = embedding.shape[0]
            uniform = torch.full((1, vocab_size), 1.0 / vocab_size, device=embedding.device)
            # E^T E in float32 where E is; its (H, H) factor in float64 on the CPU, as
            # many accelerators lack fast (or any) float64.
            gram = sum(rows.float().T @ rows.float() for rows in embedding.split(16384))
            factor = torch.linalg.cholesky(gram.cpu().double())
            uniform_embeds = uniform.to(embedding.dtype) @ embedding
            self._target_cache = {key: (uniform_embeds, factor.to(self.device, self.dtype))}
        return self._target_cache[key]

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
        only its embedding (see ``target_constants``). Given the noise, the block's
        tokens are independent draws from ``softmax(logits)``, one per slot."""
        num_steps = num_steps or self.config.num_steps
        if str(num_steps) not in self.config.jump_grids:
            supported = ", ".join(sorted(self.config.jump_grids, key=int))
            raise ValueError(f"num_steps must be one of {supported} (the config's jump_grids), got {num_steps}")
        grid = self.config.jump_grids[str(num_steps)]
        batch_size, block_size = anchor_ids.shape[0], self.config.block_size
        embedding, device = embed_tokens.weight, anchor_ids.device

        # The anchor's embedding enters the context at the anchor's position.
        anchor = cache.length
        anchor_context = self.anchor_norm(self.anchor_proj(embed_tokens(anchor_ids[:, None]).to(self.dtype)))
        anchor_position = torch.full((batch_size, 1), anchor, device=device)
        anchor_rope = rope_tables(anchor_position, self.head_dim, self.config.rope_theta)
        for i, layer in enumerate(self.layers):
            cache.write(i, anchor, *layer.cross_attn.project_kv(anchor_context, anchor_rope))
        context = cache.context(anchor + 1)
        positions = anchor + 1 + torch.arange(block_size, device=device).expand(batch_size, -1)

        state = noise
        if state is None and num_steps == 1:
            uniform_embeds, factor = self.target_constants(embedding)
            shape = (batch_size, block_size, factor.shape[0])
            z = torch.randn(shape, device=device, dtype=factor.dtype, generator=generator)
            inputs = self.mix_state(uniform_embeds, z @ factor.T, 0.0, grid[1])
        elif state is None:
            shape = (batch_size, block_size, embedding.shape[0])
            state = torch.randn(shape, device=device, generator=generator)

        for s, t in zip(grid[:-1], grid[1:]):
            if state is not None:
                inputs = self.embed_state(state, s, t, embedding)
            logits = lm_head(self(inputs, s, t, positions, context).to(lm_head.weight.dtype))
            if t < 1.0:
                # move to where the jump lands (eq. 19 of arXiv:2602.16813)
                state = ((1.0 - t) * state + (t - s) * logits.float().softmax(dim=-1)) / (1.0 - s)
        return logits

    # ---- speculative decoding -----------------------------------------------------

    def check_target(self, target: PreTrainedModel) -> None:
        """Raise if ``target`` cannot serve as this drafter's target; warn if it is not
        the model the drafter was trained for (the output stays lossless, but drafts
        from another model's hidden states are poor)."""
        config = target.config.get_text_config()
        embedding, lm_head = target.get_input_embeddings().weight, target.get_output_embeddings().weight
        expected = (self.config.vocab_size, self.config.target_hidden_size)
        if tuple(embedding.shape) != expected or tuple(lm_head.shape) != expected:
            raise ValueError(
                f"this drafter needs a target with vocabulary {expected[0]} and hidden size {expected[1]} "
                f"({self.config.target_model}), got an embedding of shape {tuple(embedding.shape)}"
            )
        if max(self.config.target_layer_ids) >= config.num_hidden_layers - 1:
            layers = config.num_hidden_layers
            raise ValueError(f"target_layer_ids {self.config.target_layer_ids} need more than {layers} target layers")
        name = (getattr(config, "_name_or_path", None) or "").rstrip("/")
        if name and name.split("/")[-1] != self.config.target_model.split("/")[-1]:
            warnings.warn(f"this drafter was trained for {self.config.target_model}, not {name}")
        devices = {self.device, embedding.device, lm_head.device}
        if len(devices) > 1:
            raise ValueError(f"the drafter and the target's embedding and lm_head must share a device, got {devices}")

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
        target's ``eos_token_id``). The drafter, the target and ``generator`` (which
        makes sampling reproducible) must be on one device."""
        if input_ids.shape[0] != 1:
            raise ValueError("spec_generate decodes one sequence at a time")
        self.check_target(target)
        if generator is not None:  # a "cuda" generator (no index) is on the current device
            device = generator.device
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if (device.type, device.index or 0) != (self.device.type, self.device.index or 0):
                raise ValueError(f"generator must be on {self.device}, got one on {generator.device}")
        if stop_token_ids is None:
            eos = target.generation_config.eos_token_id
            stop_token_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        input_ids = input_ids.to(self.device)
        stop = torch.tensor(stop_token_ids, device=self.device, dtype=torch.long)
        embed_tokens, lm_head = target.get_input_embeddings(), target.get_output_embeddings()
        block_size, prompt_length = self.config.block_size, input_ids.shape[1]

        def sample(logits):  # (1, n, V) -> tokens (1, n) and their distributions
            if temperature <= 0:
                return logits.argmax(dim=-1), None
            probs = (logits.float() / temperature).softmax(dim=-1)
            return torch.multinomial(probs[0], 1, generator=generator).T, probs

        with self.target_features(target) as target_features:
            target_cache = DynamicCache()
            out = target(input_ids, past_key_values=target_cache, use_cache=True, logits_to_keep=1)
            tokens = sample(out.logits[:, -1:])[0]
            max_length = prompt_length + max_new_tokens + block_size + 1  # the last round may overshoot
            cache = FlashMapCache(self.config, 1, max_length, device=self.device, dtype=self.dtype)
            self.extend_context(cache, target_features())
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
                excess = target_cache.get_seq_length() - (position + accepted + 1)
                if excess > 0:  # crop(-n) drops n tokens in every transformers version
                    target_cache.crop(-excess)
                self.extend_context(cache, target_features()[:, : accepted + 1])

        tokens = tokens[:, :max_new_tokens]
        return FlashMapGenerateOutput(
            sequences=torch.cat([input_ids, tokens], dim=1),
            num_new_tokens=tokens.shape[1],
            acceptance_lengths=acceptance_lengths,
        )


__all__ = ["FlashMapCache", "FlashMapDraftModel", "FlashMapGenerateOutput"]
