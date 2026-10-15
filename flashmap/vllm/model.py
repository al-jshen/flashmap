"""FlashMap draft model for vLLM.

The model is driven by ``FlashMapSpeculator`` (speculator.py), which reuses
vLLM's DFlash machinery: the target's aux hidden states are fused by ``fc``
(``combine_hidden_states``) and every committed token's cross-attention keys and
values go to the draft's paged KV cache; each draft step then runs ``forward`` on
``block_size`` query rows per request, laid out as in DSpark: row 0 carries the
anchor token at the anchor position ``p`` and row ``k`` sits at ``p + k``.

A FlashMap draft step, for each request:

* writes the keys/values of the context committed since the last step and of the
  anchor (its embedding through ``anchor_proj`` and ``anchor_norm``, at ``p``,
  where the target's own entry for that token lands once it is verified), all
  projected by one GEMM over every layer's weights;
* evaluates the DiT on ``block_size`` slots at positions ``p + 1 ..`` from noise,
  their cross-attention reading positions ``0 .. p`` straight from the paged KV
  cache with FlashAttention;
* returns the final hidden states, which the speculator maps to logits through
  the DiT head folded into the target's LM head.

One jump (the default) needs no vocabulary-sized tensor: the state at ``t = 0``
embeds to a constant (its posterior is uniform) plus the noise's embedding,
drawn in the embedding's eigenbasis (fold.py). Everything that depends only on
the jump times (every AdaLN modulation) is computed once at load. More jumps
run the reference algorithm exactly, with a vocabulary-sized state.

Requirements: the V2 model runner, tensor parallel size 1, the FlashAttention
backend and a bf16 KV cache (checked at load).

The draft noise is keyed by each request's seed and the slot's position, so a
seeded request drafts the same blocks on every run. ``FLASHMAP_NUM_STEPS``
overrides the checkpoint's number of jumps.
"""

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from flashmap.vllm.fold import (
    fold_head,
    fold_noise,
    quantize_fp8_per_channel,
    timestep_embedding,
)
from flashmap.vllm.kernels import (
    block_self_attention,
    rms_norm,
    rms_norm_rope,
    rope_table,
    seeded_normal,
    write_context_kv,
)

try:
    from vllm.vllm_flash_attn import get_scheduler_metadata
except ImportError:  # no FlashAttention 3 build
    get_scheduler_metadata = None

logger = init_logger(__name__)

__all__ = ["FlashMapDraftModel", "FlashMapModel"]


@torch.library.custom_op("flashmap::paged_cross_attention", mutates_args=())
def paged_cross_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    scheduler_metadata: torch.Tensor | None,
    descales: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    fa_version: int,
) -> torch.Tensor:
    """Non-causal FlashAttention of ``query (tokens, heads, head_dim)`` onto the
    first ``seqused_k`` cached positions of each request. An fp8 cache takes an
    fp8 ``query`` and ``descales (3, requests, heads)`` for the query, keys and
    values."""
    head_dim = query.shape[-1]
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    q_descale = k_descale = v_descale = None
    if descales is not None:
        key_cache = key_cache.view(query.dtype)
        value_cache = value_cache.view(query.dtype)
        q_descale, k_descale, v_descale = descales.unbind(0)
    return flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=False,
        block_table=block_table,
        scheduler_metadata=scheduler_metadata,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        fa_version=fa_version,
    )


@paged_cross_attention.register_fake
def _(query, kv_cache, cu_seqlens_q, seqused_k, block_table, *args):
    return torch.empty(query.shape, dtype=torch.bfloat16, device=query.device)


class RMSNorm(nn.Module):
    """FlashMap's RMSNorm: ``x * bf16(weight * rsqrt(mean(x^2) + eps))``."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, frequency_dim: int | None = None):
        super().__init__()
        self.frequency_dim = frequency_dim or dim
        self.linear_1 = nn.Linear(self.frequency_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.frequency_dim)
        emb = emb.to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(emb)))


class AdaLN(nn.Module):
    """AdaLN-zero's condition projection; returns ``(scale, shift, gate)``."""

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 3 * dim)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.linear(F.silu(cond)).chunk(3, dim=-1))


class FlashMapAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)


class FlashMapMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)


class FlashMapLayer(nn.Module):
    def __init__(self, config, layer_prefix: str, vllm_config: VllmConfig):
        super().__init__()
        dim, cond_dim = config.hidden_size, config.timestep_embed_dim
        heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or dim // heads
        self.self_attn_adaln = AdaLN(cond_dim, dim)
        self.self_attn = FlashMapAttention(dim, heads, head_dim, config.qk_norm_eps)
        self.cross_attn_adaln = AdaLN(cond_dim, dim)
        self.cross_attn = FlashMapAttention(dim, heads, head_dim, config.qk_norm_eps)
        self.mlp_adaln = AdaLN(cond_dim, dim)
        self.mlp = FlashMapMLP(dim, config.intermediate_size)
        # The cross-attention context lives in vLLM's paged KV cache; this layer
        # only owns the cache (the draft step reads it with its own kernels).
        self.cross_attn.attn = Attention(
            heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=heads,
            cache_config=vllm_config.cache_config,
            prefix=f"{layer_prefix}.cross_attn.attn",
        )


class FlashMapModel(nn.Module):
    """The FlashMap weights under their canonical names. ``embed_tokens`` is the
    target's, attached at load (``load_dflash_model``)."""

    def __init__(self, config, vllm_config: VllmConfig, prefix: str, start: int):
        super().__init__()
        dim, target_hidden = config.hidden_size, config.target_hidden_size
        eps = config.rms_norm_eps
        num_aux = len(config.eagle_aux_hidden_state_layer_ids)
        self.fc = nn.Linear(num_aux * target_hidden, dim, bias=False)
        self.hidden_norm = RMSNorm(dim, eps)
        self.anchor_proj = nn.Linear(target_hidden, dim)
        self.anchor_norm = RMSNorm(dim, eps)
        self.state_norm = RMSNorm(target_hidden, eps)
        self.noise_norm = RMSNorm(target_hidden, eps)
        self.noise_gate = nn.Parameter(torch.ones(1), requires_grad=False)
        self.x_embed = nn.Linear(target_hidden, dim)
        frequency_dim = getattr(config, "timestep_frequency_dim", None)
        self.time_embed = TimestepEmbedding(config.timestep_embed_dim, frequency_dim)
        self.jump_embed = TimestepEmbedding(config.timestep_embed_dim, frequency_dim)
        self.layers = nn.ModuleList(
            [
                FlashMapLayer(config, f"{prefix}.layers.{start + i}", vllm_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_adaln = AdaLN(config.timestep_embed_dim, dim)
        self.head = nn.Linear(dim, target_hidden)


class FlashMapDraftModel(nn.Module):
    # Scales the starting noise; tests set 0 to compare deterministic drafts.
    noise_scale = 1.0
    # torch.compile the draft step's prologue and DiT (fuses elementwise chains).
    compile = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError(
                "FlashMap drafting supports tensor parallel size 1 only (its "
                "load-time folds read the whole target embedding and LM head)."
            )
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "FlashMap drafting needs vLLM's V2 model runner; unset "
                "VLLM_USE_V2_MODEL_RUNNER=0."
            )
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.config = config = speculative_config.draft_model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        if self.dtype != torch.bfloat16:
            raise ValueError(f"FlashMap drafting runs in bfloat16, not {self.dtype}")
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = int(config.block_size)
        # FLASHMAP_NUM_STEPS overrides the checkpoint's number of jumps.
        self.num_steps = int(os.environ.get("FLASHMAP_NUM_STEPS", config.num_steps))
        self.time_grid = [float(t) for t in config.time_grids[self.num_steps - 1]]
        start = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = FlashMapModel(
            config, vllm_config, f"{prefix}.model" if prefix else "model", start
        )
        # Draft vocabulary: a restricted head drafts over d2t's tokens only.
        self.vocab_size = int(config.vocab_size)
        draft_vocab_size = int(
            getattr(config, "draft_vocab_size", None) or self.vocab_size
        )
        self.d2t: nn.Parameter | None = None
        if draft_vocab_size != self.vocab_size:
            self.d2t = nn.Parameter(
                torch.zeros(draft_vocab_size, dtype=torch.long), requires_grad=False
            )
        self._pending_context: tuple[torch.Tensor, ...] | None = None
        self._cache_args = None
        self._noise_keys: tuple[torch.Tensor, torch.Tensor] | None = None
        # FLASHMAP_FP8=1: e4m3 copies of the large GEMM weights (per output
        # channel), activations quantized per token (sm89+).
        self.fp8_weights = os.environ.get("FLASHMAP_FP8", "0") == "1"
        self.fp8_linears: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.kv_fp8 = False
        self.kv_scales: torch.Tensor | None = None
        # FLASHMAP_CANDIDATES=N: draft up to N blocks per request from
        # independent noise and propose the path most of them share (greedy
        # drafting); a batch of B requests drafts min(N, BUDGET // B) each.
        self.max_candidates = int(os.environ.get("FLASHMAP_CANDIDATES", "1"))
        self.candidate_budget = int(os.environ.get("FLASHMAP_CANDIDATE_BUDGET", "32"))
        if self.max_candidates > 1 and self.num_steps != 1:
            raise ValueError("FLASHMAP_CANDIDATES needs one-jump drafting")

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.layers

    # ------------------------------------------------------------------ loading

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for checkpoint_name, tensor in weights:
            if checkpoint_name in ("embed_tokens.weight", "lm_head.weight", "t2d"):
                continue  # the target's own, shared at load
            if checkpoint_name in ("ids", "counts"):
                # A release folder's draft_vocab.safetensors: the target's
                # tokens, most frequent first; the draft head keeps the first
                # draft_vocab_size of them.
                if checkpoint_name == "ids" and self.d2t is not None:
                    size = self.d2t.shape[0]
                    hot = tensor[:size].to(torch.int64).sort().values
                    self.d2t.data.copy_(hot - torch.arange(size, device=hot.device))
                    loaded.add("d2t")
                continue
            name = (
                checkpoint_name
                if checkpoint_name == "d2t"
                else "model." + (checkpoint_name)
            )
            param = params.get(name)
            if param is None:
                raise ValueError(f"unexpected FlashMap weight {checkpoint_name!r}")
            if param.shape != tensor.shape:
                raise ValueError(
                    f"FlashMap weight {checkpoint_name!r} has shape "
                    f"{tuple(tensor.shape)}, expected {tuple(param.shape)}"
                )
            param.data.copy_(tensor)
            loaded.add(name)
        # The attention layers' own (KV-cache scale) parameters are not weights.
        missing = {name for name in params if ".attn." not in name} - loaded
        if missing:
            raise ValueError(f"FlashMap checkpoint is missing {sorted(missing)}")
        return set(params)

    @torch.no_grad()
    def prepare(self) -> None:  # noqa: C901
        """Precompute what the draft step needs, once the target's embedding
        (``model.embed_tokens``) and LM head (``lm_head``) are attached."""
        model = self.model
        embed_tokens = model.embed_tokens.weight[: self.vocab_size]
        lm_head = self.lm_head.weight[: self.vocab_size]
        device, dtype = model.fc.weight.device, self.dtype
        attn = self.layers[0].cross_attn
        self.num_heads, self.head_dim = attn.num_heads, attn.head_dim
        self.dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self._check_attention()

        # AdaLN modulations for each jump (s, t) of the grid.
        self.modulations, self.final_modulations = [], []
        grid = torch.tensor(self.time_grid, device=device, dtype=torch.float32)
        scale = float(self.config.time_scale)
        for step in range(self.num_steps):
            cond = model.time_embed(grid[step : step + 1] * scale) + model.jump_embed(
                grid[step + 1 : step + 2] * scale
            )
            per_layer = []
            for layer in self.layers:
                for adaln in (
                    layer.self_attn_adaln,
                    layer.cross_attn_adaln,
                    layer.mlp_adaln,
                ):
                    mod_scale, shift, gate = adaln(cond)
                    per_layer += [(1 + mod_scale), shift, gate]
            # (layers, 9, dim): (1 + scale, shift, gate) for self, cross, mlp.
            self.modulations.append(
                torch.cat(per_layer).view(len(self.layers), 9, -1).contiguous()
            )
            f_scale, f_shift, _ = model.final_adaln(cond)
            self.final_modulations.append(
                torch.cat([1 + f_scale, f_shift]).contiguous()
            )
        gate = model.noise_gate.float()
        if self.config.noise_gate == "jump":
            self.noise_gates = [
                float(gate * (grid[s + 1] - grid[s])) for s in range(self.num_steps)
            ]
        else:
            self.noise_gates = [float(gate)] * self.num_steps

        # embed_state at t = 0: the uniform posterior's embedding (bf16, as the
        # reference forms softmax(0) @ E) through state_norm.
        vocab = embed_tokens.shape[0]
        uniform = torch.full((1, vocab), 1.0 / vocab, device=device, dtype=dtype)
        self.first_hidden = model.x_embed(model.state_norm(uniform @ embed_tokens))[0]
        self.noise_embed, self.noise_eigvals = fold_noise(
            embed_tokens, model.noise_norm.weight, model.x_embed.weight
        )
        self.embed_weight = embed_tokens

        # Context keys/values of every layer in one GEMM; self-attention
        # queries, keys and values in one GEMM per layer.
        self.context_kv_weight = torch.cat(
            [
                torch.cat(
                    [layer.cross_attn.k_proj.weight, layer.cross_attn.v_proj.weight]
                )
                for layer in self.layers
            ]
        )
        self.context_k_norm = torch.stack(
            [layer.cross_attn.k_norm.weight.float() for layer in self.layers]
        ).contiguous()
        self.self_qkv_weights = [
            torch.cat(
                [
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                ]
            )
            for layer in self.layers
        ]
        for layer in self.layers:  # drop the now-duplicated projections
            for proj in (
                layer.cross_attn.k_proj,
                layer.cross_attn.v_proj,
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                proj.weight = nn.Parameter(
                    proj.weight.new_empty(0), requires_grad=False
                )
        rope = getattr(self.config, "rope_parameters", None) or {}
        self.cos_sin = rope_table(
            self.max_model_len + self.block_size + 1,
            self.head_dim,
            float(rope.get("rope_theta", getattr(self.config, "rope_theta", 1e4))),
            device,
        )

        # The DiT head folded into the LM head (no nonlinearity between them).
        self.hot_ids = None
        if self.d2t is not None:
            self.hot_ids = torch.arange(self.d2t.shape[0], device=device) + self.d2t
        head = model.head
        self.head_weight, self.head_bias = fold_head(
            lm_head, head.weight, head.bias, self.hot_ids
        )
        self.jump_head = None
        if self.num_steps > 1:
            # Between jumps the state mixes in a full-vocabulary distribution.
            self.jump_head = (
                (self.head_weight, self.head_bias)
                if self.hot_ids is None
                else fold_head(lm_head, head.weight, head.bias)
            )
        self._prologue_fn = self._prologue
        self._one_jump_fn = self._one_jump
        self._dit_fn = self._dit
        if self.compile:
            self._prologue_fn = torch.compile(
                self._prologue, dynamic=True, fullgraph=True
            )
            self._one_jump_fn = torch.compile(
                self._one_jump, dynamic=True, fullgraph=True
            )
            self._dit_fn = torch.compile(self._dit, dynamic=True, fullgraph=True)
        if self.fp8_weights:
            self._quantize_weights()
        torch.cuda.empty_cache()
        logger.info(
            "FlashMap draft: %d jump(s) on grid %s, draft vocabulary %d",
            self.num_steps,
            self.time_grid,
            self.head_weight.shape[0],
        )

    def _quantize_weights(self) -> None:
        """e4m3 copies (per output channel) of the draft step's large GEMM
        weights, for ``_lin``."""
        if not current_platform.has_device_capability(89):
            raise ValueError("FLASHMAP_FP8=1 needs fp8 GEMMs (Ada or Hopper GPUs)")
        weights = {
            "fc": self.model.fc.weight,
            "context_kv": self.context_kv_weight,
            "noise": self.noise_embed,
            "head": self.head_weight,
        }
        for i, layer in enumerate(self.layers):
            weights[f"self_qkv.{i}"] = self.self_qkv_weights[i]
            weights[f"self_o.{i}"] = layer.self_attn.o_proj.weight
            weights[f"cross_q.{i}"] = layer.cross_attn.q_proj.weight
            weights[f"cross_o.{i}"] = layer.cross_attn.o_proj.weight
            weights[f"fc1.{i}"] = layer.mlp.fc1.weight
            weights[f"fc2.{i}"] = layer.mlp.fc2.weight
        fp8 = current_platform.fp8_dtype()
        self.fp8_linears = {
            name: quantize_fp8_per_channel(weight, fp8)
            for name, weight in weights.items()
        }

    def _check_attention(self) -> None:
        for layer in self.layers:
            attn = layer.cross_attn.attn
            version = getattr(attn.impl, "vllm_flash_attn_version", None)
            if version is None:
                raise ValueError(
                    "FlashMap drafting needs the FlashAttention backend, got "
                    f"{type(attn.impl).__name__}"
                )
            kv_dtype = str(getattr(attn, "kv_cache_dtype", "auto"))
            if kv_dtype in ("fp8", "fp8_e4m3"):
                if version < 3:  # noqa: PLR2004
                    raise ValueError(
                        "An fp8 FlashMap draft KV cache needs FlashAttention 3 "
                        "(Hopper)."
                    )
                self.kv_fp8 = True
            elif kv_dtype not in ("auto", "bfloat16"):
                raise ValueError(
                    "FlashMap drafting supports a bf16 or fp8_e4m3 draft KV cache, "
                    f"got kv_cache_dtype={kv_dtype!r}"
                )
        self.fa_version = version
        if self.kv_fp8:
            # Per-tensor scales of vLLM's fp8 KV caches (1 unless calibrated).
            self.kv_scales = torch.stack(
                [
                    torch.stack(
                        [
                            layer.cross_attn.attn._k_scale.float().reshape(()),  # noqa: SLF001
                            layer.cross_attn.attn._v_scale.float().reshape(()),  # noqa: SLF001
                        ]
                    )
                    for layer in self.layers
                ]
            ).to(self.model.fc.weight.device)
            self.q_scales = [
                float(layer.cross_attn.attn._q_scale.float())  # noqa: SLF001
                for layer in self.layers
            ]

    def set_noise_keys(self, seeds: torch.Tensor, rows: torch.Tensor) -> None:
        """Key each query row's noise by ``seeds[rows[i]]`` and its position."""
        self._noise_keys = (seeds, rows)

    # ------------------------------------------------------------------ context

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._lin(hidden_states, "fc", self.model.fc.weight)

    def _lin(
        self,
        x: torch.Tensor,
        name: str,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``F.linear``, or vLLM's fp8 scaled mm for a weight quantized at load
        (per output channel, activations per token)."""
        quantized = self.fp8_linears.get(name)
        if quantized is None:
            return F.linear(x, weight, bias)
        w_q, w_scale = quantized
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        x_q, x_scale = ops.scaled_fp8_quant(rows, use_per_token_if_dynamic=True)
        out = ops.cutlass_scaled_mm(
            x_q, w_q, scale_a=x_scale, scale_b=w_scale, out_dtype=x.dtype, bias=bias
        )
        return out.view(*x.shape[:-1], w_q.shape[1])

    def cache_layout(self):
        """Data pointers and strides of every layer's KV cache, or None before
        the caches are allocated."""
        caches = [layer.cross_attn.attn.kv_cache for layer in self.layers]
        if not all(isinstance(c, torch.Tensor) and c.numel() for c in caches):
            return None
        ptrs = [c.data_ptr() for c in caches]
        if self._cache_args is None or self._cache_args[0] != ptrs:
            expected = (
                (torch.uint8, torch.float8_e4m3fn) if self.kv_fp8 else (torch.bfloat16,)
            )
            if caches[0].dtype not in expected:
                raise ValueError(
                    f"unexpected FlashMap draft KV cache dtype {caches[0].dtype}"
                )
            key, value = caches[0].transpose(1, 2).split(self.head_dim, dim=-1)
            assert key.stride(-1) == 1
            assert all(c.stride() == caches[0].stride() for c in caches)
            strides = (
                key.stride(0),
                key.stride(2),
                key.stride(1),
                (value.data_ptr() - key.data_ptr()) // key.element_size(),
            )
            ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device=caches[0].device)
            self._cache_args = (ptrs, ptr_tensor, strides, key.shape[1])
        return self._cache_args

    def _write(self, kv: torch.Tensor, positions, slots) -> None:
        layout = self.cache_layout()
        if layout is None:
            return
        _, ptrs, strides, block_size = layout
        write_context_kv(
            kv.view(-1, len(self.layers), 2, self.num_heads, self.head_dim),
            positions,
            slots,
            self.context_k_norm,
            self.cos_sin,
            ptrs,
            strides,
            block_size,
            self.layers[0].cross_attn.k_norm.eps,
            self.kv_scales,
        )

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        context = self.model.hidden_norm(context_states)
        kv = self._lin(context, "context_kv", self.context_kv_weight)
        if context_slot_mapping is not None:
            self._write(kv, context_positions, context_slot_mapping)

    def defer_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
    ) -> None:
        """``precompute_and_store_context_kv``, done by the next ``forward``
        together with its anchors (one GEMM over the context K/V weight)."""
        self._pending_context = (
            context_states,
            context_positions,
            context_slot_mapping,
        )

    def _prologue(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        context: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ):
        """Keys/values, positions and slots to write (the pending context, then
        each request's anchor) and each request's context length ``p + 1``."""
        model = self.model
        blocks = input_ids.shape[0] // self.block_size
        anchors = input_ids.view(blocks, self.block_size)[:, 0]
        anchor_positions = positions.view(blocks, self.block_size)[:, 0]
        slots = slot_mapping.view(blocks, self.block_size)[:, 0]
        rows = model.anchor_norm(
            model.anchor_proj(F.embedding(anchors, self.embed_weight))
        )
        write_positions = anchor_positions
        if context is not None:
            states, context_positions, context_slots = context
            rows = torch.cat([model.hidden_norm(states), rows])
            write_positions = torch.cat([context_positions, anchor_positions])
            slots = torch.cat([context_slots, slots])
        kv = self._lin(rows, "context_kv", self.context_kv_weight)
        # The context is positions 0..p. (The DFlash metadata's seq_lens count
        # the block's other rows too, clamped to the model length.)
        seqused_k = torch.where(
            seq_lens > 0, anchor_positions.to(seq_lens.dtype) + 1, 0
        )
        return kv, write_positions, slots, seqused_k

    # ------------------------------------------------------------------ drafting

    def _dit(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        modulations: torch.Tensor,
        final_modulations: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attention: tuple | None,
    ) -> torch.Tensor:
        """The DiT on ``hidden (tokens, dim)`` for query rows at ``positions``
        (slots at ``positions + 1``) with one jump's modulations; returns the
        final AdaLN output."""
        tokens = hidden.shape[0]
        blocks = tokens // self.block_size
        heads, head_dim, dim = self.num_heads, self.head_dim, self.dim
        eps = self.config.adaln_eps
        slot_positions = positions + 1
        for layer_id, layer in enumerate(self.layers):
            mods = modulations[layer_id]
            self_attn, cross_attn = layer.self_attn, layer.cross_attn
            normed = F.layer_norm(hidden, (dim,), eps=eps) * mods[0] + mods[1]
            qkv = self._lin(
                normed, f"self_qkv.{layer_id}", self.self_qkv_weights[layer_id]
            )
            attn = block_self_attention(
                qkv.view(blocks, self.block_size, 3, heads, head_dim),
                self_attn.q_norm.weight,
                self_attn.k_norm.weight,
                self.cos_sin,
                slot_positions,
                self_attn.q_norm.eps,
            )
            hidden = hidden + mods[2] * self._lin(
                attn, f"self_o.{layer_id}", self_attn.o_proj.weight
            )

            normed = F.layer_norm(hidden, (dim,), eps=eps) * mods[3] + mods[4]
            query = rms_norm_rope(
                self._lin(normed, f"cross_q.{layer_id}", cross_attn.q_proj.weight).view(
                    tokens, heads, head_dim
                ),
                cross_attn.q_norm.weight,
                self.cos_sin,
                slot_positions,
                cross_attn.q_norm.eps,
            )
            if attention is None:  # profiling run: no KV cache yet
                context = torch.zeros_like(query)
            else:
                cu_seqlens_q, seqused_k, block_table, schedule, queries = attention
                descales = None
                if self.kv_fp8:
                    q_scale = self.q_scales[layer_id]
                    query = (query.float() / q_scale).to(torch.float8_e4m3fn)
                    descales = torch.stack(
                        [
                            self.kv_scales.new_full((), q_scale),
                            self.kv_scales[layer_id, 0],
                            self.kv_scales[layer_id, 1],
                        ]
                    )[:, None, None].expand(3, seqused_k.shape[0], heads)
                context = paged_cross_attention(
                    query,
                    kv_caches[layer_id],
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    schedule,
                    descales,
                    queries,
                    self.max_model_len,
                    self.softmax_scale,
                    self.fa_version,
                )
            hidden = hidden + mods[5] * self._lin(
                context.view(tokens, dim),
                f"cross_o.{layer_id}",
                cross_attn.o_proj.weight,
            )

            normed = F.layer_norm(hidden, (dim,), eps=eps) * mods[6] + mods[7]
            mlp = layer.mlp
            hidden = hidden + mods[8] * self._lin(
                F.gelu(
                    self._lin(normed, f"fc1.{layer_id}", mlp.fc1.weight, mlp.fc1.bias)
                ),
                f"fc2.{layer_id}",
                mlp.fc2.weight,
                mlp.fc2.bias,
            )
        normed = F.layer_norm(hidden, (dim,), eps=eps)
        return normed * final_modulations[0] + final_modulations[1]

    def _one_jump(
        self,
        z: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attention: tuple | None,
    ) -> torch.Tensor:
        """One jump from noise ``z (tokens, hidden)``: its embedded state (the
        uniform posterior's constant plus the folded noise read-out), then the
        DiT."""
        rms = torch.rsqrt((z * z) @ self.noise_eigvals + self.model.noise_norm.eps)
        noise = self._lin(z.to(self.dtype), "noise", self.noise_embed)
        hidden = (
            self.first_hidden
            + (self.noise_gates[0] * rms)[:, None].to(self.dtype) * noise
        )
        return self._dit(
            hidden,
            positions,
            self.modulations[0],
            self.final_modulations[0],
            kv_caches,
            attention,
        )

    def _embed_state(self, state: torch.Tensor, step: int) -> torch.Tensor:
        """``x_embed(embed_state(state, s, t))`` of a vocabulary-sized float32
        ``state`` at jump ``step``, as the reference computes it."""
        model, weight = self.model, self.embed_weight
        start = self.time_grid[step]
        evidence = start / max(1.0 - start, 1e-3) ** 2
        posterior = (state * evidence).softmax(dim=-1)
        both = torch.cat([posterior, state]).to(weight.dtype) @ weight
        posterior_emb, state_emb = both.split(state.shape[0])
        embedded = model.state_norm(posterior_emb) + (
            self.noise_gates[step] * model.noise_norm(state_emb)
        ).to(weight.dtype)
        return model.x_embed(embedded)

    def _noise(
        self,
        positions: torch.Tensor,
        n_cols: int,
        requests: int = 0,
        candidates: int = 1,
    ) -> torch.Tensor:
        tokens = positions.shape[0]
        if self._noise_keys is None:
            noise = torch.randn(
                tokens, n_cols, device=positions.device, dtype=torch.float32
            )
            return noise if self.noise_scale == 1.0 else noise * self.noise_scale
        seeds, rows = self._noise_keys
        if candidates > 1:
            rows = (
                rows[: requests * self.block_size]
                .view(requests, 1, self.block_size)
                .expand(requests, candidates, self.block_size)
                .reshape(-1)
            )
        return seeded_normal(seeds, rows[:tokens], positions, n_cols, self.noise_scale)

    def num_candidates(self, requests: int) -> int:
        """Candidate blocks per request in a batch of ``requests``."""
        if self.max_candidates <= 1:
            return 1
        return max(
            1, min(self.max_candidates, self.candidate_budget // max(requests, 1))
        )

    def _cross_attention_schedule(self, metadata, seqused_k, candidates: int = 1):
        """FlashAttention 3's tile schedule for the step's cross-attention,
        shared by every layer (it depends only on shapes and lengths)."""
        if self.fa_version != 3 or get_scheduler_metadata is None:  # noqa: PLR2004
            return None
        caches = self.layers[0].cross_attn.attn.kv_cache
        return get_scheduler_metadata(
            batch_size=seqused_k.shape[0],
            max_seqlen_q=self.block_size * candidates,
            max_seqlen_k=self.max_model_len,
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_heads,
            headdim=self.head_dim,
            cache_seqlens=seqused_k,
            qkv_dtype=current_platform.fp8_dtype() if self.kv_fp8 else self.dtype,
            cu_seqlens_q=metadata.query_start_loc * candidates,
            page_size=caches.shape[2],
            causal=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        tokens = input_ids.shape[0]
        requests = tokens // self.block_size
        candidates = self.num_candidates(requests)
        kv_caches = [layer.cross_attn.attn.kv_cache for layer in self.layers]
        metadata = get_forward_context().attn_metadata
        pending, self._pending_context = self._pending_context, None
        attention = None
        if metadata is not None:
            metadata = metadata[self.layers[0].cross_attn.attn.layer_name]
            kv, write_positions, write_slots, seqused_k = self._prologue_fn(
                input_ids,
                positions,
                metadata.seq_lens,
                metadata.slot_mapping[:tokens],
                pending,
            )
            self._write(kv, write_positions, write_slots)
            # A request's candidates are adjacent query rows of its
            # cross-attention, so its context is read once for all of them.
            attention = (
                metadata.query_start_loc * candidates,
                seqused_k,
                metadata.block_table,
                self._cross_attention_schedule(metadata, seqused_k, candidates),
                self.block_size * candidates,
            )

        if self.num_steps == 1:
            if candidates == 1:
                z = self._noise(positions, self.noise_embed.shape[1])
                return self._one_jump_fn(z, positions, kv_caches, attention)
            # Rows (request, candidate, slot); noise keyed per candidate.
            positions = (
                positions.view(requests, 1, self.block_size)
                .expand(requests, candidates, self.block_size)
                .reshape(-1)
            )
            offsets = torch.arange(
                candidates, device=positions.device
            ).repeat_interleave(self.block_size)
            noise_keys = positions * candidates + offsets.repeat(requests)
            z = self._noise(noise_keys, self.noise_embed.shape[1], requests, candidates)
            return self._one_jump_fn(z, positions, kv_caches, attention)

        # Several jumps (Algorithm 8): x <- X_{t_n, t_n+1}(x) between jumps.
        state = self._noise(positions, self.vocab_size)
        for step in range(self.num_steps):
            hidden = self._dit_fn(
                self._embed_state(state, step),
                positions,
                self.modulations[step],
                self.final_modulations[step],
                kv_caches,
                attention,
            )
            if step + 1 < self.num_steps:
                weight, bias = self.jump_head
                delta = F.linear(hidden, weight, bias).float().softmax(dim=-1)
                start, end = self.time_grid[step], self.time_grid[step + 1]
                state = ((1.0 - end) * state + (end - start) * delta) / (1.0 - start)
        return hidden

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Logits over the draft vocabulary (``hot_ids`` when restricted)."""
        return self._lin(hidden_states, "head", self.head_weight, self.head_bias)
