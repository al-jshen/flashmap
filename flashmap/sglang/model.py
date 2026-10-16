# coding=utf-8
"""FlashMap as an sglang draft model, served by sglang's DSPARK worker.

A FlashMap release folder is served as it is (with ``flashmap[sglang]``
installed, its sglang plugin registers this model):

    sglang serve --model-path Qwen/Qwen3-8B \\
        --speculative-algorithm DSPARK \\
        --speculative-draft-model-path j-shen/FlashMap-Qwen3-8B

DSPARK drives a block draft through a small protocol, implemented here:

* ``write_target_hidden_kv``: DSPARK hands over the target's captured hidden
  states for every committed position. FlashMap's cross-attention context (the
  keys and values each DiT layer projects from them) goes into the draft KV
  pool, which sglang sizes from the config to the DiT's layers, heads and
  head_dim. All
  layers are projected in one matmul and written, k-normed and rotated, by
  sglang's fused DSpark kernel, which also skips the uncommitted verify
  columns, so the write never synchronizes with the host.
* ``forward``: one block of ``block_size`` drafts after each request's anchor
  (the last verified token), for the whole batch in one pass that DSPARK
  captures in a CUDA graph. It is compiled with torch.compile, and
  - the anchor's own context entry is written into its verify slot, so the
    cross-attention reads every request's paged context directly
    (FlashAttention with sglang's ``req_to_token`` rows as page tables, one
    tile schedule for all layers);
  - self-attention within each block is one Triton kernel
    (:mod:`.kernels`), its q/k/v one matmul;
  - every AdaLN modulation depends only on the jump's times, which are fixed,
    so they are computed once at load;
  - one jump starts from noise at ``t = 0``, where the state's posterior is
    uniform whatever the noise: its embedding is a constant, and the noise
    enters only through ``x @ E ~ N(0, E^T E)``, drawn as ``z @ L^T`` with
    ``L L^T = E^T E`` (Cholesky), so no vocabulary-sized noise is ever formed.
    ``FLASHMAP_FOLDED_NOISE=1`` (opt-in, measured on Qwen3-8B only) draws it from the
    eigendecomposition instead, which folds the noise's RMSNorm and ``x_embed``
    into one GEMM (``_folded_noise``);
  - more jumps (``FLASHMAP_NUM_STEPS``) take their vocabulary-sized inputs from
    a bank of precomputed noise draws (:mod:`.jump_bank`) and read
    only the embedding rows of each jump's top predictions.
* ``compute_base_logits``: the DiT's output head (dim -> hidden) folded into the
  target's LM head (hidden -> vocab) as one smaller matmul, restricted to the
  draft vocabulary (the target's most frequent tokens, the config's
  ``draft_vocab_size`` of the ranking in ``draft_vocab.safetensors``); drafted
  indices map back to token ids in the draft graph.

The draft borrows the target's token embedding and LM head (DSPARK attaches
them); its Markov head is an identity (:mod:`.sampler`), so DSPARK
samples and verifies the map's logits unchanged, in one pass per block. The
DiT's modules and weights are :class:`flashmap.FlashMapDraftModel`'s.

Environment knobs: ``FLASHMAP_NUM_STEPS`` (jumps per draft; default the
config's ``num_steps``), ``FLASHMAP_DRAFT_VOCAB`` (0: the full vocabulary,
``<k>``: the ranking's top k), ``FLASHMAP_JUMP_BANK`` /
``FLASHMAP_JUMP_TOPK`` (bank draws, 8192; support per jump, 16),
``FLASHMAP_COMPILE`` (0 disables torch.compile), ``FLASHMAP_COMPILE_MODE``
(``max-autotune-no-cudagraphs``; modes that capture their own CUDA graphs
would nest inside sglang's), and two opt-in variants measured on Qwen3-8B
only: ``FLASHMAP_FOLDED_NOISE=1`` (above) and ``FLASHMAP_FUSED_SAMPLER=1``
(:mod:`.sampler`), ``FLASHMAP_FP8=1`` (FP8 weights) and ``FLASHMAP_TREE_N``
(tree drafting, :mod:`.tree_worker`).

Supported: sglang 0.5.20 (``SUPPORTED_SGLANG``), tensor parallelism 1, a bf16
or e4m3 draft KV pool, an unpadded target vocabulary without muP logit scaling.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from . import check_sglang_version

check_sglang_version()

from sgl_kernel.flash_attn import (  # noqa: E402
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from sglang.kernels.ops.speculative.dspark.fused_kv_write import (  # noqa: E402
    fused_kv_norm_rope_write,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: E402
from sglang.srt.model_executor.forward_context import (  # noqa: E402
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_exec, get_model  # noqa: E402
from sglang.srt.speculative.dspark_components import dspark_worker_v2  # noqa: E402
from torch import nn  # noqa: E402

from ..modeling_flashmap import FlashMapDraftModel as FlashMapDiT  # noqa: E402
from .jump_bank import JumpNoiseBank, top_predictions  # noqa: E402
from .kernels import (  # noqa: E402
    block_self_attention,
    fp8_linear,
    kv_norm_rope_write_fp8,
    quantize_weight_fp8,
    rms_norm,
    rms_norm_rope,
)
from .sampler import FlashMapDraftSampler, IdentityMarkov  # noqa: E402
from .tree import build_tree_abi  # noqa: E402

logger = logging.getLogger(__name__)


def rope_freqs(positions: torch.Tensor, head_dim: int, theta: float) -> torch.Tensor:
    """The rotary ``(cos, sin)`` table ``(..., head_dim // 2, 2)`` at ``positions``,
    in float32, as the kernels read it (the angles of ``modeling_flashmap.rope_tables``)."""
    inv_freq = 1.0 / theta ** (
        torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim
    )
    angles = positions.float()[..., None] * inv_freq
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


# A custom op, so torch.compile traces the draft step through it in one graph.
@torch.library.custom_op("flashmap_sglang::paged_cross_attention", mutates_args=())
def _paged_cross_attention(
    query: torch.Tensor,
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    scheduler_metadata: torch.Tensor,
) -> torch.Tensor:
    """``query (bs, K, heads, head_dim)`` onto each request's first
    ``cache_seqlens`` pool rows, listed by ``page_table`` (page size 1)."""
    return flash_attn_with_kvcache(
        q=query,
        k_cache=key_buffer.unsqueeze(1),
        v_cache=value_buffer.unsqueeze(1),
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        causal=False,
        scheduler_metadata=scheduler_metadata,
    )


@_paged_cross_attention.register_fake
def _(query, key_buffer, value_buffer, cache_seqlens, page_table, scheduler_metadata):
    return torch.empty_like(query)


@torch.library.custom_op(
    "flashmap_sglang::paged_cross_attention_fp8", mutates_args=()
)
def _paged_cross_attention_fp8(
    query: torch.Tensor,
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    scheduler_metadata: torch.Tensor,
) -> torch.Tensor:
    """``_paged_cross_attention`` onto an e4m3 pool (written with scale 1): the
    bf16 query is quantized per (request, head) and every operand descaled."""
    q_scale = query.float().abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    query8 = (query.float() / q_scale[:, None, :, None]).to(key_buffer.dtype)
    ones = torch.ones_like(q_scale)
    return flash_attn_with_kvcache(
        q=query8,
        k_cache=key_buffer.unsqueeze(1),
        v_cache=value_buffer.unsqueeze(1),
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        causal=False,
        scheduler_metadata=scheduler_metadata,
        q_descale=q_scale,
        k_descale=ones,
        v_descale=ones,
    ).to(query.dtype)


@_paged_cross_attention_fp8.register_fake
def _(query, key_buffer, value_buffer, cache_seqlens, page_table, scheduler_metadata):
    return torch.empty_like(query)


def _cross_attention_schedule(query_shape, cache_seqlens, page_table, dtype):
    """FlashAttention's tile schedule for ``_paged_cross_attention``, computed
    once per draft step for all layers (it depends only on shapes and lengths)."""
    bsz, block_size, heads, head_dim = query_shape
    return get_scheduler_metadata(
        batch_size=bsz,
        max_seqlen_q=block_size,
        max_seqlen_k=page_table.shape[1],
        num_heads=heads,
        num_heads_k=heads,
        headdim=head_dim,
        cache_seqlens=cache_seqlens,
        qkv_dtype=dtype,
        page_size=1,
        causal=False,
    )


def _server_setting(read, default):
    """A setting from sglang's runtime context, or ``default`` outside a server
    (offline checks)."""
    try:
        value = read()
    except Exception:
        return default
    return default if value is None else value


class _FoldedHead(nn.Module):
    """The draft's head as DSPARK reads it: the DiT head folded into the
    target's LM head over the draft vocabulary. ``org_vocab_size`` is the full
    vocabulary, which DSPARK's sampling buffers are sized to."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, vocab_size: int):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        self.org_vocab_size = int(vocab_size)


class FlashMapDraftModel(FlashMapDiT):
    """:class:`flashmap.FlashMapDraftModel` (its modules and weights, loaded from
    a release folder as they are) with sglang's draft protocol and the fused
    draft step below."""

    # Read by the DSPARK worker and the hooks below.
    flashmap = True
    uses_own_vocab_modules = False
    supports_pre_gather_target_hidden_projection = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        del quant_config, prefix
        if getattr(config, "flow_map", True) is False:
            raise ValueError(
                "this checkpoint is a stage-1 flow (flow_map: false), which drafts "
                "only by integrating over many steps; serve the flow map instead"
            )
        super().__init__(config)
        fm = self.config
        self.block_size = int(fm.block_size)
        self.vocab_size = fm.vocab_size
        self.target_hidden_size = fm.target_hidden_size
        self.dim = fm.hidden_size
        self.num_heads = fm.num_attention_heads
        self.gamma = self.block_size
        self.num_stages = fm.num_hidden_layers
        self.rope_theta = fm.rope_theta
        self.rms_norm_eps = fm.rms_norm_eps
        self.qk_norm_eps = fm.qk_norm_eps
        self.layer_norm_eps = fm.layer_norm_eps

        # DSPARK's draft protocol.
        self.sample_from_anchor = True
        self.markov_head = IdentityMarkov(vocab_size=self.vocab_size)
        self.confidence_head = None
        self.logits_mup_width_multiplier = None
        self.lm_head = None  # the folded head, once the target's is attached
        self.embedding = None  # the target's token embedding (not a submodule)
        self.hot_ids = None  # draft vocabulary -> token ids (None: all tokens)
        self.folded_head = None

        self.num_steps = int(os.environ.get("FLASHMAP_NUM_STEPS") or fm.num_steps)
        self.grid = [float(t) for t in fm.jump_grids[str(self.num_steps)]]
        assert self.grid[0] == 0.0 and self.grid[-1] == 1.0, self.grid
        self.jump_bank_size = int(os.environ.get("FLASHMAP_JUMP_BANK", "8192"))
        self.jump_topk = int(os.environ.get("FLASHMAP_JUMP_TOPK", "16"))
        self.jump_bank = None
        size = os.environ.get("FLASHMAP_DRAFT_VOCAB")
        size = int(size if size is not None else getattr(config, "draft_vocab_size", 0))
        self.draft_vocab_size = size if 0 < size < self.vocab_size else 0
        self._vocab_ranking = None  # draft_vocab.safetensors' ids
        # Scales the starting noise; checks set 0 for deterministic drafts.
        self.noise_temperature = 1.0
        self.folded_noise = os.environ.get("FLASHMAP_FOLDED_NOISE", "0") == "1"
        # FP8 (e4m3) copies of the large weights, per output channel, with the
        # activations quantized per token on the fly (_quantize_fp8).
        self.fp8 = os.environ.get("FLASHMAP_FP8", "0") == "1"
        self._fp8 = {}
        self._constants = None
        self._fused_write = None  # (pool id, kernel arguments)
        # Tree drafting (tree_worker.py): up to FLASHMAP_TREE_N candidate
        # blocks per request from independent noise in one draft pass, merged
        # into a trie cut to the verify width (block_size + 1 with the root); a
        # batch of bs requests drafts clamp(FLASHMAP_TREE_BUDGET // bs, 1, N).
        self.tree_max_n = int(os.environ.get("FLASHMAP_TREE_N", "0"))
        self.tree_budget = int(os.environ.get("FLASHMAP_TREE_BUDGET", "32"))
        self.tree_mode = self.tree_max_n >= 1
        self.tree_width = self.block_size + 1
        self._tree_buffers = None
        self._tree_scratch = None
        self._build_tree = build_tree_abi
        self._kv_fp8 = False  # an e4m3 draft KV pool (set with the pool)
        self._draft = self._draft_impl
        self._anchor_kv = self._anchor_kv_impl
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        tp_size = _server_setting(get_tensor_model_parallel_world_size, 1)
        if tp_size != 1:
            raise NotImplementedError(
                f"the FlashMap draft runs unsharded: serve with --tp-size 1 (got {tp_size})"
            )
        if os.environ.get("FLASHMAP_COMPILE", "1") == "1":
            mode = os.environ.get("FLASHMAP_COMPILE_MODE") or None
            if mode not in (None, "default", "max-autotune-no-cudagraphs"):
                raise ValueError(
                    f"FLASHMAP_COMPILE_MODE={mode!r} would capture CUDA graphs "
                    "inside sglang's; use default or max-autotune-no-cudagraphs"
                )
            self._draft = torch.compile(self._draft_impl, dynamic=True, mode=mode)
            self._anchor_kv = torch.compile(self._anchor_kv_impl, dynamic=True)
            # ~50 small index ops per step otherwise, inside the draft graph.
            self._build_tree = torch.compile(build_tree_abi, dynamic=True)
        logger.info(
            "FlashMap draft: %d jump(s) on %s, block_size=%d, draft vocabulary %s",
            self.num_steps,
            self.grid,
            self.block_size,
            self.draft_vocab_size or "full",
        )

    # --- loading -----------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """The folder's weights. sglang reads every ``*.safetensors`` file there,
        so ``draft_vocab.safetensors`` arrives too: its ``ids`` rank every token
        id, most frequent first (``counts`` are their frequencies)."""
        params = {**dict(self.named_parameters()), **dict(self.named_buffers())}
        loaded = set()
        for name, tensor in weights:
            if name in ("ids", "draft_vocab_ids"):
                self._vocab_ranking = tensor
                continue
            if name == "counts":
                continue
            if name not in params:
                raise KeyError(f"unexpected FlashMap weight {name!r}")
            with torch.no_grad():
                params[name].copy_(tensor.to(params[name].dtype))
            loaded.add(name)
        missing = sorted(
            n for n in params if n not in loaded and not n.startswith("markov_head.")
        )
        if missing:
            raise KeyError(f"FlashMap weights missing from the checkpoint: {missing}")
        layers = self.layers
        # Every layer's context keys and values in one matmul (rows [k; v] per
        # layer); each k_proj/v_proj weight becomes a view of its rows.
        self._ctx_kv_weight = torch.cat(
            [
                torch.cat(
                    [layer.cross_attn.k_proj.weight, layer.cross_attn.v_proj.weight]
                )
                for layer in layers
            ]
        )
        for i, layer in enumerate(layers):
            rows = self._ctx_kv_weight[2 * self.dim * i : 2 * self.dim * (i + 1)]
            layer.cross_attn.k_proj.weight.data = rows[: self.dim]
            layer.cross_attn.v_proj.weight.data = rows[self.dim :]
        # Likewise each self-attention's queries, keys and values.
        self._self_qkv_weights = []
        for layer in layers:
            attn = layer.self_attn
            qkv = torch.cat(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]
            )
            attn.q_proj.weight.data = qkv[: self.dim]
            attn.k_proj.weight.data = qkv[self.dim : 2 * self.dim]
            attn.v_proj.weight.data = qkv[2 * self.dim :]
            self._self_qkv_weights.append(qkv)
        # The norms' weights in float32, as the kernels take them.
        self._self_norm_weights = [
            (
                layer.self_attn.q_norm.weight.float(),
                layer.self_attn.k_norm.weight.float(),
            )
            for layer in layers
        ]
        self._cross_q_norm_weights = [
            layer.cross_attn.q_norm.weight.float() for layer in layers
        ]
        self._hidden_norm_weight = self.hidden_norm.weight.float()

    def attach_shared_modules(self, *, embed_tokens: nn.Module, lm_head: nn.Module):
        """DSPARK hands over the target's embedding and LM head; everything the
        draft step reads from them is precomputed here, before sglang sizes its
        KV pools. A target with tied embeddings (Qwen3-4B) hands over its
        embedding as the LM head; one without a head of its own falls back to
        the embedding too."""
        if self.logits_mup_width_multiplier:
            raise NotImplementedError("FlashMap does not support muP-scaled targets")
        embedding = embed_tokens.weight.detach()
        head = getattr(lm_head, "weight", None)
        head = embedding if head is None else head.detach()
        for name, weight in (("embedding", embedding), ("LM head", head)):
            # Rows past the vocabulary would be tensor-parallel padding.
            if weight.shape != (self.vocab_size, self.target_hidden_size):
                raise ValueError(
                    f"the target's {name} is {tuple(weight.shape)}, the draft "
                    f"expects ({self.vocab_size}, {self.target_hidden_size}); padded "
                    "or sharded vocabularies are not supported"
                )
        self.embedding = embedding
        self.prepare(head)

    @torch.no_grad()
    def prepare(self, lm_head_weight: torch.Tensor) -> None:
        if self.draft_vocab_size and self._vocab_ranking is None:
            logger.warning(
                "FlashMap: no draft_vocab.safetensors next to the weights; drafting "
                "over the full vocabulary"
            )
            self.draft_vocab_size = 0
        if self.draft_vocab_size:
            self.hot_ids = (
                self._vocab_ranking[: self.draft_vocab_size]
                .to(device=self.embedding.device, dtype=torch.int64)
                .sort()
                .values
            )
        self.folded_head = self._fold_lm_head(lm_head_weight, self.hot_ids)
        self.lm_head = _FoldedHead(*self.folded_head, self.vocab_size)
        self.markov_head.hot_ids = self.hot_ids
        if self.fp8:
            self._quantize_fp8()
        self._constants = self._jump_constants()
        if self.tree_mode:
            if self.num_steps != 1:
                raise NotImplementedError("tree drafting takes one jump")
            self._tree_buffers = self._alloc_tree_buffers(self._graph_max_bs())
        if self.num_steps > 1:
            self.jump_bank = JumpNoiseBank(
                self,
                self.grid,
                size=self.jump_bank_size,
                temperature=self.noise_temperature,
            )
            logger.info(
                "FlashMap: %d jumps from a bank of %d noise draws (%.0f MB), "
                "predictions cut to %d tokens",
                self.num_steps,
                self.jump_bank_size,
                self.jump_bank.nbytes() / 2**20,
                self.jump_topk,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _quantize_fp8(self) -> None:
        """FP8 copies of every large GEMM's weight: the DiT's linears, ``x_embed``,
        the folded head, ``fc`` and the context/anchor projections. Norms, the
        AdaLN constants and the noise draw (``z @ L^T``, whose input is the noise
        itself) stay bf16."""
        if torch.cuda.get_device_capability() < (8, 9):
            raise RuntimeError("FLASHMAP_FP8=1 needs FP8 tensor cores (Ada/Hopper)")
        weights = {
            "x_embed": (self.x_embed.weight, self.x_embed.bias),
            "fc": (self.fc.weight, None),
            "ctx_kv": (self._ctx_kv_weight, None),
            "anchor_proj": (self.anchor_proj.weight, self.anchor_proj.bias),
            "head": self.folded_head,
        }
        for i, layer in enumerate(self.layers):
            weights[f"qkv{i}"] = (self._self_qkv_weights[i], None)
            weights[f"self_o{i}"] = (layer.self_attn.o_proj.weight, None)
            weights[f"cross_q{i}"] = (layer.cross_attn.q_proj.weight, None)
            weights[f"cross_o{i}"] = (layer.cross_attn.o_proj.weight, None)
            weights[f"fc1_{i}"] = (layer.mlp.fc1.weight, layer.mlp.fc1.bias)
            weights[f"fc2_{i}"] = (layer.mlp.fc2.weight, layer.mlp.fc2.bias)
        for key, (weight, bias) in weights.items():
            self._fp8[key] = (*quantize_weight_fp8(weight), bias)

    def _linear(self, x, key: str, weight, bias=None) -> torch.Tensor:
        """``x @ weight^T + bias``, in FP8 with ``FLASHMAP_FP8=1``."""
        if self.fp8:
            return fp8_linear(x, *self._fp8[key])
        return F.linear(x, weight, bias)

    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The draft-vocabulary logits of the DiT's output (the folded head)."""
        return self._linear(hidden, "head", *self.folded_head)

    def _fold_lm_head(self, lm_head, rows: Optional[torch.Tensor], chunk: int = 8192):
        """The DiT head (dim -> hidden, with bias) and the LM head (hidden ->
        vocab) as one ``(vocab rows, dim)`` projection and bias; there is no
        nonlinearity between them (Qwen3 has no final-logit softcap). Folded
        ``chunk`` rows at a time in float32."""
        count = lm_head.shape[0] if rows is None else rows.numel()
        weight = lm_head.new_empty(count, self.dim)
        bias = lm_head.new_empty(count)
        head_weight, head_bias = self.head.weight.float(), self.head.bias.float()
        for start in range(0, count, chunk):
            if rows is None:
                part = lm_head[start : start + chunk].float()
            else:
                part = lm_head[rows[start : start + chunk]].float()
            weight[start : start + chunk] = (part @ head_weight).to(lm_head.dtype)
            bias[start : start + chunk] = (part @ head_bias).to(lm_head.dtype)
        return weight, bias

    def _jump_constants(self):
        """Everything a draft needs that depends only on the jump times: per
        jump the AdaLN modulations ``(1 + scale, shift, gate)`` of every norm;
        the embedded starting state's constant (the uniform posterior's
        embedding); and, for one jump, what its noise is drawn with: the
        Cholesky factor of ``E^T E``, or the folded path (``_folded_noise``)."""
        device = self.fc.weight.device
        per_jump = []
        for start, end in zip(self.grid[:-1], self.grid[1:]):
            # (scale, shift, gate) of every norm, from the reference model.
            layers, (scale, shift, _) = self.time_conditioning(start, end)
            modulations = [
                tuple(
                    (1 + scale[0, 0], shift[0, 0], gate[0, 0])
                    for scale, shift, gate in norms
                )
                for norms in layers
            ]
            per_jump.append((modulations, (1 + scale[0, 0], shift[0, 0])))
        embedding = self.embedding
        posterior = torch.full(
            (1, self.vocab_size), 1.0 / self.vocab_size, device=device
        ).to(embedding.dtype)
        uniform = self.state_norm(posterior @ embedding)[0]
        noise = None
        if self.num_steps == 1 and self.folded_noise:
            noise = self._folded_noise(uniform)
        elif self.num_steps == 1:
            gram = self._embedding_gram()
            noise = torch.linalg.cholesky(gram).T.to(embedding.dtype)
        return per_jump, uniform, noise

    def _embedding_gram(self) -> torch.Tensor:
        """``E^T E`` in float64 (``E`` the target's embedding), accumulated a
        chunk of rows at a time."""
        hidden = self.target_hidden_size
        gram = torch.zeros(
            hidden, hidden, dtype=torch.float64, device=self.embedding.device
        )
        for start in range(0, self.vocab_size, 8192):
            part = self.embedding[start : start + 8192].double()
            gram += part.T @ part
        return gram

    def noise_basis(self):
        """``(lam, Q)`` in float64 with ``E^T E = Q diag(lam) Q^T``."""
        lam, basis = torch.linalg.eigh(self._embedding_gram())
        return lam.clamp_min(0), basis

    def _folded_noise(self, uniform: torch.Tensor):
        """The first layer's input for one jump from pure noise ``x``: with
        ``x @ E = (z * sqrt(lam)) @ Q^T = y``, ``z ~ N(0, I_H)``,

            x_embed(uniform + gate * noise_norm(y))
              = x_embed(uniform) + gate * r * (z @ M),
            r = rsqrt(sum(lam z^2) / H + eps),  M = diag(sqrt(lam)) Q^T diag(w) W_x^T,

        ``w`` the noise norm's weight. Returns ``(lam / H, M, x_embed(uniform))``."""
        lam, basis = self.noise_basis()
        weight = self.noise_norm.weight.double()
        fold = (lam.sqrt()[:, None] * basis.T) * weight[None, :]
        proj = (fold @ self.x_embed.weight.double().T).to(self.embedding.dtype)
        variance = (lam / self.target_hidden_size).float()
        return variance, proj, self.x_embed(uniform)

    @staticmethod
    def _graph_max_bs() -> int:
        return _server_setting(
            lambda: get_exec().graph.cuda_graph_config.decode.max_bs, 64
        )

    def _alloc_tree_buffers(self, max_bs: int):
        """The tree a draft step hands to verification, in EAGLE's encoding
        (build_tree_kernel_efficient): per request its W - 1 non-root nodes'
        tokens and ids ``parent * W + index``, the parent list, and (trie nodes,
        distinct candidates); the draft's CUDA graphs write them in place."""
        width, device = self.tree_width, self._ctx_kv_weight.device

        def zeros(*shape):
            return torch.zeros(*shape, dtype=torch.int64, device=device)

        return (
            zeros(max_bs, width - 1),
            zeros(max_bs, width - 1),
            zeros(max_bs, width * (self.block_size - 1) + 1),
            zeros(max_bs, 2),
        )

    def tree_outputs(self, bsz: int):
        """The tree buffers for a batch of ``bsz``: the persistent ones, which
        captured graphs write, or (an eager batch above them) scratch ones."""
        buffers = self._tree_buffers
        if bsz > buffers[0].shape[0]:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"capturing a draft graph for {bsz} requests, above the tree buffers"
                )
            if self._tree_scratch is None or self._tree_scratch[0].shape[0] < bsz:
                self._tree_scratch = self._alloc_tree_buffers(bsz)
            buffers = self._tree_scratch
        return tuple(buffer[:bsz] for buffer in buffers)

    def tree_n_for(self, bsz: int) -> int:
        """Candidates per request in a batch of ``bsz`` (fixed per captured
        batch size, so graphs replay what they captured)."""
        return max(1, min(self.tree_max_n, self.tree_budget // max(bsz, 1)))

    # Rows of the head's logits held at once while taking the tree's argmaxes.
    _TREE_HEAD_CHUNK = 1024

    def _draft_tree(self, hidden: torch.Tensor, bsz: int, n_cand: int) -> None:
        """Greedy tokens of every candidate, merged into a trie cut to the
        verify width, written to the tree buffers (inside the draft graph)."""
        rows = hidden.shape[0]
        tokens = torch.empty(rows, dtype=torch.int64, device=hidden.device)
        for start in range(0, rows, self._TREE_HEAD_CHUNK):
            end = min(rows, start + self._TREE_HEAD_CHUNK)
            tokens[start:end] = self.head_logits(hidden[start:end]).argmax(dim=-1)
        if self.hot_ids is not None:
            tokens = self.hot_ids[tokens]
        out_tokens, out_ids, out_parents, out_stats = self.tree_outputs(bsz)
        node_tokens, ids, stats = self._build_tree(
            tokens.view(bsz, n_cand, self.block_size), self.tree_width
        )
        out_tokens.copy_(node_tokens)
        out_ids.copy_(ids)
        out_parents[:, 1 : ids.shape[1] + 1].copy_(ids)
        out_stats.copy_(stats)

    def noise_gate_at(self, start: float, end: float) -> torch.Tensor:
        """The weight of the state's noise term for a jump from ``start`` to
        ``end``: ``noise_gate * (end - start)``, or ``noise_gate`` alone for a
        ``"constant"`` gate (``modeling_flashmap``'s ``mix_state``)."""
        jump = 1.0 if self.config.noise_gate == "constant" else end - start
        return (self.noise_gate.float() * jump).to(self.noise_gate.dtype)

    # --- DSPARK protocol ------------------------------------------------------

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Its presence tells DSPARK the draft embeds its own block in-graph, so
        # it skips staging the target's embedding of the block every step; the
        # draft never reads it (the anchor is embedded by _anchor_kv).
        return F.embedding(input_ids, self.embedding)

    def prune_to_ctx_kv_injection(self) -> None:
        pass

    def compute_base_logits(self, hidden: torch.Tensor):
        return self.head_logits(hidden), None

    @torch.no_grad()
    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
    ) -> None:
        assert not target_hidden_is_projected
        if target_hidden.shape[0] == 0:
            return
        args = self._fused_write_args(pool)
        if cache_loc_2d is not None and commit_lens is not None:
            # Verify windows: only each request's committed columns are written.
            self._write_context(
                target_hidden,
                positions,
                cache_loc_2d.reshape(-1),
                args,
                commit_lens=commit_lens,
                locs_row_width=cache_loc_2d.shape[1],
            )
            return
        # A prefill, a chunk of rows at a time: every layer's keys and values of
        # one row take 6 * 2 * 2048 bf16 (48 KB for Qwen3-8B's map).
        chunk = self._WRITE_CHUNK
        for start in range(0, target_hidden.shape[0], chunk):
            self._write_context(
                target_hidden[start : start + chunk],
                positions[start : start + chunk],
                cache_loc[start : start + chunk],
                args,
            )

    # Prefill rows projected at once by write_target_hidden_kv.
    _WRITE_CHUNK = 2048

    def _write_context(self, target_hidden, positions, locs, args, **commit):
        """k-norm, RoPE and the write for every layer in one launch (sglang's
        fused DSpark kernel, which skips uncommitted verify columns)."""
        context = rms_norm(
            self._linear(
                target_hidden.to(self._ctx_kv_weight.dtype), "fc", self.fc.weight
            ),
            self._hidden_norm_weight,
            self.rms_norm_eps,
        )
        self._kv_write(
            self._linear(context, "ctx_kv", self._ctx_kv_weight),
            args,
            positions,
            locs,
            **commit,
        )

    def _kv_write(self, kv, args, positions, locs, **commit) -> None:
        """Every layer's context keys (k-normed, rotated) and values from the
        projections ``kv``, written at ``locs``: sglang's fused DSpark kernel for
        a bf16 pool, ours for an e4m3 one."""
        meta, k_norm_weights, cos_sin, kv_size = args
        write = kv_norm_rope_write_fp8 if self._kv_fp8 else fused_kv_norm_rope_write
        if not self._kv_fp8:
            commit["is_neox_style"] = False  # FlashMap rotates interleaved pairs
        write(
            kv,
            meta,
            k_norm_weights,
            cos_sin,
            positions,
            locs,
            self.num_stages,
            kv_size,
            self.head_dim,
            self.qk_norm_eps,
            **commit,
        )

    def _fused_write_args(self, pool):
        """``fused_kv_norm_rope_write``'s arguments for ``pool``."""
        if self._fused_write is not None and self._fused_write[0] == id(pool):
            return self._fused_write[1]
        buffers = [
            (pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id))
            for layer_id in range(self.num_stages)
        ]
        for buf in (buf for pair in buffers for buf in pair):
            if not buf.is_cuda or buf.dtype not in (
                torch.bfloat16,
                torch.float8_e4m3fn,
            ):
                raise ValueError(
                    f"the FlashMap draft KV pool must be bf16 or fp8_e4m3 on CUDA, got "
                    f"{buf.dtype} (--speculative-draft-kv-cache-dtype)"
                )
            assert buf.shape[1:] == (self.num_heads, self.head_dim), buf.shape
            assert buf.stride(1) == self.head_dim and buf.stride(2) == 1
        self._kv_fp8 = buffers[0][0].dtype == torch.float8_e4m3fn
        device = self._ctx_kv_weight.device
        meta = torch.tensor(
            [
                [k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0)]
                for k, v in buffers
            ],
            dtype=torch.int64,
            device=device,
        )
        k_norm_weights = torch.stack(
            [layer.cross_attn.k_norm.weight.float() for layer in self.layers]
        )
        # Row p: cos then sin of p times each frequency, for every position the
        # server can reach (the fused kernel does not bound-check): a request's
        # tokens fit its req_to_token row, and a draft reaches block_size past them.
        row = _server_setting(lambda: get_req_to_token_pool().req_to_token.shape[1], 0)
        context_length = _server_setting(lambda: get_model().context_length, 0)
        max_positions = (
            max(
                int(row),
                int(context_length),
                int(getattr(self.config, "max_position_embeddings", 0) or 0),
            )
            + self.block_size
            + 1
        )
        freqs = rope_freqs(
            torch.arange(max_positions, device=device), self.head_dim, self.rope_theta
        )
        cos_sin = torch.cat([freqs[..., 0], freqs[..., 1]], dim=-1).contiguous()
        args = (meta, k_norm_weights, cos_sin, self.num_heads * self.head_dim)
        self._fused_write = (id(pool), args)
        return args

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        """A block for every request; returns the DiT's final hidden states,
        which ``compute_base_logits`` (or the folded sampler) maps to logits.

        DSPARK asks for ``gamma`` drafts per request, the block size unless
        ``--speculative-dspark-block-size`` verifies fewer (at high concurrency
        the target's verify is compute-bound and the last drafts are rarely
        accepted); the map always drafts its whole block, as it was trained,
        and hands over the first ``gamma``."""
        del input_embeds, get_embedding, pp_proxy_tensors
        bsz, block_size = forward_batch.batch_size, self.block_size
        gamma = positions.numel() // bsz
        assert gamma <= block_size, (
            f"--speculative-dspark-block-size {gamma} exceeds the map's block of "
            f"{block_size} drafts"
        )
        pool = get_token_to_kv_pool()
        anchor_positions = positions.view(bsz, gamma)[:, 0]
        block_positions = anchor_positions[:, None] + torch.arange(
            block_size, device=positions.device
        )
        # The anchor's context entry goes into its own slot, the first of the
        # request's verify window, so each request attends to positions
        # 0..anchor; committing the verified tokens later overwrites the slot.
        self._kv_write(
            self._anchor_kv(input_ids.view(bsz, gamma)[:, 0]),
            self._fused_write_args(pool),
            anchor_positions,
            forward_batch.out_cache_loc.view(bsz, gamma)[:, 0],
        )
        page_table = get_req_to_token_pool().req_to_token[
            forward_batch.req_pool_indices
        ]
        cache_seqlens = (forward_batch.seq_lens + 1).to(torch.int32)
        # Tree drafting puts a request's candidate blocks side by side in the
        # cross-attention's query axis, so its context is read once, not N times.
        n_cand = self.tree_n_for(bsz) if self.tree_mode else 1
        schedule = _cross_attention_schedule(
            (bsz, n_cand * block_size, self.num_heads, self.head_dim),
            cache_seqlens,
            page_table,
            torch.float8_e4m3fn if self._kv_fp8 else torch.bfloat16,
        )
        hidden = self._draft(
            block_positions,
            page_table,
            cache_seqlens,
            schedule,
            [pool.get_key_buffer(i) for i in range(self.num_stages)],
            [pool.get_value_buffer(i) for i in range(self.num_stages)],
            n_cand,
        )
        if self.tree_mode:
            assert gamma == block_size, "tree drafting verifies the whole block"
            self._draft_tree(hidden, bsz, n_cand)
            # DSPARK's proposer expects one block per request: hand it the first
            # candidate's (the tree path reads the trie instead).
            hidden = hidden.view(bsz, n_cand, block_size, -1)[:, 0]
            hidden = hidden.reshape(bsz * block_size, -1)
        if gamma < block_size:
            hidden = hidden.view(bsz, block_size, -1)[:, :gamma].reshape(
                bsz * gamma, -1
            )
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    def _anchor_kv_impl(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        """Every layer's context key/value row for each anchor token, before
        k-norm and RoPE (the fused write applies those)."""
        anchor = F.embedding(anchor_ids, self.embedding)
        anchor = self._linear(
            anchor, "anchor_proj", self.anchor_proj.weight, self.anchor_proj.bias
        )
        return self._linear(self.anchor_norm(anchor), "ctx_kv", self._ctx_kv_weight)

    def _draft_impl(
        self,
        block_positions,
        page_table,
        cache_seqlens,
        schedule,
        keys,
        values,
        n_cand: int = 1,
    ) -> torch.Tensor:
        """The block's ``num_steps`` jumps, a pure tensor function so it can be
        compiled; returns the last jump's hidden states ``(bs * K, dim)``.

        ``n_cand`` > 1 (one jump only, tree drafting) drafts that many blocks
        per request from independent noise, candidate-major per request."""
        _, uniform, noise = self._constants
        requests, block_size = block_positions.shape
        bsz = requests * n_cand
        slots = bsz * block_size
        # The slots sit at anchor + 1 .. anchor + K.
        freqs = rope_freqs(block_positions + 1, self.head_dim, self.rope_theta)
        if n_cand > 1:
            freqs = freqs.repeat_interleave(n_cand, dim=0)
        attend = (freqs, page_table, cache_seqlens, schedule, keys, values, n_cand)
        if self.num_steps == 1 and not self.folded_noise:
            # One jump from t = 0: the noise drawn in the embedding's space.
            z = self.noise_temperature * torch.randn(
                bsz,
                block_size,
                noise.shape[0],
                device=block_positions.device,
                dtype=noise.dtype,
            )
            gate = self.noise_gate_at(self.grid[0], self.grid[1])
            embedded = uniform + gate * self.noise_norm(z @ noise)
            hidden = self._linear(
                embedded, "x_embed", self.x_embed.weight, self.x_embed.bias
            )
            return self._denoise(0, hidden, *attend).reshape(slots, -1)
        if self.num_steps == 1:
            # The same, embedded by the folded path (_folded_noise).
            variance, proj, embedded_uniform = noise
            z = self.noise_temperature * torch.randn(
                bsz,
                block_size,
                proj.shape[0],
                device=block_positions.device,
                dtype=proj.dtype,
            )
            mean_square = (z.float().square() * variance).sum(-1, keepdim=True)
            scale = self.noise_gate_at(self.grid[0], self.grid[1]) * torch.rsqrt(
                mean_square + self.rms_norm_eps
            )
            hidden = embedded_uniform + (scale * (z @ proj).float()).to(proj.dtype)
            return self._denoise(0, hidden, *attend).reshape(slots, -1)
        # Banked jumps (jump_bank.py): no vocabulary-sized state, the
        # predictions between jumps from the draft vocabulary's head.
        assert n_cand == 1, "several candidates per request take one jump"
        bank = self.jump_bank
        rows, seeds = bank.draw(slots, block_positions.device)
        embedded = bank.first_input(self, rows, uniform)
        supports = []
        for step in range(self.num_steps):
            if step:
                embedded = bank.jump_input(self, step, rows, seeds, supports)
            hidden = self._linear(
                embedded.view(bsz, block_size, -1),
                "x_embed",
                self.x_embed.weight,
                self.x_embed.bias,
            )
            hidden = self._denoise(step, hidden, *attend).reshape(slots, -1)
            if step + 1 < self.num_steps:
                logits = self.head_logits(hidden)
                supports.append(top_predictions(logits, self.jump_topk, self.hot_ids))
        return hidden

    def _denoise(
        self,
        step,
        hidden,
        freqs,
        page_table,
        cache_seqlens,
        schedule,
        keys,
        values,
        n_cand: int = 1,
    ) -> torch.Tensor:
        """One evaluation of the map at jump ``step``'s times on the embedded
        state after ``x_embed``, ``(bs, K, dim)``; returns the final AdaLN's
        output."""
        modulations, (final_scale, final_shift) = self._constants[0][step]
        bsz, block_size, _ = hidden.shape
        shape = (bsz, block_size, self.num_heads, self.head_dim)
        eps = self.layer_norm_eps
        for i, layer in enumerate(self.layers):
            (s_scale, s_shift, s_gate), (c_scale, c_shift, c_gate), mlp = modulations[i]
            m_scale, m_shift, m_gate = mlp
            # Self-attention within each request's block.
            normed = F.layer_norm(hidden, (self.dim,), eps=eps) * s_scale + s_shift
            qkv = self._linear(normed, f"qkv{i}", self._self_qkv_weights[i])
            attended = block_self_attention(
                qkv.view(bsz, block_size, 3, self.num_heads, self.head_dim),
                *self._self_norm_weights[i],
                freqs,
                self.qk_norm_eps,
            )
            attended = attended.reshape(bsz, block_size, self.dim)
            hidden = hidden + s_gate * self._linear(
                attended, f"self_o{i}", layer.self_attn.o_proj.weight
            )
            # Cross-attention onto the paged context.
            normed = F.layer_norm(hidden, (self.dim,), eps=eps) * c_scale + c_shift
            attn = layer.cross_attn
            query = rms_norm_rope(
                self._linear(normed, f"cross_q{i}", attn.q_proj.weight).view(shape),
                self._cross_q_norm_weights[i],
                freqs,
                self.qk_norm_eps,
            )
            cross_attention = (
                _paged_cross_attention_fp8 if self._kv_fp8 else _paged_cross_attention
            )
            # A request's candidates as one query sequence (their rows are
            # adjacent, candidate-major).
            context = cross_attention(
                query.reshape(bsz // n_cand, n_cand * block_size, *shape[2:]),
                keys[i],
                values[i],
                cache_seqlens,
                page_table,
                schedule,
            )
            context = context.reshape(bsz, block_size, self.dim)
            hidden = hidden + c_gate * self._linear(
                context, f"cross_o{i}", attn.o_proj.weight
            )
            normed = F.layer_norm(hidden, (self.dim,), eps=eps) * m_scale + m_shift
            mlp = layer.mlp
            normed = self._linear(normed, f"fc1_{i}", mlp.fc1.weight, mlp.fc1.bias)
            normed = F.gelu(normed)
            normed = self._linear(normed, f"fc2_{i}", mlp.fc2.weight, mlp.fc2.bias)
            hidden = hidden + m_gate * normed
        return F.layer_norm(hidden, (self.dim,), eps=eps) * final_scale + final_shift


EntryClass = [FlashMapDraftModel]


# --- DSPARK worker hooks, for FlashMap drafts only -----------------------------

_Worker = dspark_worker_v2.DSparkWorkerV2


def _no_attention_metadata(*args, **kwargs) -> None:
    return None


def _wrap_init_cuda_graphs(init_cuda_graphs):
    def wrapped(self, *args, **kwargs):
        if getattr(self.draft_model, "flashmap", False):
            # FlashMap attends through its own paged cross-attention and never
            # reads the draft attention backend: skip the metadata it prepares
            # every step.
            backend = self.draft_model_runner.attn_backend
            assert backend is not self.model_runner.attn_backend
            for name in (
                "init_forward_metadata",
                "init_forward_metadata_capture_cuda_graph",
                "init_forward_metadata_replay_cuda_graph",
            ):
                setattr(backend, name, _no_attention_metadata)
        return init_cuda_graphs(self, *args, **kwargs)

    return wrapped


def _wrap_build_draft_sampler(build_draft_sampler):
    def wrapped(self, *args, **kwargs):
        sampler = build_draft_sampler(self, *args, **kwargs)
        if sampler is None or not getattr(self.draft_model, "flashmap", False):
            return sampler
        max_bs = max(get_exec().graph.cuda_graph_config.decode.bs)
        sampler = FlashMapDraftSampler(sampler, max_bs=max_bs, device=self.device)
        logger.info(
            "FlashMap draft sampler: one pass over the block (folded sampling %s, "
            "fused kernel %s)",
            sampler.folded_sampling,
            sampler.fused,
        )
        return sampler

    return wrapped


# Installed once, however often the module is imported.
if not getattr(_Worker, "_flashmap_sglang_hooks", False):
    _Worker.init_cuda_graphs = _wrap_init_cuda_graphs(_Worker.init_cuda_graphs)
    _Worker._maybe_build_draft_sampler = _wrap_build_draft_sampler(
        _Worker._maybe_build_draft_sampler
    )
    _Worker._flashmap_sglang_hooks = True
