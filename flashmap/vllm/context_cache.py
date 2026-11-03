"""Where the drafter's cross-attention context lives in vLLM's paged KV cache.

The 0.2.0 layout gives each DiT layer one attention layer in vLLM, of the DiT's
heads. On a hybrid target (linear attention or Mamba layers next to full
attention, as in Qwen3.8) that layout is costly: vLLM gives every layer one page
size, the largest any layer needs, and puts the same number of layers in every
cache group, the fewest of any layer type. A FlashMap DiT's page per token is
larger than the target attention's (Qwen3.8-27B: 20 heads of 128 against 4 of
256, 2.5x), so every linear-attention state and every target attention page was
padded to it; and its few layers (6) set the group size, which split the target's
16 attention layers into 3 groups with padding layers.

The virtual layout registers the context instead as as many attention layers as
the target has full-attention layers, each with the target's page per token (8
heads of 128 for Qwen3.8), and packs the DiT's heads into them in order. Every
page is then the target's own, and the drafter's layers fill a cache group of the
target's group size exactly. A DiT layer's heads may span two or three of them:
the writer and the cross-attention read each head where the plan puts it.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["ContextCachePlan", "attend_by_segments", "plan_context_cache"]

# FLASHMAP_CONTEXT_CACHE: "auto" (virtual on hybrid targets), "layers" or "virtual".
MODES = ("auto", "layers", "virtual")


@dataclass(frozen=True)
class ContextCachePlan:
    """``num_caches`` attention layers of ``heads_per_cache`` heads hold the context
    of ``num_layers`` DiT layers of ``heads`` heads. DiT head ``h`` of layer ``l``
    is head ``slot[l][h]`` of cache ``cache[l][h]``; ``segments[l]`` lists layer
    ``l``'s heads as runs ``(cache, first slot, count, first head)``."""

    num_layers: int
    heads: int
    num_caches: int
    heads_per_cache: int
    virtual: bool
    reason: str

    @property
    def cache(self) -> list[list[int]]:
        return [[self._global(l, h) // self.heads_per_cache for h in range(self.heads)] for l in range(self.num_layers)]

    @property
    def slot(self) -> list[list[int]]:
        return [[self._global(l, h) % self.heads_per_cache for h in range(self.heads)] for l in range(self.num_layers)]

    @property
    def segments(self) -> list[list[tuple[int, int, int, int]]]:
        runs = []
        for layer in range(self.num_layers):
            layer_runs: list[tuple[int, int, int, int]] = []
            for head in range(self.heads):
                cache, slot = divmod(self._global(layer, head), self.heads_per_cache)
                if layer_runs and layer_runs[-1][0] == cache:
                    c, first_slot, count, first_head = layer_runs[-1]
                    layer_runs[-1] = (c, first_slot, count + 1, first_head)
                else:
                    layer_runs.append((cache, slot, 1, head))
            runs.append(layer_runs)
        return runs

    def _global(self, layer: int, head: int) -> int:
        """Head ``head`` of layer ``layer`` in the packed order, layer major (the
        0.2.0 layout is the case ``heads_per_cache == heads``)."""
        return layer * self.heads + head


def plan_context_cache(
    num_layers: int,
    heads: int,
    head_dim: int,
    *,
    hybrid: bool,
    target_attention_layers: int,
    target_kv_heads: int,
    target_head_size: int,
    bf16_cache: bool,
    mode: str | None = None,
) -> ContextCachePlan:
    """The layout for a DiT of ``num_layers`` layers of ``heads`` heads of
    ``head_dim``, next to a target with ``target_attention_layers`` full-attention
    layers of ``target_kv_heads`` heads of ``target_head_size``. ``mode`` (default:
    ``FLASHMAP_CONTEXT_CACHE`` or "auto") forces either layout; "virtual" falls back
    to "layers" where the target's shapes do not allow it."""
    mode = mode or os.environ.get("FLASHMAP_CONTEXT_CACHE", "auto")
    if mode not in MODES:
        raise ValueError(f"FLASHMAP_CONTEXT_CACHE must be one of {MODES}, got {mode!r}")
    def legacy(reason: str) -> ContextCachePlan:
        return ContextCachePlan(num_layers, heads, num_layers, heads, False, reason)

    if mode == "layers":
        return legacy("FLASHMAP_CONTEXT_CACHE=layers")
    if mode == "auto" and not hybrid:
        return legacy("a target without linear-attention or Mamba layers")
    if not bf16_cache:
        return legacy("an fp8 KV cache")
    elements = target_kv_heads * target_head_size
    if target_attention_layers <= 0 or elements <= 0 or elements % head_dim:
        return legacy(f"target attention of {target_kv_heads} heads of {target_head_size} does not hold heads of {head_dim}")
    heads_per_cache = elements // head_dim
    if target_attention_layers * heads_per_cache < num_layers * heads:
        return legacy(
            f"{target_attention_layers} target attention layers of {heads_per_cache} heads cannot hold "
            f"{num_layers} x {heads} heads"
        )
    return ContextCachePlan(
        num_layers, heads, target_attention_layers, heads_per_cache, True,
        f"{target_attention_layers} caches of {heads_per_cache} heads of {head_dim}, the target attention's page",
    )


def attend_by_segments(
    query: torch.Tensor,
    caches: list[torch.Tensor],
    segments: list[tuple[int, int, int, int]],
    attend: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """One DiT layer's cross-attention of ``query (tokens, heads, head_dim)`` over
    its context in ``caches`` (vLLM's ``(blocks, heads, block_size, 2 * head_dim)``):
    ``attend(query heads, cache heads, count)`` for each run ``segments`` lists, on
    views of the run's heads, the outputs concatenated along the heads. A run that
    covers the whole query and its whole cache (the per-layer layout) passes both
    as they are."""
    heads = query.shape[1]
    parts = []
    for cache_id, first_slot, count, first_head in segments:
        cache = caches[cache_id]
        part_query = query if count == heads else query[:, first_head : first_head + count].contiguous()
        part_cache = cache if count == cache.shape[1] else cache[:, first_slot : first_slot + count]
        parts.append(attend(part_query, part_cache, count))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
