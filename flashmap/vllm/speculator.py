"""FlashMap speculator for vLLM's model runner V2.

FlashMap reuses the DFlash runtime unchanged: model loading
(``load_dflash_model``, which attaches the target's embedding and LM head), the
context K/V precompute, the fused input-preparation kernel, the attention
metadata and the FULL CUDA graph that covers the whole draft step. Its block has
DSpark's layout (``sample_from_anchor``): ``block_size`` query rows per request,
row 0 holding the anchor, every row a draft slot. What differs is the draft
model (model.py) and sampling, which reads logits off the folded head over the
draft vocabulary.

``num_speculative_tokens`` may be smaller than the block: the draft still
evaluates its whole block (the DiT's self-attention sees all of it) and proposes
the first ``num_speculative_tokens`` slots, so verification, which dominates at
high concurrency, checks fewer tokens.
"""

from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model

from flashmap.vllm.fold import split_argmax
from flashmap.vllm.kernels import consensus_path, scatter_rows

logger = init_logger(__name__)

__all__ = ["FlashMapSpeculator"]


class FlashMapSpeculator(DFlashSpeculator):
    _speculator_name = "FlashMap"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        block_size = int(self.draft_model_config.hf_config.block_size)
        # Drafts proposed for verification, a prefix of the block.
        self.num_verify = self.num_speculative_steps
        if self.num_verify > block_size:
            raise ValueError(
                f"num_speculative_tokens={self.num_verify} exceeds FlashMap's "
                f"block_size={block_size}"
            )
        # Every one of the block's rows is a draft slot; row 0 is the anchor.
        self.sample_from_anchor = True
        self.num_query_per_req = block_size
        if self.num_verify < block_size:
            # Per-slot buffers for the whole block (the DFlash ones are sized
            # for the verified drafts).
            self.num_speculative_steps = block_size
            num_slots = self.max_num_reqs * block_size
            self.draft_tokens = torch.zeros(
                self.max_num_reqs, block_size, dtype=torch.int64, device=device
            )
            self.sample_indices = torch.zeros(
                num_slots, dtype=torch.int64, device=device
            )
            self.sample_pos = torch.zeros(num_slots, dtype=torch.int64, device=device)
            self.sample_idx_mapping = torch.full(
                (num_slots,), -1, dtype=torch.int32, device=device
            )
            self.sample_col = torch.arange(
                block_size, dtype=torch.int32, device=device
            ).repeat(self.max_num_reqs)

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # A restricted draft vocabulary writes only its own columns of the
        # cached proposal distribution; every other token has probability 0.
        dtype, fill = super().draft_logits_spec(vllm_config)
        hf_config = self.draft_model_config.hf_config
        draft_vocab_size = getattr(hf_config, "draft_vocab_size", None)
        if draft_vocab_size and draft_vocab_size < hf_config.vocab_size:
            fill = float("-inf")
        return dtype, fill

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],  # noqa: ARG002
    ) -> nn.Module:
        model = load_dflash_model(target_model, self.vllm_config)
        model.prepare()
        # Key each query row's noise by its request's seed and its position.
        model.set_noise_keys(self.seeds, self.sample_idx_mapping)
        if model.max_candidates > 1 and self.draft_logits is not None:
            raise ValueError(
                "FLASHMAP_CANDIDATES needs greedy drafting (draft_sample_method "
                "'greedy', the default)."
            )
        if self.draft_watermarker is not None and model.hot_ids is not None:
            raise ValueError(
                "Watermarked drafting needs FlashMap's full draft vocabulary."
            )
        return model

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if self.acceptance_estimator is not None:
            # Acceptance is observed for the verified drafts only.
            self.acceptance_estimator = OnlineAcceptanceEstimator(
                self.max_num_reqs, self.num_verify, self.device
            )

    def set_attn(self, *args, **kwargs) -> None:
        super().set_attn(*args, **kwargs)
        target_attn_groups = kwargs.get("target_attn_groups", args[-1] if args else [])
        # The cross-attention schedules itself (model.py); vLLM's ahead-of-time
        # FlashAttention schedule for the draft group would be unused work, and
        # it mistakes an fp8 draft cache under a bf16 target for an unsupported
        # dtype.
        for groups in self.attn_groups:
            for group in groups:
                builder = group.get_metadata_builder()
                if getattr(builder, "aot_schedule", False):
                    builder.aot_schedule = False  # type: ignore[attr-defined]
        # The target runner builds metadata for the draft group too (cudagraph
        # capture), through its own builders.
        for gid in self.draft_kv_cache_group_ids:
            for group in target_attn_groups[gid]:
                builder = group.get_metadata_builder()
                if getattr(builder, "aot_schedule", False) and getattr(
                    builder, "kv_cache_dtype", None
                ) not in (torch.bfloat16, torch.float16):
                    builder.aot_schedule = False  # type: ignore[attr-defined]

    def _precompute_context_kv(
        self, start: int, end: int, dummy_run: bool = False
    ) -> None:
        # The context rows a draft step stores right before its forward are
        # projected by the forward's GEMM, together with the anchors.
        if start == 0 and not dummy_run and self._layer_group_idx is None:
            self.model.defer_context_kv(
                self.hidden_states[start:end],
                self.context_positions[start:end],
                self._context_slot_mappings[0][start:end],
            )
            return
        super()._precompute_context_kv(start, end, dummy_run)

    def capture(self) -> None:
        # Record the KV caches' layout (a host-to-device copy) before capturing.
        self.model.cache_layout()
        super().capture()

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if self.model.max_candidates <= 1:
            super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            return
        # Several candidate blocks per request: greedy tokens of each, then the
        # path most of them share.
        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        block = self.num_speculative_steps
        requests = num_tokens_padded // block
        tokens = split_argmax(self.model.compute_draft_logits(hidden))
        if self.model.hot_ids is not None:
            tokens = self.model.hot_ids[tokens]
        path = consensus_path(tokens.view(requests, -1, block))
        self.draft_tokens[:num_reqs] = path[:num_reqs]

    def propose(self, *args, **kwargs) -> torch.Tensor:
        return super().propose(*args, **kwargs)[:, : self.num_verify]

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        block, verify = self.num_speculative_steps, self.num_verify
        if verify == block:
            return self._sample(
                hidden_states,
                sample_src_positions,
                idx_mapping,
                temperature,
                seeds,
                draft_step,
                draft_logits,
            )
        # Sample the verified prefix of each block only; the rest stay zero.
        num_blocks = hidden_states.shape[0] // block

        def prefix(t: torch.Tensor) -> torch.Tensor:
            return t.view(num_blocks, block, *t.shape[1:])[:, :verify].flatten(0, 1)

        sampled = self._sample(
            prefix(hidden_states),
            prefix(sample_src_positions),
            prefix(idx_mapping),
            temperature,
            seeds,
            prefix(draft_step),
            draft_logits,
        )
        tokens = sampled.new_zeros(num_blocks, block)
        tokens[:, :verify] = sampled.view(num_blocks, verify)
        return tokens.view(-1)

    def _sample(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = self.model.compute_draft_logits(hidden_states)
        hot_ids = self.model.hot_ids
        if draft_logits is None:
            sampled = split_argmax(logits)
        elif hot_ids is None:
            sampled = gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                sample_src_positions,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
            )
            if self.draft_watermarker is not None:
                sampled = self.draft_watermarker.sample(
                    logits, sampled, idx_mapping, temperature
                )
        else:
            # Sample over the draft vocabulary and cache its logits for
            # probabilistic verification (the other columns stay -inf).
            sampled = gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                sample_src_positions,
                apply_temperature=True,
                is_drafting=True,
                use_fp64=self.use_fp64_gumbel,
            )
            scatter_rows(logits, idx_mapping, draft_step, hot_ids, draft_logits)
        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)
        return sampled if hot_ids is None else hot_ids[sampled]
