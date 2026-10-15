# coding=utf-8
"""DSPARK's Markov head and draft sampler, specialized for FlashMap.

sglang's DSPARK worker requires a Markov head (a low-rank bias on each draft
position's logits from the previous token) and samples the block one position
at a time through it. FlashMap has no such head: its block positions are
independent given the noise. :class:`IdentityMarkov` stands in for the head
(rank 1, never loaded, contributes nothing) so DSPARK samples the map's logits
unchanged and verifies them with its usual rejection sampling.

With no bias, the positions are independent, yet DSPARK's graph-folded sampler
would still walk them one at a time (an exponential-noise fill and two
online-softmax kernels per position, or an argmax per position for greedy
batches). :class:`FlashMapDraftSampler` samples the whole block at once inside
the draft's CUDA graph: the logits written into DSPARK's buffer by one GEMM
against the folded head, then one ``SampleStepTokens`` pass with fresh
exponential noise per row and each request's temperature (exactly DSPARK's
per-position sampling, since the rows are independent; greedy rows take their
argmax). Drafted indices of a restricted vocabulary map back to token ids in
the graph too, and the logits rejection sampling reads stay over the full
vocabulary (-inf outside the draft vocabulary).

``FLASHMAP_FUSED_SAMPLER=1`` (opt-in, measured on Qwen3-8B only) samples with
one kernel instead: each greedy row's
argmax, each other row Gumbel-max-sampled at its temperature with in-kernel
Philox noise, so there is no noise buffer, and the full-vocabulary logits are
scattered only when DSPARK reads them for a sampling batch, so greedy batches
never pay for them.
"""

from __future__ import annotations

import os

import torch
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import SampleStepTokens
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

from .kernels import sample_rows


class IdentityMarkov(VanillaMarkov):
    """A Markov head that leaves the draft's logits unchanged.

    With a restricted draft vocabulary (``hot_ids``, set by the draft) its
    eager block sampler also maps drafted indices back to token ids and
    scatters the logits back over the full vocabulary (-inf elsewhere, so
    rejection sampling stays exact)."""

    def __init__(self, *, vocab_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=1)
        for weight in self.parameters():
            weight.requires_grad_(False)
            torch.nn.init.zeros_(weight)
        self.hot_ids = None

    def apply_step_logits(self, logits, *args, **kwargs):
        return logits

    def apply_block_logits(self, base_logits, *args, **kwargs):
        return base_logits

    def sample_block_greedy_fused(self, *args, **kwargs):
        return None  # the caller falls back to sample_block

    def sample_block(
        self,
        base_logits,
        *,
        first_prev_tokens,
        hidden_states,
        sampler,
        collect_corrected: bool = True,
    ):
        del first_prev_tokens, hidden_states
        tokens = torch.stack(
            [sampler(base_logits[:, i], i) for i in range(base_logits.shape[1])], dim=1
        )
        if self.hot_ids is None:
            return tokens, base_logits if collect_corrected else None
        logits = None
        if collect_corrected:
            logits = base_logits.new_full(
                (*base_logits.shape[:2], self.vocab_size), float("-inf")
            )
            logits.index_copy_(2, self.hot_ids, base_logits)
        return self.hot_ids[tokens], logits


class _FullLogits:
    """DSPARK's ``corrected_out`` for a restricted draft vocabulary: slicing
    the first ``rows`` scatters those rows' draft-vocabulary logits into the
    full-vocabulary buffer (-inf elsewhere) and returns them."""

    def __init__(self, full: torch.Tensor, hot: torch.Tensor, hot_ids: torch.Tensor):
        self.full, self.hot, self.hot_ids = full, hot, hot_ids
        self.full.fill_(float("-inf"))

    def __getitem__(self, index: slice) -> torch.Tensor:
        assert index.start is None and index.step is None, index
        rows = index.stop
        return self.full[:rows].index_copy_(1, self.hot_ids, self.hot[:rows])


class FlashMapDraftSampler(DsparkDraftSampler):
    """DSPARK's graph-folded draft sampler for a FlashMap draft: the whole block
    in one pass (see the module docstring). Same contract."""

    # DsparkDraftSampler state this sampler takes over (sglang 0.5.20).
    _INHERITED = (
        "model", "gamma", "sample_from_anchor", "query_token_num", "out",
        "confidence_out", "folded_sampling", "temperatures", "greedy_mask",
        "exp_noise", "corrected_out", "_tp_sync",
    )  # fmt: skip

    def __init__(self, sampler: DsparkDraftSampler, *, max_bs: int, device):
        missing = [name for name in self._INHERITED if not hasattr(sampler, name)]
        assert not missing, f"DsparkDraftSampler lacks {missing}: unsupported sglang"
        self.__dict__.update(sampler.__dict__)
        assert self.sample_from_anchor and self.confidence_out is None
        self.fused = os.environ.get("FLASHMAP_FUSED_SAMPLER", "0") == "1"
        self.hot_logits = None
        if not self.folded_sampling:
            return
        hot_ids = self.model.hot_ids
        rows, draft_vocab = int(max_bs) * self.gamma, self.model.folded_head[0].shape[0]
        self.exp_noise = None
        if not self.fused:
            # Exponential noise for every row of the block, not one position's.
            self.row_noise = torch.empty(
                (rows, draft_vocab), dtype=torch.float32, device=device
            )
            if hot_ids is not None:
                self.corrected_out.fill_(float("-inf"))
            return
        self.seed = torch.zeros((), dtype=torch.int64, device=device)
        if hot_ids is not None:
            # The draft-vocabulary logits outlive the graph replay until DSPARK
            # reads them (``_FullLogits``).
            self.hot_logits = torch.empty(
                (rows, draft_vocab), dtype=self.corrected_out.dtype, device=device
            )
            self.corrected_out = _FullLogits(
                self.corrected_out, self.hot_logits, hot_ids
            )

    def __call__(self, hidden_states, input_ids):
        del input_ids
        bs = hidden_states.shape[0] // self.query_token_num
        rows = bs * self.gamma
        hot_ids = self.model.hot_ids
        if not self.folded_sampling:
            logits = self.model.head_logits(hidden_states)
            tokens = self._tp_sync.sync(
                SpecTpSyncSite.DSPARK_GRAPH_GREEDY, logits.argmax(dim=-1)
            )
        else:
            temperatures = self.temperatures[:bs].repeat_interleave(self.gamma)
            greedy = self.greedy_mask[:bs].repeat_interleave(self.gamma)
            if self.fused:
                out = self.hot_logits if hot_ids is not None else self.corrected_out
                logits = out[:rows]
                logits.copy_(self.model.head_logits(hidden_states))
                self.seed.add_(1)
                tokens = sample_rows(logits, temperatures, greedy, self.seed)
            else:
                logits = self.model.head_logits(hidden_states)
                if hot_ids is None:
                    self.corrected_out[:rows].copy_(logits)
                else:
                    self.corrected_out[:rows].index_copy_(1, hot_ids, logits)
                tokens = SampleStepTokens.execute(
                    step_logits=logits,
                    temperatures=temperatures,
                    greedy_mask=greedy,
                    exp_noise=self.row_noise[:rows].exponential_(),
                )
            tokens = self._tp_sync.sync(SpecTpSyncSite.DSPARK_GRAPH_SAMPLE, tokens)
        if hot_ids is not None:
            tokens = hot_ids[tokens]
        self.out[:rows].copy_(tokens.reshape(-1))


__all__ = ["FlashMapDraftSampler", "IdentityMarkov"]
