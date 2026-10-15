# coding=utf-8
"""Tree verification of multi-candidate FlashMap drafts inside sglang's DSPARK
worker.

Opt-in with ``FLASHMAP_TREE_N=<N>`` on a FlashMap server, with
``SGLANG_DSPARK_FOLDED_PROPOSAL=0`` (the tree path does its own proposal), a
target attention backend that honours a tree mask (``fa3``, ``flashinfer`` or
``triton``) and ``--page-size 1``. Greedy decoding only: the scheduler refuses
a request that samples (temperature > 0), asks for logprobs or constrains its
output (JSON schema, regex, EBNF, structural tag) when it arrives, with an error
response saying why (``tree_refusal``); the server keeps serving.

Each decode step, replacing DSPARK's chain step:

1. The draft (one CUDA-graph replay) drafts up to N blocks per request from
   independent noise in one forward, merges them into a trie and keeps the
   block_size nodes the most candidates pass through, in EAGLE's tree encoding
   (``tree.build_tree_abi``). With the root that is DSPARK's own
   verify width, so the target's verify graphs and KV reservation are unchanged.
2. ``build_tree_kernel_efficient`` turns it into the tree mask, positions and
   child/sibling links; the target verifies the tree in one forward over the
   request's window ``req_to_token[prefix : prefix + W]``.
3. ``verify_tree_greedy`` walks the tree along the target's argmax. Instead of
   moving the accepted nodes' KV, the window's ``req_to_token`` entries are
   reordered so the accepted path comes first (``tree_commit``, one launch);
   the path's hidden states become draft context there.

``FLASHMAP_SPLITKV_VERIFY=1`` (with ``--attention-backend triton``) runs the
target's verify attention with ``verify_attention.SplitKVVerify``: GQA
rows packed per KV head, the prefix split across programs, the tree mask only
on each request's W x W block.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

TREE_ENABLED = int(os.environ.get("FLASHMAP_TREE_N", "0")) >= 1
SPLITKV_ENABLED = os.environ.get("FLASHMAP_SPLITKV_VERIFY", "0") == "1"

if TREE_ENABLED or SPLITKV_ENABLED:
    from . import check_sglang_version

    check_sglang_version()

    from sglang.srt.layers.attention import flashattention_backend, triton_backend
    from sglang.srt.lora.layers import unwrap_lora_layer
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.speculative import dflash_utils
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_verify_logits_adjustments,
        resolve_dflash_verify_mask_policy,
    )
    from sglang.srt.speculative.dspark_components import dspark_worker_v2
    from sglang.srt.speculative.dspark_components.dspark_draft import (
        make_next_draft_input,
    )
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        alloc_verify_window,
    )
    from sglang.srt.speculative.eagle_utils import (
        TreeMaskMode,
        build_tree_kernel_efficient,
        verify_tree_greedy_func,
    )

    from .kernels import tree_commit
    from .verify_attention import SplitKVVerify


def _install_splitkv() -> None:
    """The split-KV verify on the target's Triton backend (sglang routes a
    verify through ``verify_splitkv_fwd`` when ``use_verify_splitkv`` is set,
    upstream on AMD only), installed before the target's graphs are captured."""
    triton_init = triton_backend.TritonAttnBackend.__init__

    def init(self, model_runner, *args, **kwargs):
        triton_init(self, model_runner, *args, **kwargs)
        if getattr(model_runner, "is_draft_worker", False) or self.use_mla:
            return
        self.use_verify_shared_kv = False
        self.use_verify_splitkv = True
        self._flashmap_splitkv = SplitKVVerify()
        # The tree mask as each request's W x W block (QLEN_ONLY).
        self._flashmap_splitkv.qlen_mask = TREE_ENABLED
        self.verify_splitkv_fwd = torch.compiler.disable(
            self._flashmap_splitkv.__call__
        )
        logger.info("FlashMap split-KV verify attention on the target's Triton backend")

    triton_backend.TritonAttnBackend.__init__ = init


_CONSTRAINTS = ("json_schema", "regex", "ebnf", "structural_tag")


def tree_refusal(req):
    """Why a tree-mode server refuses ``req`` (an sglang ``Req``), or None.

    The tree verify walks the target's argmax, so it serves greedy requests
    only; a request that samples, asks for logprobs or constrains its output is
    refused when it arrives rather than failing the batch it would join."""
    params = req.sampling_params
    asks = []
    if params.top_k > 1:  # sglang's greedy is top_k 1 (temperature 0)
        asks.append("sampling (temperature > 0)")
    if req.return_logprob:
        asks.append("logprobs")
    if any(getattr(params, name, None) for name in _CONSTRAINTS):
        asks.append("constrained decoding")
    if not asks:
        return None
    return (
        "this server drafts FlashMap trees (FLASHMAP_TREE_N), which serve greedy "
        f"requests (temperature 0) only; the request asks for {' and '.join(asks)}. "
        "Send it with temperature 0, or serve without FLASHMAP_TREE_N."
    )


def _install_tree() -> None:
    # Capture the target's verify graphs with a custom-mask buffer on the
    # backends that read one (DSPARK otherwise captures them mask-free).
    dflash_utils._DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = (
        dflash_utils._DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS
        - {"FlashInferAttnBackend", "TritonAttnBackend"}
    )
    fa3_init = flashattention_backend.FlashAttentionBackend.__init__

    def fa3_tree_init(self, model_runner, *args, **kwargs):
        """FA3 verifies a tree, as for EAGLE with top-k > 1, by cascade
        attention; every such branch keys on ``topk > 1``, which DSPARK pins to
        1, so any top-k above 1 selects it on the target."""
        fa3_init(self, model_runner, *args, **kwargs)
        if not model_runner.is_draft_worker and self.topk <= 1:
            self.topk = 2

    flashattention_backend.FlashAttentionBackend.__init__ = fa3_tree_init
    forward_decode = dspark_worker_v2.DSparkWorkerV2._forward_decode

    def _forward_decode(self, batch, on_publish, grammar_barrier=None):
        if not getattr(self.draft_model, "tree_mode", False) or (
            batch.forward_mode.is_idle()
        ):
            return forward_decode(self, batch, on_publish, grammar_barrier)
        return _forward_decode_tree(self, batch, on_publish)

    dspark_worker_v2.DSparkWorkerV2._forward_decode = _forward_decode

    # Refuse what the tree verify cannot serve when it arrives: every request
    # (new, or out of the grammar queue) enters the waiting queue here. An
    # aborted request is still run once (a one-token prefill, and with overlap
    # scheduling possibly one decode step) before its error is returned, so it
    # is also made greedy (aborting already drops its grammar and logprobs):
    # every batch it joins stays one the tree step serves.
    add_request_to_queue = Scheduler._add_request_to_queue

    def _add_request_to_queue(self, req, is_retracted: bool = False):
        if not is_retracted and not req.finished():
            refusal = tree_refusal(req)
            if refusal is not None:
                req.set_finish_with_abort(refusal)
                req.sampling_params.top_k = 1
        return add_request_to_queue(self, req, is_retracted)

    Scheduler._add_request_to_queue = _add_request_to_queue


def _tree_state(self):
    """Per-worker constants of the tree path, checked once."""
    state = getattr(self, "_flashmap_tree", None)
    if state is not None:
        return state
    backend, reads_mask = resolve_dflash_verify_mask_policy(
        self.model_runner.attn_backend
    )
    attn = self.model_runner.attn_backend
    while getattr(attn, "full_attn_backend", None) is not None:
        attn = attn.full_attn_backend
    if not (reads_mask or (backend == "FlashAttentionBackend" and attn.topk > 1)):
        raise RuntimeError(
            "FlashMap tree verify needs a target attention backend that reads a "
            f"tree mask (fa3, flashinfer or triton), got {backend}"
        )
    if self._draft_sampler is not None:
        raise RuntimeError(
            "FlashMap tree verify does its own proposal: set "
            "SGLANG_DSPARK_FOLDED_PROPOSAL=0"
        )
    width = int(self.draft_model.tree_width)
    assert width == self.verify_num_draft_tokens == self.gamma + 1, (
        width,
        self.verify_num_draft_tokens,
    )
    # Reordering a window's req_to_token entries keeps each slot's KV where it
    # is only if every token is its own page.
    if self.model_runner.page_size != 1:
        raise RuntimeError("FlashMap tree verify needs --page-size 1")
    splitkv = getattr(attn, "_flashmap_splitkv", None)
    state = {
        "width": width,
        "mask_mode": (
            TreeMaskMode.QLEN_ONLY
            if getattr(splitkv, "qlen_mask", False)
            else TreeMaskMode.FULL_MASK
        ),
        "offsets": torch.arange(width, device=self.device, dtype=torch.int64),
        "req_to_token": self.model_runner.req_to_token_pool.req_to_token,
    }
    self._flashmap_tree = state
    logger.info(
        "FlashMap tree verify: up to %d candidates per request (budget %d per "
        "step), %d verify tokens, target backend %s",
        self.draft_model.tree_max_n,
        self.draft_model.tree_budget,
        width,
        backend,
    )
    return state


def _forward_decode_tree(self, batch, on_publish):
    if batch.spec_info is None:
        batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
    draft_input = batch.spec_info
    sampling_info = batch.sampling_info
    if (
        batch.has_grammar
        or batch.return_logprob
        or not (sampling_info is None or sampling_info.is_all_greedy)
    ):
        # The scheduler refuses such requests on arrival (tree_refusal), so
        # this is unreachable. Falling back to DSPARK's chain step is not safe:
        # tried with greedy logprob requests, it broke their outputs on a
        # tree-mode server.
        raise NotImplementedError(
            "FlashMap tree verify (FLASHMAP_TREE_N) serves greedy requests "
            "(temperature 0) without grammar or logprobs only"
        )
    state = _tree_state(self)
    width, depth = state["width"], self.gamma

    batch.seq_lens.record_stream(torch.get_device_module(self.device).current_stream())
    bs, device = len(batch.seq_lens), self.device
    prefix_lens = batch.seq_lens
    target_model = self.target_worker.model_runner.model
    seq_lens_cpu = (
        batch.seq_lens_cpu
        if batch.seq_lens_cpu is not None
        else draft_input.nxt_kv_lens_cpu
    )
    window = alloc_verify_window(
        batch=batch,
        bs=bs,
        device=device,
        verify_num_draft_tokens=width,
        block_pos_offsets=state["offsets"],
        model_runner=self.model_runner,
    )

    # 1. Draft: one graph replay fills the draft's tree buffers.
    with self._draft_context():
        self._proposer._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=window,
            bs=bs,
            device=device,
            embed_module=unwrap_lora_layer(target_model.get_input_embeddings()),
            draft_sampler=None,
            sampling_info=sampling_info,
        )
    node_tokens, node_ids, parent_list, _ = self.draft_model.tree_outputs(bs)

    # 2. The tree's mask, positions and links; verify it.
    (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        verify_ids,
    ) = build_tree_kernel_efficient(
        draft_input.bonus_tokens.view(-1),
        parent_list,
        node_ids,
        node_tokens,
        prefix_lens,
        int(seq_lens_cpu.sum()) + bs * width,
        topk=width,
        spec_steps=depth,
        num_verify_tokens=width,
        tree_mask_mode=state["mask_mode"],
    )
    verify_input = DFlashVerifyInput(
        draft_token=verify_ids,
        positions=positions,
        draft_token_num=width,
        custom_mask=tree_mask,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        live_seq_lens_cpu=batch.seq_lens_cpu,
    )
    batch.out_cache_loc = window.verify_cache_loc
    executor = self._verify_executor
    seq_lens_cpu_backup, seq_lens_sum_backup = batch.seq_lens_cpu, batch.seq_lens_sum
    if not executor._verify_backend_self_adds_seq_lens():
        if seq_lens_cpu_backup is not None:
            batch.seq_lens_cpu = seq_lens_cpu_backup + width
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
    verified = executor._forward_prepared_verify(
        batch=batch,
        verify_input=verify_input,
        seq_lens_cpu_backup=seq_lens_cpu_backup,
        seq_lens_sum_backup=seq_lens_sum_backup,
    )
    logits_output = verified.logits_output
    logits = logits_output.next_token_logits
    if sampling_info is not None:
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits, sampling_info=sampling_info, draft_token_num=width
        )

    # 3. Accept the longest path the target agrees with.
    predict = torch.zeros(bs * width, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, depth + 1), -1, dtype=torch.int32, device=device)
    num_correct = torch.empty((bs,), dtype=torch.int32, device=device)
    verify_tree_greedy_func(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=num_correct,
        candidates=verify_ids.view(bs, width),
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        target_predict=logits.argmax(dim=-1).view(bs, width),
    )
    # 4. Commit without moving KV: reorder each request's window so the
    # accepted path (root first) is its first commit_lens slots, where its KV
    # already is; the path's hidden states become draft context there (rows off
    # the path go to the pool's dummy slot 0).
    out_tokens, bonus, commit_lens, new_seq_lens, inject_locs = tree_commit(
        accept_index,
        num_correct,
        window.verify_cache_loc_2d,
        predict,
        batch.req_pool_indices,
        prefix_lens,
        state["req_to_token"],
    )
    # sglang's scheduler assumes a forward only reads req_to_token: its
    # write-after-read barrier must wait for this write too.
    reordered = torch.get_device_module(self.device).Event()
    reordered.record()
    self.model_runner.shared_read_done_event = reordered
    if on_publish is not None:
        on_publish(new_seq_lens)
    executor.kv_injector.inject_target_hidden(
        target_hidden=logits_output.hidden_states,
        cache_loc=inject_locs,
        positions=positions,
    )
    logits_output.hidden_states = None
    return GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=out_tokens.reshape(-1),
        accept_lens=commit_lens,
        block_accept_lens=commit_lens,
        cap_lens=None,
        can_run_cuda_graph=verified.can_run_cuda_graph,
        next_draft_input=make_next_draft_input(
            bonus_tokens=bonus, new_seq_lens=new_seq_lens
        ),
        speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
        new_seq_lens=new_seq_lens,
    )


# Installed once, however often the module is imported.
if SPLITKV_ENABLED and not getattr(
    triton_backend, "_flashmap_sglang_splitkv", False
):
    _install_splitkv()
    triton_backend._flashmap_sglang_splitkv = True
if TREE_ENABLED and not getattr(dspark_worker_v2, "_flashmap_sglang_tree", False):
    _install_tree()
    dspark_worker_v2._flashmap_sglang_tree = True
