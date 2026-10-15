"""Check the sglang FlashMap draft (flashmap.sglang) against the reference model.

Builds ``flashmap.sglang.model.FlashMapDraftModel`` from a release folder with the
target's embedding and LM head, and drives it as sglang's DSPARK worker does, with a stand-in KV pool and forward
batch: two requests' context is written with ``write_target_hidden_kv`` (one
prefill; one prefill plus a verify window whose uncommitted columns carry
garbage), then both are drafted in one ``forward``. The drafted logits must
match ``flashmap.FlashMapDraftModel`` (its context cache and forward: every
attention over the whole context, no folding, no fused kernels) on the same noise:

* eager (``FLASHMAP_COMPILE=0``), the served path's own noise is replayed into
  the reference;
* compiled, the drafts are compared noiseless (torch.compile draws its noise
  differently), which still runs every layer;
* the eigendecomposition the noise is drawn with must reproduce ``E^T E``.

Needs a GPU and flashmap[sglang]:

    python scripts/check_sglang.py --draft FlashMap-Qwen3-8B --target Qwen/Qwen3-8B
"""

import argparse
import json
import math
import os
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import AutoConfig

import flashmap  # noqa: F401  (registers the flashmap config)
from flashmap import FlashMapCache
from flashmap import FlashMapDraftModel as Reference

DEVICE = torch.device("cuda")


class FakePool:
    """The slice of sglang's MHA KV pool the draft uses."""

    def __init__(self, *, layers, slots, heads, head_dim, dtype=torch.bfloat16):
        shape = (slots, heads, head_dim)
        self.keys = [
            torch.zeros(shape, dtype=dtype, device=DEVICE) for _ in range(layers)
        ]
        self.values = [
            torch.zeros(shape, dtype=dtype, device=DEVICE) for _ in range(layers)
        ]

    def get_key_buffer(self, layer_id):
        return self.keys[layer_id]

    def get_value_buffer(self, layer_id):
        return self.values[layer_id]


def target_vocab_weights(target: str, device: str = "cuda"):
    """The target's (embed_tokens, lm_head) weights from its safetensors. A
    target that ties them (``tie_word_embeddings``, e.g. Qwen3-4B) may store
    no ``lm_head.weight``: its embedding is the LM head."""
    if not os.path.isdir(target):
        from huggingface_hub import snapshot_download

        target = snapshot_download(target, allow_patterns=["*.json", "*.safetensors"])
    with open(os.path.join(target, "config.json")) as handle:
        config = json.load(handle)
    tied = bool(config.get("tie_word_embeddings", False))
    index_path = os.path.join(target, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as handle:
            files = json.load(handle)["weight_map"]
    else:  # a single-file checkpoint
        with safe_open(os.path.join(target, "model.safetensors"), "pt") as handle:
            files = {name: "model.safetensors" for name in handle.keys()}
    embed = "model.embed_tokens.weight"
    head = embed if tied or "lm_head.weight" not in files else "lm_head.weight"
    out = []
    for name in (embed, head):
        with safe_open(os.path.join(target, files[name]), "pt", device=device) as f:
            out.append(f.get_tensor(name))
    return out


@torch.no_grad()
def reference_logits(draft, embed, lm_head, hidden, anchor, noise_embed, start, end):
    """The map's logits for one block after ``hidden (T, features)`` and the
    anchor token at position T, from the embedded noise ``noise_embed (K, H)``
    (= x @ E), for a jump from ``start`` to ``end``, by the reference model
    (``flashmap.FlashMapDraftModel``, whose weights the served draft shares)."""
    length, block = hidden.shape[0], draft.block_size
    cache = FlashMapCache(draft.config, 1, length + 1, device=DEVICE, dtype=torch.bfloat16)
    Reference.extend_context(draft, cache, hidden[None])
    # The anchor's embedding at position T, as Reference.draft writes it.
    anchor_context = draft.anchor_norm(draft.anchor_proj(embed[anchor].view(1, 1, -1)))
    rope = flashmap.modeling_flashmap.rope_tables(
        torch.full((1, 1), length, device=DEVICE), draft.head_dim, draft.config.rope_theta
    )
    for i, layer in enumerate(draft.layers):
        cache.write(i, length, *layer.cross_attn.project_kv(anchor_context, rope))
    vocab = embed.shape[0]
    uniform = torch.full((1, vocab), 1.0 / vocab, device=DEVICE).to(embed.dtype) @ embed
    state = Reference.mix_state(draft, uniform, noise_embed[None], start, end)
    positions = length + 1 + torch.arange(block, device=DEVICE)[None]
    out = Reference.forward(draft, state, start, end, positions, cache.context(length + 1))
    return out[0] @ lm_head.T


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--draft", required=True, help="FlashMap release folder")
    parser.add_argument("--target", required=True, help="target model path or hub id")
    parser.add_argument("--context-lens", default="150,220")
    parser.add_argument(
        "--kv-dtype",
        choices=("bf16", "fp8"),
        default="bf16",
        help="draft KV pool dtype (fp8: e4m3, as --speculative-draft-kv-cache-dtype)",
    )
    args = parser.parse_args()
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    from flashmap.sglang.model import FlashMapDraftModel

    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(args.draft)
    embed, lm_head = target_vocab_weights(args.target)
    weights = {}
    for name in ("model.safetensors", "draft_vocab.safetensors"):  # as sglang reads them
        if os.path.exists(os.path.join(args.draft, name)):
            with safe_open(os.path.join(args.draft, name), "pt") as handle:
                weights.update({key: handle.get_tensor(key) for key in handle.keys()})
    lens = [int(x) for x in args.context_lens.split(",")]
    features = len(config.target_layer_ids) * int(config.target_hidden_size)
    hiddens = [
        torch.randn(n + 8, features, device=DEVICE, dtype=torch.bfloat16) for n in lens
    ]

    for compiled in (False, True):
        os.environ["FLASHMAP_COMPILE"] = "1" if compiled else "0"
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(DEVICE):
            draft = FlashMapDraftModel(config)
        torch.set_default_dtype(torch.float32)
        draft.load_weights(iter(weights.items()))
        draft.attach_shared_modules(
            embed_tokens=SimpleNamespace(weight=embed),
            lm_head=SimpleNamespace(weight=lm_head),
        )
        noise = draft._constants[2]
        if not compiled:
            gram = embed.double().T @ embed.double()
            if draft.folded_noise:
                lam, basis = draft.noise_basis()
                err = ((basis * lam) @ basis.T - gram).norm() / gram.norm()
            else:
                factor = noise.double()
                err = (factor.T @ factor - gram).norm() / gram.norm()
            print(f"noise factor: reproduces E^T E to {err:.2e}")
            assert err < 1e-2, err
        block = draft.block_size
        pool = FakePool(
            layers=draft.num_stages,
            slots=4096,
            heads=draft.num_heads,
            head_dim=draft.head_dim,
            dtype=torch.float8_e4m3fn if args.kv_dtype == "fp8" else torch.bfloat16,
        )
        slots = 1 + torch.randperm(4095, device=DEVICE)  # slot 0 is sglang's dummy slot
        req_to_token = slots[: len(lens) * 1024].view(len(lens), 1024).to(torch.int32)
        for req, (hidden, length) in enumerate(zip(hiddens, lens)):
            locs = req_to_token[req].long()
            if req == 0:  # one prefill
                draft.write_target_hidden_kv(
                    target_hidden=hidden[:length],
                    pool=pool,
                    positions=torch.arange(length, device=DEVICE),
                    cache_loc=locs[:length],
                )
                continue
            # a prefill, then a verify window with 5 of 8 columns committed
            split, width, commit = length - 5, 8, 5
            draft.write_target_hidden_kv(
                target_hidden=hidden[:split],
                pool=pool,
                positions=torch.arange(split, device=DEVICE),
                cache_loc=locs[:split],
            )
            window = torch.cat(
                [hidden[split:length], 100 * torch.randn_like(hidden[: width - commit])]
            )
            draft.write_target_hidden_kv(
                target_hidden=window,
                pool=pool,
                positions=torch.arange(split, split + width, device=DEVICE),
                cache_loc=locs[split : split + width],
                cache_loc_2d=locs[split : split + width].view(1, -1),
                commit_lens=torch.tensor([commit], device=DEVICE),
            )
            assert all(
                not pool.keys[i][locs[length : split + width]].float().any()
                for i in range(draft.num_stages)
            ), "uncommitted verify columns were written"

        anchors = torch.randint(1000, 50000, (len(lens),), device=DEVICE)
        seq_lens = torch.tensor(lens, device=DEVICE)
        input_ids = torch.full((len(lens), block), 151669, device=DEVICE)
        input_ids[:, 0] = anchors
        positions = (seq_lens[:, None] + torch.arange(block, device=DEVICE)).flatten()
        out_cache_loc = torch.stack(
            [req_to_token[r, lens[r] : lens[r] + block] for r in range(len(lens))]
        ).long()
        batch = SimpleNamespace(
            batch_size=len(lens),
            seq_lens=seq_lens,
            req_pool_indices=torch.arange(len(lens), device=DEVICE),
            out_cache_loc=out_cache_loc.flatten(),
        )
        context = ForwardContext(
            attn_backend=SimpleNamespace(
                token_to_kv_pool=pool,
                req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            )
        )
        draft.noise_temperature = 0.0 if compiled else 1.0
        torch.manual_seed(1)
        with forward_context(context), torch.no_grad():
            hidden = draft.forward(input_ids.flatten(), positions, batch).hidden_states
            served = draft.compute_base_logits(hidden)[0].view(len(lens), block, -1)
        torch.manual_seed(1)  # replay the served path's noise draw
        z = draft.noise_temperature * torch.randn(
            len(lens), block, embed.shape[1], device=DEVICE, dtype=torch.bfloat16
        )
        # The noise's embedding x @ E, as the served path draws it.
        if draft.folded_noise:
            lam, basis = draft.noise_basis()
            noise_embed = ((z.double() * lam.sqrt()) @ basis.T).to(torch.bfloat16)
        else:
            noise_embed = z @ noise
        start, end = draft.grid[0], draft.grid[1]
        hot = (
            draft.hot_ids
            if draft.hot_ids is not None
            else torch.arange(embed.shape[0], device=DEVICE)
        )
        for req, length in enumerate(lens):
            ref = reference_logits(
                draft,
                embed,
                lm_head,
                hiddens[req][:length],
                anchors[req],
                noise_embed[req],
                start,
                end,
            )
            got = served[req].float()
            ref_hot = ref[:, hot].float()
            error = (got - ref_hot).abs().max().item()
            diff = error / ref_hot.abs().max().item()
            agree = got.argmax(-1) == ref_hot.argmax(-1)
            full_agree = (hot[got.argmax(-1)] == ref.argmax(-1)).float().mean().item()
            # A slot whose reference lead over its runner-up exceeds twice the
            # largest logit error cannot flip by rounding; random context makes
            # many slots near-ties, which may.
            top2 = ref_hot.topk(2, dim=-1).values
            decisive = (top2[:, 0] - top2[:, 1]) > 2 * error
            print(
                f"{'compiled' if compiled else 'eager':8s} request {req}: max |served - reference| = "
                f"{error:.3f} ({diff:.2e} of max |reference|), argmax agreement {agree.float().mean().item():.0%} "
                f"(vs full-vocabulary reference {full_agree:.0%}), "
                f"{int(decisive.sum())} decisive slots"
            )
            # A gross-error bound: bf16 rounding through six DiT layers and the
            # folded head reaches ~8% of the largest logit on random context,
            # FP8 weights or an e4m3 KV pool ~15% (their acceptance is measured
            # end to end).
            lossy = draft.fp8 or args.kv_dtype == "fp8"
            assert diff < (2.5e-1 if lossy else 1e-1), diff
            assert (
                lossy or agree[decisive].all()
            ), "a decisive slot drafted another token"
        assert math.isfinite(served.float().sum().item())
    print("ok")


if __name__ == "__main__":
    main()
