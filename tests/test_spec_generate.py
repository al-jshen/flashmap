"""spec_generate with tiny random models on the CPU.

    python -m pytest tests
"""

import itertools
import tempfile
import types

import torch
from transformers import AutoModel, Qwen3Config, Qwen3ForCausalLM

from flashmap import FlashMapCache, FlashMapConfig, FlashMapDraftModel


def tiny_models(vocab_size: int, hidden_size: int = 32, block_size: int = 4, seed: int = 0):
    # the vocabulary must outnumber the target's hidden size, as in any LLM, for E^T E to be invertible
    torch.manual_seed(seed)
    target = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()
    target.lm_head.weight.data.mul_(30)  # peaked distributions, so drafts get rejected
    draft = FlashMapDraftModel(
        FlashMapConfig(
            target_hidden_size=hidden_size,
            vocab_size=vocab_size,
            target_layer_ids=[0, 2],
            block_size=block_size,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            cond_dim=16,
            time_embed_dim=16,
        )
    ).eval()
    return target, draft


def replace_draft(draft, propose, run_real=True):
    """Make ``draft.draft`` return the logits ``propose(cache)``, after running the real
    draft (which writes the anchor into the cache)."""
    real = draft.draft

    def fake(self, anchor_ids, cache, *args, **kwargs):
        if run_real:
            real(anchor_ids, cache, *args, **kwargs)
        return propose(cache)

    draft.draft = types.MethodType(fake, draft)


def oracle(expected, vocab_size, block_size, seed=1):
    """Drafts that agree with ``expected`` (the target's greedy decoding) for a random
    number of slots, 0 to block_size, and are random after."""
    generator = torch.Generator().manual_seed(seed)

    def propose(cache):
        truth = expected[0, cache.length + 1 : cache.length + 1 + block_size]
        tokens = torch.randint(vocab_size, (block_size,), generator=generator)
        n = int(torch.randint(block_size + 1, (1,), generator=generator))
        tokens[: min(n, truth.numel())] = truth[:n]
        logits = torch.full((1, block_size, vocab_size), -10.0)
        logits[0, torch.arange(block_size), tokens] = 10.0
        return logits

    return propose


def test_greedy_is_target_greedy():
    target, draft = tiny_models(vocab_size=64)
    input_ids = torch.randint(64, (1, 7))
    expected = target.generate(input_ids, max_new_tokens=40, do_sample=False)
    for noise_gate in ("jump", "constant"):
        draft.config.noise_gate = noise_gate
        for num_steps in (1, 2):
            out = draft.spec_generate(target, input_ids, max_new_tokens=40, num_steps=num_steps, stop_token_ids=[])
            assert torch.equal(out.sequences, expected)


def test_greedy_with_any_number_accepted():
    """Every accepted count 0..K, stop tokens inside a block, and short generations."""
    target, draft = tiny_models(vocab_size=64)
    input_ids = torch.randint(64, (1, 7))
    prompt_length, block_size = input_ids.shape[1], draft.config.block_size
    expected = target.generate(input_ids, max_new_tokens=60, do_sample=False)
    replace_draft(draft, oracle(expected, 64, block_size))
    accepted = []
    for _ in range(10):
        out = draft.spec_generate(target, input_ids, max_new_tokens=50, stop_token_ids=[])
        assert torch.equal(out.sequences, expected[:, : prompt_length + 50])
        accepted += out.acceptance_lengths
    assert set(accepted) == set(range(1, block_size + 2))
    stop = int(expected[0, prompt_length + 20])
    first = int((expected[0, prompt_length:] == stop).nonzero()[0])
    for _ in range(10):
        out = draft.spec_generate(target, input_ids, max_new_tokens=50, stop_token_ids=[stop])
        assert torch.equal(out.sequences, expected[:, : prompt_length + first + 1])
    for max_new_tokens in (0, 1, 2, 3, 17):
        out = draft.spec_generate(target, input_ids, max_new_tokens=max_new_tokens, stop_token_ids=[])
        assert torch.equal(out.sequences, expected[:, : prompt_length + max_new_tokens])
        assert out.num_new_tokens == max_new_tokens


def test_context_cache_matches_a_fresh_one():
    """After rounds of partial acceptance, the drafter's context equals one built from a
    single target forward over the final sequence."""
    target, draft = tiny_models(vocab_size=64)
    input_ids = torch.randint(64, (1, 7))
    expected = target.generate(input_ids, max_new_tokens=60, do_sample=False)
    caches = []
    propose = oracle(expected, 64, draft.config.block_size)
    replace_draft(draft, lambda cache: caches.append(cache) or propose(cache))
    draft.spec_generate(target, input_ids, max_new_tokens=45, stop_token_ids=[])
    cache = caches[-1]
    fresh = FlashMapCache(draft.config, 1, cache.length + 1)
    with torch.no_grad():
        hidden_states = target(expected[:, : cache.length], output_hidden_states=True).hidden_states
    draft.extend_context(fresh, draft.select_target_hidden(hidden_states))
    torch.testing.assert_close(fresh.kv[..., : cache.length, :], cache.kv[..., : cache.length, :])


@torch.no_grad()
def test_sampling_is_target_sampling():
    """Four new tokens, drafted two at a time against a fixed draft distribution close to
    the target's (so every outcome happens: rejections at each slot, full acceptance
    and the bonus token), follow the target's own joint distribution."""
    vocab_size, block_size, length, temperature, n = 4, 2, 4, 0.7, 5000
    target, draft = tiny_models(vocab_size, hidden_size=2, block_size=block_size, seed=3)
    target.lm_head.weight.data.div_(10)
    input_ids = torch.randint(vocab_size, (1, 4), generator=torch.Generator().manual_seed(5))
    fixed = torch.randn(1, block_size, vocab_size, generator=torch.Generator().manual_seed(7)) * 0.5
    replace_draft(draft, lambda cache: fixed.clone(), run_real=False)
    # exact p(y_1 .. y_length)
    outcomes = torch.tensor(list(itertools.product(range(vocab_size), repeat=length)))
    sequences = torch.cat([input_ids.expand(len(outcomes), -1), outcomes], dim=1)
    logits = target(sequences).logits[:, input_ids.shape[1] - 1 : -1].float() / temperature
    exact = logits.log_softmax(-1).gather(-1, outcomes[..., None])[..., 0].sum(-1).exp()

    generator = torch.Generator().manual_seed(0)
    counts = torch.zeros(len(outcomes))
    accepted = set()
    for _ in range(n):
        out = draft.spec_generate(
            target, input_ids, max_new_tokens=length, temperature=temperature, stop_token_ids=[], generator=generator
        )
        y = out.sequences[0, -length:]
        counts[int((y * vocab_size ** torch.arange(length - 1, -1, -1)).sum())] += 1
        accepted.update(out.acceptance_lengths)
    assert accepted == set(range(1, block_size + 2))
    tv = 0.5 * (counts / n - exact).abs().sum()
    # the same statistic for n exact draws
    reference = torch.stack(
        [
            0.5 * (torch.bincount(torch.multinomial(exact, n, replacement=True), minlength=len(exact)) / n - exact)
            .abs()
            .sum()
            for _ in range(200)
        ]
    )
    assert tv <= reference.quantile(0.99), (tv, reference.quantile(0.99))


def test_save_and_reload():
    _, draft = tiny_models(vocab_size=64)
    with tempfile.TemporaryDirectory() as path:
        draft.save_pretrained(path)
        for kwargs in ({"trust_remote_code": True}, {"attn_implementation": "sdpa"}):
            loaded = AutoModel.from_pretrained(path, **kwargs)
            assert type(loaded).__name__ == "FlashMapDraftModel"
            for (name, a), b in zip(draft.state_dict().items(), loaded.state_dict().values()):
                assert torch.equal(a, b), name
