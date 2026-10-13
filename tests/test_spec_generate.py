"""Losslessness of spec_generate, with tiny random models on the CPU.

    python -m pytest tests
"""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from flashmap import FlashMapConfig, FlashMapDraftModel


def tiny_models(vocab_size: int, hidden_size: int = 32, seed: int = 0):
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
            block_size=4,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            cond_dim=16,
            time_embed_dim=16,
        )
    ).eval()
    return target, draft


def test_greedy_is_target_greedy():
    target, draft = tiny_models(vocab_size=64)
    input_ids = torch.randint(64, (1, 7))
    expected = target.generate(input_ids, max_new_tokens=40, do_sample=False)
    for num_steps in (1, 2):
        out = draft.spec_generate(target, input_ids, max_new_tokens=40, num_steps=num_steps, stop_token_ids=[])
        assert torch.equal(out.sequences, expected)
        assert sum(out.acceptance_lengths) >= out.num_new_tokens - 1


@torch.no_grad()
def test_sampling_is_target_sampling():
    """The first two new tokens (the second one drawn by rejection sampling against
    the draft) follow the target's own joint distribution."""
    vocab_size, temperature, n = 12, 1.0, 4000
    target, draft = tiny_models(vocab_size, hidden_size=8)
    input_ids = torch.randint(vocab_size, (1, 5))
    # exact p(y1, y2)
    first = (target(input_ids).logits[0, -1] / temperature).softmax(-1)
    pairs = torch.cat([input_ids.expand(vocab_size, -1), torch.arange(vocab_size)[:, None]], dim=1)
    second = (target(pairs).logits[:, -1] / temperature).softmax(-1)
    exact = (first[:, None] * second).flatten()

    generator = torch.Generator().manual_seed(0)
    counts = torch.zeros(vocab_size * vocab_size)
    for _ in range(n):
        out = draft.spec_generate(
            target, input_ids, max_new_tokens=2, temperature=temperature, stop_token_ids=[], generator=generator
        )
        y1, y2 = out.sequences[0, -2:].tolist()
        counts[y1 * vocab_size + y2] += 1
    tv = 0.5 * (counts / n - exact).abs().sum()
    # the same statistic for n exact draws
    reference = torch.stack(
        [0.5 * (torch.bincount(torch.multinomial(exact, n, replacement=True), minlength=exact.numel()) / n - exact).abs().sum()
         for _ in range(200)]
    )
    assert tv <= reference.quantile(0.99), (tv, reference.quantile(0.99))
