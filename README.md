# FlashMap: Flow Map Speculative Decoding

FlashMap is a drafter for lossless speculative decoding. Like
[DFlash](https://github.com/z-lab/dflash) it drafts a whole block of tokens in one forward pass of a small
network conditioned on the target model's hidden states; unlike DFlash, it samples the block **jointly**
rather than predicting each position independently. The block is drawn by a flow map
([Flow Map Language Models](https://arxiv.org/abs/2602.16813)): a network that transports Gaussian noise to
the block's tokens in a single jump, distilled from a continuous flow over one-hot token blocks. One jump is
the fastest drafter; two jumps cost a second network evaluation and accept more tokens.

This repository is a minimal, standalone implementation of FlashMap inference with Hugging Face
`transformers`: the drafter is a `PreTrainedModel` loadable with `AutoModel`, and `spec_generate` runs
lossless speculative decoding with any `transformers` Qwen3 target. Training code is not included.

## Models

| Drafter | Target | Parameters |
|---|---|---|
| [al-jshen/FlashMap-Qwen3-8B](https://huggingface.co/al-jshen/FlashMap-Qwen3-8B) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 500M |
| [al-jshen/FlashMap-Qwen3-4B](https://huggingface.co/al-jshen/FlashMap-Qwen3-4B) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 475M |

The drafters use the target's token embedding and lm_head, which are not part of their weights.

## Quick start

The model code ships with the weights, so `torch` and `transformers` are all you need:

```python
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda().eval()
draft = AutoModel.from_pretrained("al-jshen/FlashMap-Qwen3-8B", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True
)["input_ids"].cuda()

output = draft.spec_generate(target, input_ids, max_new_tokens=512, temperature=0.0, num_steps=1)
print(tokenizer.decode(output.sequences[0, input_ids.shape[1]:], skip_special_tokens=True))
print("acceptance length:", output.acceptance_length)
```

Or install this repository (`pip install -e .`) and run the examples:

```bash
python examples/generate.py --draft al-jshen/FlashMap-Qwen3-8B --target Qwen/Qwen3-8B --prompt "..."
python examples/benchmark.py --draft al-jshen/FlashMap-Qwen3-8B --target Qwen/Qwen3-8B \
    --dataset gsm8k --num-prompts 16 --baseline   # needs `datasets`
```

## API

`FlashMapDraftModel.spec_generate(target, input_ids, max_new_tokens=256, temperature=0.0, num_steps=None,
stop_token_ids=None, generator=None)` decodes one prompt `input_ids` of shape `(1, P)` and returns a
`FlashMapGenerateOutput` with

- `sequences`: the prompt followed by the generated tokens, `(1, P + num_new_tokens)`;
- `num_new_tokens`;
- `acceptance_lengths`: the tokens committed by each target forward pass (accepted draft tokens plus the
  target's own); `acceptance_length` is their mean, the usual acceptance length (tau or gamma).

Every round drafts the 16 tokens after the last one, the target scores them in one forward pass, and the
longest prefix the target accepts is kept together with one token of the target's own. At temperature 0 the
output is the target's greedy decoding. Above it, drafts are sampled from `softmax(logits / temperature)`
per slot and accepted by rejection sampling against the target's `softmax(logits / temperature)`, so the
output is distributed exactly as the target's own samples. (Given its noise, a FlashMap draft's slots are
independent draws, so these per-slot probabilities are an exact proposal distribution.) `num_steps` is the
number of flow-map jumps per draft: 1 (default) or 2; the config also holds the jump grids for 3 and 4.
Generation stops after `max_new_tokens` or at a stop token (default: the target's `eos_token_id`).

`spec_generate` is built from three lower-level pieces, which other decoding loops can reuse:

- `FlashMapCache(config, batch_size, max_length)`: the drafter's cross-attention keys and values;
- `draft.extend_context(cache, draft.select_target_hidden(target_output.hidden_states))`: append the
  target's hidden states at newly verified positions (projected once and cached, so the cost of a draft
  does not grow with the context);
- `draft.draft(anchor_ids, cache, target.get_input_embeddings(), target.get_output_embeddings(), num_steps)`:
  the logits `(B, 16, V)` of the block after the anchor tokens.

## How it works

A block of `K = 16` tokens is a point in `R^{K x V}` on the path `x_s = (1 - s) x0 + s onehot(y)` from noise
`x0 ~ N(0, I)` (`s = 0`) to the one-hot tokens `y` (`s = 1`). The drafter is a two-time denoiser: given the
state at time `s` it predicts, for every slot, a distribution over the vocabulary that tells where the jump
to time `t` lands. One step is the jump from `s = 0` to `t = 1`, whose output distributions are the draft.

The network (`flashmap/modeling_flashmap.py`) is a 6-block DiT of width 2048:

- **State.** The state is read through the target's token embedding `E`: the embedding of the posterior
  over clean tokens it implies, plus, weighted by the jump length, its linear projection `x @ E` (at
  `s = 0` the posterior is uniform, so this is how the map sees its noise). A one-step draft only needs
  `x0 @ E ~ N(0, E^T E)`, which is drawn as `z @ L^T` with `z ~ N(0, I_H)` and `L L^T = E^T E`, so no
  vocabulary-sized noise is materialized.
- **Time.** The start and end times of the jump are embedded and modulate every block through adaptive
  LayerNorms (AdaLN-zero).
- **Context.** Each block self-attends across the 16 slots and cross-attends to the target's hidden states
  (layers 1, 9, 17, 25, 33, fused by a linear layer) at all positions before the anchor, plus the
  embedding of the anchor token. Their keys and values are computed once per verified token and cached.
- **Output.** A linear head maps back to the target's hidden size, and the target's lm_head gives the logits.

## Results

Acceptance length (tokens per target forward pass) on Qwen3-8B with thinking disabled, from the paper
(SGLang, concurrency 1, 2048 new tokens, 80 prompts per benchmark; DFlash retrained on the same data):

| T | Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | DFlash | 7.43 | 7.09 | 5.92 | 4.77 | 5.74 | 6.35 | 3.14 | 2.96 | 3.38 |
| 0 | FlashMap, 1 step | 7.69 | 7.24 | 6.00 | 4.95 | 5.82 | 6.65 | 3.34 | 3.14 | 3.71 |
| 0 | FlashMap, 2 steps | 8.31 | 7.69 | 6.36 | 5.24 | 6.26 | 7.19 | 3.40 | 3.20 | 3.80 |
| 1 | DFlash | 6.30 | 5.40 | 4.10 | 4.37 | 4.88 | 4.62 | 2.85 | 2.62 | 2.61 |
| 1 | FlashMap, 1 step | 6.40 | 5.40 | 4.37 | 4.29 | 4.87 | 4.90 | 3.00 | 2.81 | 2.83 |
| 1 | FlashMap, 2 steps | 7.11 | 5.74 | 4.52 | 4.75 | 5.35 | 5.06 | 3.16 | 2.96 | 2.95 |

With one step this is a 6.14x speedup on GSM8K at T = 0 (4.73x on HumanEval, 2.73x on MT-Bench) on an
H200 in SGLang. The model cards in `model_cards/` have the Qwen3-4B numbers.

This implementation favors readability over speed (eager PyTorch, a full-vocabulary lm_head, batch size 1);
for serving, use the SGLang or vLLM integrations.

## Converting DeepSpec checkpoints

FlashMap was trained in [DeepSpec](https://github.com/deepseek-ai/DeepSpec). `scripts/convert_deepspec_checkpoint.py`
turns a DeepSpec flow-map checkpoint into a release folder (config, weights without the target's
embedding and lm_head, the model code and a model card) and checks every converted tensor against the
source:

```bash
python scripts/convert_deepspec_checkpoint.py --checkpoint <run>/step_23268 --target-model Qwen/Qwen3-8B \
    --output FlashMap-Qwen3-8B --model-card model_cards/FlashMap-Qwen3-8B.md
```

## Citation

```bibtex
@article{flashmap2026,
  title   = {{FlashMap: Flow Map Speculative Decoding}},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```
