---
base_model: Qwen/Qwen3-4B
library_name: transformers
license: apache-2.0
tags:
  - flashmap
  - speculative-decoding
  - draft-model
  - flow-map
  - qwen3
---

# FlashMap-Qwen3-4B

A [FlashMap](https://github.com/al-jshen/flashmap) drafter for lossless speculative decoding of
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).

FlashMap drafts a whole block of 16 tokens in **one forward pass** of a small (0.5B) DiT, like
[DFlash](https://github.com/z-lab/dflash), but samples the block **jointly** instead of predicting every
position independently: the block is drawn by a distilled discrete flow map
([Flow Map Language Models](https://arxiv.org/abs/2602.16813)) that jumps from Gaussian noise to the 16
tokens, conditioned on the target model's hidden states. One jump (1 step) is the fastest drafter; two jumps
are a little slower and accept more tokens.

The target verifies each block in one forward pass and keeps the longest prefix it agrees with, so the
output is exactly what Qwen3-4B generates on its own (greedy decoding at temperature 0, the target's
sampling distribution above it).

## Quick start

Only `torch` and `transformers` are needed (the model code ships with the weights):

```python
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
draft = AutoModel.from_pretrained("al-jshen/FlashMap-Qwen3-4B", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True
)["input_ids"].cuda()

output = draft.spec_generate(target, input_ids, max_new_tokens=512, temperature=0.0, num_steps=1)
print(tokenizer.decode(output.sequences[0, input_ids.shape[1]:], skip_special_tokens=True))
print("acceptance length:", output.acceptance_length)
```

`num_steps=2` drafts with two jumps of the map. The drafter uses the target's token embedding and
lm_head, so it is only valid with Qwen/Qwen3-4B. See the [GitHub repository](https://github.com/al-jshen/flashmap)
for the API and benchmarks.

## Results

Acceptance length (mean tokens committed per target forward pass, including the target's own token),
Qwen3-4B, thinking disabled, measured in SGLang at concurrency 1 with 2048 new tokens on 80 prompts per
benchmark. DFlash is retrained on the same 500K samples of Open-PerfectBlend with the same settings.

| T | Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | DFlash | 7.18 | 7.13 | 6.20 | 5.11 | 5.47 | 5.96 | 3.30 | 3.10 | 3.50 |
| 0 | FlashMap, 1 step | 7.24 | 6.97 | 6.20 | 5.18 | 5.62 | 6.09 | 3.40 | 3.17 | 3.67 |
| 0 | FlashMap, 2 steps | 7.79 | 7.39 | 6.47 | 5.56 | 6.10 | 6.40 | 3.49 | 3.23 | 3.77 |
| 1 | DFlash | 6.46 | 5.40 | 4.46 | 4.47 | 5.02 | 4.51 | 2.95 | 2.94 | 2.75 |
| 1 | FlashMap, 1 step | 6.45 | 5.28 | 4.33 | 4.55 | 5.12 | 4.59 | 3.12 | 2.82 | 2.99 |
| 1 | FlashMap, 2 steps | 7.09 | 5.74 | 4.50 | 4.89 | 5.66 | 4.79 | 3.13 | 2.95 | 3.05 |

Speedup over autoregressive decoding with Qwen3-4B (SGLang, one H200 NVL), FlashMap with 1 step:

| T | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.14x | 5.02x | 4.47x | 3.71x | 4.04x | 4.28x | 2.46x | 2.28x | 2.66x |
| 1 | 4.59x | 3.81x | 3.12x | 3.26x | 3.69x | 3.24x | 2.25x | 2.04x | 2.17x |

The transformers implementation here is written for clarity rather than speed; use the SGLang or vLLM
integrations for serving.

## Model details

- Target: Qwen/Qwen3-4B (hidden states of layers 1, 9, 17, 25 and 33; token embedding and lm_head, which Qwen3-4B ties).
- Drafter: 6 DiT blocks, width 2048, 16 heads, 475M parameters, block size 16, bf16.
- Training: a flow over one-hot token blocks trained on 500K samples of
  [Open-PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend), then distilled into a
  flow map with the semigroup objective of Flow Map Language Models.

## Citation

```bibtex
@article{flashmap2026,
  title   = {{FlashMap: Flow Map Speculative Decoding}},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```
