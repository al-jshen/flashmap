---
base_model: Qwen/Qwen3-8B
library_name: transformers
license: apache-2.0
tags:
  - flashmap
  - speculative-decoding
  - draft-model
  - flow-map
  - qwen3
---

# FlashMap-Qwen3-8B

A [FlashMap](https://github.com/al-jshen/flashmap) drafter for lossless speculative decoding of
[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

FlashMap drafts a whole block of 16 tokens in **one forward pass** of a small (0.5B) DiT, like
[DFlash](https://github.com/z-lab/dflash), but samples the block **jointly** instead of predicting every
position independently: the block is drawn by a distilled discrete flow map
([Flow Map Language Models](https://arxiv.org/abs/2602.16813)) that jumps from Gaussian noise to the 16
tokens, conditioned on the target model's hidden states. One jump (1 step) is the fastest drafter; two jumps
are a little slower and accept more tokens.

The target verifies each block in one forward pass and keeps the longest prefix it agrees with, so the
output is exactly what Qwen3-8B generates on its own (greedy decoding at temperature 0, the target's
sampling distribution above it).

## Quick start

Only `torch` and `transformers` are needed (the model code ships with the weights):

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

`num_steps=2` drafts with two jumps of the map. The drafter uses the target's token embedding and
lm_head, so it is only valid with Qwen/Qwen3-8B. See the [GitHub repository](https://github.com/al-jshen/flashmap)
for the API and benchmarks.

## Results

Acceptance length (mean tokens committed per target forward pass, including the target's own token),
Qwen3-8B, thinking disabled, measured in SGLang at concurrency 1 with 2048 new tokens on 80 prompts per
benchmark. DFlash is retrained on the same 500K samples of Open-PerfectBlend with the same settings.

| T | Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | DFlash | 7.43 | 7.09 | 5.92 | 4.77 | 5.74 | 6.35 | 3.14 | 2.96 | 3.38 |
| 0 | FlashMap, 1 step | 7.69 | 7.24 | 6.00 | 4.95 | 5.82 | 6.65 | 3.34 | 3.14 | 3.71 |
| 0 | FlashMap, 2 steps | 8.31 | 7.69 | 6.36 | 5.24 | 6.26 | 7.19 | 3.40 | 3.20 | 3.80 |
| 1 | DFlash | 6.30 | 5.40 | 4.10 | 4.37 | 4.88 | 4.62 | 2.85 | 2.62 | 2.61 |
| 1 | FlashMap, 1 step | 6.40 | 5.40 | 4.37 | 4.29 | 4.87 | 4.90 | 3.00 | 2.81 | 2.83 |
| 1 | FlashMap, 2 steps | 7.11 | 5.74 | 4.52 | 4.75 | 5.35 | 5.06 | 3.16 | 2.96 | 2.95 |

Speedup over autoregressive decoding with Qwen3-8B (SGLang, one H200 NVL), FlashMap with 1 step:

| T | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6.14x | 5.89x | 4.89x | 4.02x | 4.73x | 5.26x | 2.73x | 2.56x | 3.04x |
| 1 | 5.12x | 4.39x | 3.56x | 3.47x | 3.95x | 3.85x | 2.44x | 2.28x | 2.30x |

The transformers implementation here is written for clarity rather than speed; use the SGLang or vLLM
integrations for serving.

## Model details

- Target: Qwen/Qwen3-8B (hidden states of layers 1, 9, 17, 25 and 33; token embedding and lm_head).
- Drafter: 6 DiT blocks, width 2048, 16 heads, 500M parameters, block size 16, bf16.
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
