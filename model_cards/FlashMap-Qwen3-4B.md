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

**[Project page](https://jshen.net/flashmap)** · **[Code](https://github.com/al-jshen/flashmap)** · **Paper** (coming soon)

A [FlashMap](https://github.com/al-jshen/flashmap) drafter for lossless speculative decoding of
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).

FlashMap drafts a whole block of 16 tokens in **one forward pass** of a small (0.5B) DiT, like
[DFlash](https://github.com/z-lab/dflash), but from a generative model that can couple the positions of a
block instead of predicting each one independently: the block is drawn by a distilled flow map
([Flow Map Language Models](https://arxiv.org/abs/2602.16813)) that jumps from Gaussian noise to the 16
tokens, conditioned on the target model's hidden states. One jump (1 step) is the fastest drafter but
captures the joint structure of a block only partly; two jumps are a little slower, capture more of it and
accept more tokens.

The target verifies each block in one forward pass and keeps the longest prefix it agrees with, so the
output is exactly what Qwen3-4B generates on its own (greedy decoding at temperature 0, the target's
sampling distribution above it). In bfloat16, greedy outputs can differ from token-by-token decoding
where the target's top two logits tie within bfloat16 precision, since the target scores 17 tokens at once;
with the target in float32 they are identical.

## Quick start

Only `torch` and `transformers>=4.56` are needed, since the model code ships with the weights. Run the
drafter and the target on one device:

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
lm_head, so it is only valid with Qwen/Qwen3-4B. It was trained on sequences of up to 4096 tokens:
longer contexts stay lossless, but their acceptance is untested. The [GitHub repository](https://github.com/al-jshen/flashmap)
has the API, examples and an installable package (`uv pip install git+https://github.com/al-jshen/flashmap`).

## Serving with SGLang or vLLM

The code above is written for clarity rather than speed. For serving, use the SGLang or vLLM integration
(tensor parallelism 1). Both draft over the top 65,536 tokens of `draft_vocab.safetensors` by default.

SGLang 0.5.20 exactly, with [our SpecForge fork](https://github.com/al-jshen/SpecForge/blob/flashmap/docs/sections/basic_usage/flashmap_sglang.md):

```bash
uv pip install "sglang==0.5.20"
uv pip install --no-deps "git+https://github.com/al-jshen/SpecForge@flashmap"
hf download al-jshen/FlashMap-Qwen3-4B --local-dir FlashMap-Qwen3-4B
python -m specforge.export.flashmap --src FlashMap-Qwen3-4B --output-dir FlashMap-Qwen3-4B-sglang
SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang \
python -m sglang.launch_server --model-path Qwen/Qwen3-4B \
  --speculative-algorithm DSPARK --speculative-draft-model-path FlashMap-Qwen3-4B-sglang \
  --speculative-dspark-block-size 16 --max-running-requests 32 --cuda-graph-max-bs-decode 32
```

vLLM 0.31.x, with [our speculators fork](https://github.com/al-jshen/speculators/blob/flashmap/docs/user_guide/algorithms/flashmap.md):

```bash
uv pip install "vllm==0.31.*" "git+https://github.com/al-jshen/speculators@flashmap"
hf download al-jshen/FlashMap-Qwen3-4B --local-dir FlashMap-Qwen3-4B
speculators convert ./FlashMap-Qwen3-4B --algorithm flashmap --verifier Qwen/Qwen3-4B \
    --output-path ./FlashMap-Qwen3-4B-speculators
vllm serve ./FlashMap-Qwen3-4B-speculators
```

Use a separate environment for each engine. The linked guides cover sampling at temperature > 0, two-jump
drafting and benchmarks.

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

## Files

- `model.safetensors`, `config.json`: the drafter (bf16), without the target's embedding and lm_head.
- `configuration_flashmap.py`, `modeling_flashmap.py`: the model code `trust_remote_code` loads.
- `draft_vocab.safetensors` (optional): Qwen3-4B's response tokens ranked by frequency (`ids`, most
  frequent first, and their `counts`, from 50K training samples). The code here does not use it; the SGLang
  and vLLM integrations draft over its top 65,536 tokens, a smaller output head that is 2-3% faster with no
  measurable change in acceptance.

## Model details

- Target: Qwen/Qwen3-4B (hidden states of layers 1, 9, 17, 25 and 33; token embedding and lm_head, which Qwen3-4B ties).
- Drafter: 6 DiT blocks, width 2048, 16 heads, 475M parameters, block size 16, bf16.
- Training: a flow over one-hot token blocks trained on 500K samples of
  [Open-PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend), then distilled into a
  flow map with the semigroup objective of Flow Map Language Models.

## License

The model weights are released under Apache-2.0, following Qwen3; the model code (`*.py`, from the
[GitHub repository](https://github.com/al-jshen/flashmap)) is under the MIT license.

## Citation

```bibtex
@misc{flashmap2026,
  title  = {{FlashMap}: Flow Map Speculative Decoding},
  author = {Shen, Jeff and others},
  year   = {2026},
  note   = {Preprint, coming soon}
}
```
