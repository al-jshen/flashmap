# FlashMap: Flow Map Speculative Decoding

FlashMap is a drafter for lossless speculative decoding. Like
[DFlash](https://github.com/z-lab/dflash) it drafts a whole block of tokens in one forward pass of a small
network conditioned on the target model's hidden states; unlike DFlash, its block is drawn from a generative
model that can couple the positions instead of predicting each one independently. The block is drawn by a
flow map ([Flow Map Language Models](https://arxiv.org/abs/2602.16813)): a network that transports Gaussian
noise to the block's tokens in a single jump, distilled from a continuous flow over one-hot token blocks. One
jump is the fastest drafter but captures the joint structure of a block only partly (the paper's shuffle test
measures how much); two jumps cost a second network evaluation, capture more of it and accept more tokens.

This repository is a minimal, standalone implementation of FlashMap inference with Hugging Face
`transformers`: the drafter is a `PreTrainedModel` loadable with `AutoModel`, and `spec_generate` runs
lossless speculative decoding with the `transformers` model each drafter was trained for. Training code is
not included.

## Models

| Drafter | Target | Parameters |
|---|---|---|
| [al-jshen/FlashMap-Qwen3-8B](https://huggingface.co/al-jshen/FlashMap-Qwen3-8B) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 500M |
| [al-jshen/FlashMap-Qwen3-4B](https://huggingface.co/al-jshen/FlashMap-Qwen3-4B) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 475M |

The drafters use the target's token embedding and lm_head, which are not part of their weights. Each
model folder also has `draft_vocab.safetensors`, the target's response tokens ranked by frequency; this
implementation does not need it, and the serving integrations use its top 65,536 tokens as a smaller draft
head. They were
trained on sequences of up to 4096 tokens: longer contexts stay lossless, but their acceptance is untested.

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

Run the drafter, the target and its embedding and lm_head on one device. Tested with transformers 4.57,
5.10 and 5.16 and torch 2.9 and 2.13 (`transformers>=4.56` is needed for the `dtype` argument). The examples load the
model code from the checkpoint too, so they run without installing this repository:

```bash
python examples/generate.py --draft al-jshen/FlashMap-Qwen3-8B --target Qwen/Qwen3-8B --prompt "..."
python examples/benchmark.py --draft al-jshen/FlashMap-Qwen3-8B --target Qwen/Qwen3-8B \
    --dataset gsm8k --num-prompts 16 --baseline   # needs `datasets`
```

Installing this repository (`pip install -e .`) gives the same classes as a package: after `import flashmap`,
`AutoModel` loads FlashMap folders without `trust_remote_code`. `pip install -e ".[test]"; pytest tests` runs
the CPU tests.

## API

`FlashMapDraftModel.spec_generate(target, input_ids, max_new_tokens=256, temperature=0.0, num_steps=None,
stop_token_ids=None, generator=None)` decodes one prompt `input_ids` of shape `(1, P)` and returns a
`FlashMapGenerateOutput` with

- `sequences`: the prompt followed by the generated tokens, `(1, P + num_new_tokens)`;
- `num_new_tokens`;
- `acceptance_lengths`: the tokens committed by each target forward pass (accepted draft tokens plus the
  target's own; the last pass counts in full even where it runs past `max_new_tokens`); `acceptance_length`
  is their mean, the usual acceptance length (tau or gamma).

Every round drafts the 16 tokens after the last one, the target scores them in one forward pass, and the
longest prefix the target accepts is kept together with one token of the target's own. At temperature 0 the
output is the target's greedy decoding. Above it, drafts are sampled from `softmax(logits / temperature)`
per slot and accepted by rejection sampling against the target's `softmax(logits / temperature)`, so the
output is distributed exactly as the target's own samples. (Given its noise, a FlashMap draft's slots are
independent draws, so these per-slot probabilities are an exact proposal distribution.) `num_steps` is the
number of flow-map jumps per draft: 1 (default) or 2; the config also holds the jump grids for 3 and 4.
Generation stops after `max_new_tokens` or at a stop token (default: the target's `eos_token_id`). A
`generator` makes sampling reproducible and must live on the model's device. `spec_generate` checks that the
target's vocabulary and hidden size fit the drafter, and warns if the target is not the one it was trained
for. It reads the target's hidden states through forward hooks on the layers it needs, so a long prompt does
not keep every layer's activations.

`spec_generate` is built from three lower-level pieces, which other decoding loops can reuse:

- `FlashMapCache(draft.config, batch_size, max_length, device=draft.device, dtype=draft.dtype)`: the
  drafter's cross-attention keys and values;
- `draft.extend_context(cache, features)`: append the target's hidden states at newly verified positions
  (projected to keys and values once and cached, so a draft never reprocesses the context). The features
  come from `draft.select_target_hidden(target_output.hidden_states)`, or from the hooks of
  `with draft.target_features(target) as features: ... features()`;
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
for serving, use the SGLang (SpecForge) and vLLM (speculators) integrations (links TBD).

### Checking this implementation

On the same prompts, with 512 new tokens, the acceptance lengths of `spec_generate` match those of DeepSpec's
own evaluator (16 prompts per benchmark: the first 16 of the paper's GSM8K and HumanEval sets, every fifth
MT-Bench question; one H100, bfloat16). The paper's numbers come from SGLang with 80 prompts and 2048 new
tokens. Even at T = 0 the drafts depend on the draft noise, and with 16 prompts two runs differ by up to
about 0.3.

| Target | T | Steps | Implementation | GSM8K | HumanEval | MT-Bench |
|---|---|---|---|---|---|---|
| Qwen3-8B | 0 | 1 | this repository | 7.82 | 5.74 | 3.39 |
| | | | DeepSpec | 7.88 | 5.75 | 3.42 |
| | | | paper | 7.69 | 5.82 | 3.34 |
| | 0 | 2 | this repository | 8.43 | 6.12 | 3.56 |
| | | | DeepSpec | 8.34 | 6.03 | 3.50 |
| | | | paper | 8.31 | 6.26 | 3.40 |
| | 1 | 1 | this repository | 6.34 | 5.08 | 2.95 |
| | | | DeepSpec | 6.23 | 4.94 | 2.96 |
| | | | paper | 6.40 | 4.87 | 3.00 |
| Qwen3-4B | 0 | 1 | this repository | 7.30 | 5.62 | 3.32 |
| | | | DeepSpec | 7.05 | 5.69 | 3.30 |
| | | | paper | 7.24 | 5.62 | 3.40 |
| | 0 | 2 | this repository | 7.88 | 6.27 | 3.39 |
| | | | DeepSpec | 7.76 | 6.17 | 3.38 |
| | | | paper | 7.79 | 6.10 | 3.49 |
| | 1 | 1 | this repository | 6.42 | 5.15 | 3.12 |
| | | | DeepSpec | 6.53 | 5.39 | 3.05 |
| | | | paper | 6.45 | 5.12 | 3.12 |

With the same noise, the draft logits equal DeepSpec's exactly in bfloat16 (on the H100) and to 5e-5 (one
step) and 9e-4 (two steps) in float32. At temperature 0 the output is the target's own greedy decoding:
identical on every prompt checked with the target in float32 (6 prompts with Qwen3-8B, 9 with Qwen3-4B, one
and two steps). In bfloat16 the batched verification pass and token-by-token decoding round differently, so
the two can part where the target's top two logits tie to within bfloat16 precision. `tests/` checks greedy
and sampled losslessness on tiny random models.

A draft takes 6.4 ms with one step and 11.3 ms with two (H100, eager PyTorch, Qwen3-8B); `spec_generate`
decodes GSM8K answers at 235 tokens/s with one step against 51 tokens/s for the target alone.

## Training and converting checkpoints

FlashMap was trained in [a fork of DeepSpec](https://github.com/al-jshen/DeepSpec) (upstream:
[deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)); it can also be trained with speculators
(vLLM) or SpecForge (SGLang). Train with any of them, then convert for standalone use:
`scripts/convert_checkpoint.py` recognizes DeepSpec checkpoints, speculators checkpoints and SpecForge's
`specforge export --to hf` folders, and writes a release folder (config, weights without the target's
embedding and lm_head, the model code and, optionally, a model card and the draft-vocabulary ranking). It
checks the weights against the source and against the names and shapes the model expects.

```bash
python scripts/convert_checkpoint.py --checkpoint <run>/step_23268 --target-model Qwen/Qwen3-8B \
    --output FlashMap-Qwen3-8B --model-card model_cards/FlashMap-Qwen3-8B.md
```

## License

The code in this repository is under the MIT license (`LICENSE`). The model weights are released under
Apache-2.0, following Qwen3.

## Citation

```bibtex
@article{flashmap2026,
  title   = {{FlashMap: Flow Map Speculative Decoding}},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```
