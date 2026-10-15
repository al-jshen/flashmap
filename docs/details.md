# FlashMap: API, internals and checks

The details behind the [README](../README.md): the API of the `transformers` implementation, how the drafter
works, how this implementation was checked, serving with SGLang and vLLM, how to convert checkpoints, and
troubleshooting.

## Using the drafter

The drafters use the target's token embedding and lm_head, which are not part of their weights, so a
drafter only works with the target it was trained for. Run the drafter, the target and its embedding and
lm_head on one device. Each model folder also has `draft_vocab.safetensors`, the target's response tokens
ranked by frequency. This implementation does not need it; the serving integrations use its top 65,536
tokens as a smaller draft head. The drafters were trained on sequences of up to 4096 tokens: longer contexts
stay lossless, but their acceptance is untested.

The examples load the model code from the checkpoint, so they also run without installing this package:

```bash
git clone https://github.com/al-jshen/flashmap && cd flashmap
uv pip install -e ".[benchmark,test]"
python examples/generate.py --draft j-shen/FlashMap-Qwen3-8B --target Qwen/Qwen3-8B --prompt "..."
python examples/benchmark.py --draft j-shen/FlashMap-Qwen3-8B --target Qwen/Qwen3-8B \
    --dataset gsm8k --num-prompts 16 --baseline
python -m pytest tests   # tiny random models (GPU and engine tests skip when unavailable)
```

With the package installed, `import flashmap` registers FlashMap with `AutoModel`, which then loads FlashMap
folders without `trust_remote_code`. The package is tested with transformers 4.57, 5.10, 5.16 and 5.20 and
torch 2.9 and 2.13 (`transformers>=4.56` is needed for the `dtype` argument).

This implementation favors readability over speed (eager PyTorch, a full-vocabulary lm_head, batch size 1).
For serving, use SGLang with `flashmap[sglang]` ([Serving with SGLang](#serving-with-sglang)) or vLLM with
`flashmap[vllm]` ([Serving with vLLM](#serving-with-vllm)).

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

## Checking this implementation

On the same prompts, the acceptance lengths of `spec_generate` match those of DeepSpec's own evaluator, for
both drafters, one and two steps, at T = 0 and T = 1 (512 new tokens; the first 16 prompts of the GSM8K and
HumanEval sets of DFlash's evaluation and every fifth MT-Bench question; one H100, bfloat16). Even at T = 0
the drafts depend on the draft noise, so two runs on 16 prompts differ slightly.

With the same noise, the draft logits equal DeepSpec's exactly in bfloat16 (on the H100) and to 5e-5 (one
step) and 9e-4 (two steps) in float32. At temperature 0 the output is the target's own greedy decoding:
identical on every prompt checked with the target in float32 (6 prompts with Qwen3-8B, 9 with Qwen3-4B, one
and two steps). In bfloat16 the batched verification pass and token-by-token decoding round differently, so
the two can part where the target's top two logits tie to within bfloat16 precision. `tests/` checks greedy
and sampled losslessness on tiny random models.

## Serving with SGLang

`flashmap[sglang]` installs `flashmap.sglang`, an SGLang plugin (entry point group `sglang.srt.plugins`,
name `flashmap`) that SGLang runs at startup in each of its processes. Without changing SGLang, it adds the
`FlashMapDraftModel` architecture, which SGLang's DSPARK speculative worker drafts with, and it serves a
release folder, local or on the Hub, as it is:

```bash
uv pip install "sglang==0.5.20" "flashmap[sglang] @ git+https://github.com/al-jshen/flashmap"
python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DSPARK --speculative-draft-model-path j-shen/FlashMap-Qwen3-8B
```

- Requirements: sglang 0.5.20 exactly (on another version the plugin logs a warning and leaves SGLang
  untouched; `FLASHMAP_SGLANG_ANY_VERSION=1` tries anyway), tensor parallelism 1 and a target whose
  vocabulary is not padded. Each is checked when the drafter loads.
- The plugin registers the model through `SGLANG_EXTERNAL_MODEL_PACKAGE=flashmap.sglang`, SGLang's
  mechanism for models outside its tree, which it sets for the server's processes. Set it yourself where
  SGLang's plugins do not run, or when `SGLANG_PLUGINS` lists others but not `flashmap`.
- The block size (16) comes from config.json. `--speculative-dspark-block-size G` with `G < 16` verifies
  only the first `G` drafts of each block, which can pay off at high concurrency.
- The draft head covers the first `draft_vocab_size` (65,536) tokens of `draft_vocab.safetensors`, which
  SGLang loads with the weights; the target still verifies every token, so outputs are unchanged.
  `FLASHMAP_DRAFT_VOCAB=0` drafts over the full vocabulary.
- Sampling at `temperature > 0` needs no option: DSPARK samples the drafts from FlashMap's distribution and
  verifies them against it.
- Options, set in the server's environment unless noted: `FLASHMAP_NUM_STEPS=2` drafts with two jumps
  (higher acceptance, slower drafts); `FLASHMAP_FP8=1` keeps FP8 copies of the draft's large weights (Ada
  or Hopper); `--speculative-draft-kv-cache-dtype fp8_e4m3` stores the draft's KV cache in FP8, which
  leaves more memory for the target's. `FLASHMAP_COMPILE=0` turns off `torch.compile` of the draft step.
- Tree drafting: `FLASHMAP_TREE_N=8` drafts up to 8 blocks per request from independent noise in one pass
  and verifies their trie, cut to the block size, in one target pass (`FLASHMAP_TREE_BUDGET`, 32 by
  default, caps the blocks per batch). It serves greedy requests only: a request that samples, asks for
  logprobs or constrains its output is refused with an error saying why. It needs
  `SGLANG_DSPARK_FOLDED_PROPOSAL=0`, `--page-size 1` and a target attention backend that applies tree masks
  (`fa3`, `flashinfer` or `triton`); with `--attention-backend triton`, `FLASHMAP_SPLITKV_VERIFY=1` verifies
  with a split-KV attention kernel. It pays off at low concurrency.

The plugin reads these config.json fields, which `scripts/convert_checkpoint.py` writes next to the
model's own:

| Field | Value | Use |
| --- | --- | --- |
| `max_position_embeddings` | `131072` | the longest context the drafter is served with (SGLang refuses a target context longer than the draft's) |
| `dspark_config` | `{"markov_rank": 1, "markov_head_type": "vanilla", "mask_token_id": 151669}` | DSPARK's own fields: the plugin replaces DSPARK's rank-1 Markov head by an identity (FlashMap has none), and the mask token fills block slots FlashMap never reads |
| `draft_vocab_size` | `65536` | size of the draft head's vocabulary, taken from `draft_vocab.safetensors` (shared with vLLM) |

`scripts/check_sglang.py` checks the served drafter against `FlashMapDraftModel` on a GPU, with the
target's embedding and lm_head:

```bash
python scripts/check_sglang.py --draft FlashMap-Qwen3-8B --target Qwen/Qwen3-8B
```

The same integration lives in [our SpecForge fork](https://github.com/al-jshen/SpecForge/tree/flashmap)
(`specforge.serving.sglang`), which trains FlashMap and is where the integration would be upstreamed to
SGLang from; its
[SGLang guide](https://github.com/al-jshen/SpecForge/blob/flashmap/docs/sections/basic_usage/flashmap_sglang.md)
covers SpecForge's export path. With `SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang`, the plugin
leaves FlashMap to SpecForge.

## Serving with vLLM

`flashmap[vllm]` installs `flashmap.vllm`, a vLLM general plugin (entry point group `vllm.general_plugins`,
name `flashmap`) that vLLM loads in each of its processes. Without changing vLLM, it adds the
`FlashMapDraftModel` architecture and the speculator that runs it on vLLM's DFlash runtime, and it serves a
release folder, local or on the Hub, as it is:

```bash
uv pip install "vllm==0.31.*" "flashmap[vllm] @ git+https://github.com/al-jshen/flashmap"
vllm serve Qwen/Qwen3-8B --speculative-config '{"model": "j-shen/FlashMap-Qwen3-8B", "num_speculative_tokens": 16}'
```

- Requirements: vLLM 0.31.x (on other versions the plugin logs a warning and leaves vLLM untouched), vLLM's
  V2 model runner (the default), tensor parallelism 1, the FlashAttention backend and a bf16 target. Each is
  checked when the drafter loads.
- The plugin fills in `"method": "dflash"` (FlashMap runs on vLLM's DFlash runtime) and, when it is left out,
  `num_speculative_tokens` (the block size, 16). A smaller `num_speculative_tokens` verifies only the first
  drafts of each block, which can pay off at high concurrency.
- `vllm serve j-shen/FlashMap-Qwen3-8B` also works: vLLM then serves the target named in the drafter's
  config.
- vLLM's prefix caching (on by default) works as usual.
- The draft head covers the first `draft_vocab_size` (65,536) tokens of `draft_vocab.safetensors`; the
  target still verifies every token, so outputs are unchanged. Without `draft_vocab_size` in config.json,
  the drafter drafts over the full vocabulary.
- For sampling at `temperature > 0`, add `"draft_sample_method": "probabilistic"`, so that drafts are sampled
  from FlashMap's distribution and verified against it.
- `FLASHMAP_NUM_STEPS=2` in the server's environment drafts with two jumps (higher acceptance, slower
  drafts); `FLASHMAP_FP8=1` keeps FP8 copies of the draft's large weights (Ada or Hopper; opt-in, as it does
  not speed up every batch size).
- `"kv_cache_dtype": "fp8_e4m3"` in the speculative config stores the draft's KV cache in FP8 (Hopper,
  FlashAttention 3), which leaves more memory for the target's KV cache.

The plugin reads these config.json fields, which `scripts/convert_checkpoint.py` writes next to the
model's own:

| Field | Value | Use |
| --- | --- | --- |
| `speculators_model_type` | `"flashmap"` | marks a vLLM draft model in the speculators format |
| `speculators_config` | algorithm, proposal (16 greedy tokens), verifier (the target) | the defaults `vllm serve <folder>` serves with |
| `transformer_layer_config` | `{}` | required by vLLM's format; the plugin reads the DiT's sizes from the top-level fields |
| `draft_vocab_size` | `65536` | size of the draft head's vocabulary, taken from `draft_vocab.safetensors` |

The plugin is the only vLLM integration:
[our speculators fork](https://github.com/al-jshen/speculators/tree/flashmap) trains FlashMap and converts its
checkpoints to this release format, but does not serve them.

## Converting checkpoints

FlashMap can be trained with speculators (vLLM) or SpecForge (SGLang); see [Training](../README.md#training).
The speculators fork writes a release folder itself (`speculators convert <flow-map checkpoint> --algorithm
flashmap --verifier <target> --output-path <folder>`, see its
[Training FlashMap](https://github.com/al-jshen/speculators/blob/flashmap/docs/user_guide/algorithms/flashmap_training.md#serving-the-result)).
`scripts/convert_checkpoint.py` recognizes speculators checkpoints and SpecForge's `specforge export --to hf`
folders, and writes a release folder for this package: the config (with the fields
[SGLang](#serving-with-sglang) and [vLLM](#serving-with-vllm) serve it with), the weights without the target's
embedding and lm_head, the model code and, optionally, a model card and the draft-vocabulary ranking. It checks
the weights against the source and against the names and shapes the model expects.

```bash
python scripts/convert_checkpoint.py --checkpoint <checkpoint folder> --target-model Qwen/Qwen3-8B \
    --output FlashMap-Qwen3-8B --model-card model_cards/FlashMap-Qwen3-8B.md
```

`--draft-vocab` ships a token-frequency ranking, and `--target-model` names the target where the checkpoint's
config does not. The released drafters were trained with the original research code (not public), built on
[DeepSpec](https://github.com/deepseek-ai/DeepSpec); the converter also reads its checkpoints, which is how the
release folders were made.

## Troubleshooting

On hosts without a CUDA toolkit, the engines can fail while compiling their own just-in-time kernels:
vLLM in FlashInfer's sampler and SGLang in DeepGEMM. This is on the engine side, not in FlashMap. Load a CUDA
toolkit (with `nvcc`), or turn those kernels off: `VLLM_USE_FLASHINFER_SAMPLER=0` for vLLM and
`SGLANG_ENABLE_JIT_DEEPGEMM=0` for SGLang.
