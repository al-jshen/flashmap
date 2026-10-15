# FlashMap: Flow Map Speculative Decoding

**[Project page](https://jshen.net/flashmap)** · **Paper** (coming soon) · **[Models](#models)**

FlashMap is a drafter for lossless speculative decoding. Like [DFlash](https://github.com/z-lab/dflash), it
proposes a whole block of 16 tokens in one forward pass of a small network conditioned on the target model's
hidden states. Unlike DFlash, which predicts each token of the block independently, FlashMap draws the block
from a flow map ([Flow Map Language Models](https://arxiv.org/abs/2602.16813)), which jumps from Gaussian
noise to the whole block in a single step and can capture dependencies between its tokens. The target model
verifies every block, so the output is exactly the target's own. On Qwen3-8B, FlashMap decodes up to 6x faster
than the target alone.

This repository is a small, standalone implementation for Hugging Face `transformers`. For fast serving, use
the [SGLang](#serve-with-sglang) or [vLLM](#serve-with-vllm) integration.

## Models

| Drafter | Target | Drafter parameters |
|---|---|---|
| [al-jshen/FlashMap-Qwen3-8B](https://huggingface.co/al-jshen/FlashMap-Qwen3-8B) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 500M |
| [al-jshen/FlashMap-Qwen3-4B](https://huggingface.co/al-jshen/FlashMap-Qwen3-4B) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 475M |

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install git+https://github.com/al-jshen/flashmap
```

The package needs only `torch` and `transformers>=4.56`. It is tested with torch 2.9 and 2.13 and with
transformers 4.57, 5.10 and 5.16.

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashmap import FlashMapDraftModel

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda().eval()
draft = FlashMapDraftModel.from_pretrained("al-jshen/FlashMap-Qwen3-8B", dtype=torch.bfloat16).cuda().eval()

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True
)["input_ids"].cuda()

output = draft.spec_generate(target, input_ids, max_new_tokens=512, temperature=0.0, num_steps=1)
print(tokenizer.decode(output.sequences[0, input_ids.shape[1]:], skip_special_tokens=True))
print("acceptance length:", output.acceptance_length)
```

`num_steps=2` drafts with two jumps of the flow map: each draft is slower, but more of its tokens are
accepted. At `temperature > 0` the output follows the target's sampling distribution exactly. The model code
also ships with the weights, so `AutoModel.from_pretrained(..., trust_remote_code=True)` works without this
package. The [examples](examples/) measure acceptance length and speedup on GSM8K, HumanEval and MT-Bench,
and [docs/details.md](docs/details.md) documents the API and how the drafter works.

## Serve with SGLang

The SGLang integration lives in [our SpecForge fork](https://github.com/al-jshen/SpecForge/tree/flashmap).
It needs `sglang==0.5.20` exactly and tensor parallelism 1. SpecForge is installed without its dependencies,
because it pins a different sglang version for training.

```bash
uv venv --python 3.12 .venv-sglang
source .venv-sglang/bin/activate
uv pip install "sglang==0.5.20"
uv pip install --no-deps "git+https://github.com/al-jshen/SpecForge@flashmap"

# Export the release checkpoint for SGLang (64k-token draft vocabulary by default)
hf download al-jshen/FlashMap-Qwen3-8B --local-dir FlashMap-Qwen3-8B
python -m specforge.export.flashmap --src FlashMap-Qwen3-8B --output-dir FlashMap-Qwen3-8B-sglang

SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path FlashMap-Qwen3-8B-sglang \
  --speculative-dspark-block-size 16 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32
```

Set `FLASHMAP_NUM_STEPS=2` when launching to draft with two jumps. The
[SGLang guide](https://github.com/al-jshen/SpecForge/blob/flashmap/docs/sections/basic_usage/flashmap_sglang.md)
covers the options, tree drafting and the full benchmarks.

## Serve with vLLM

The vLLM integration is a plugin in [our speculators fork](https://github.com/al-jshen/speculators/tree/flashmap),
which vLLM loads automatically, so vLLM itself is unchanged. It needs vLLM 0.31.x, tensor parallelism 1 and
vLLM's FlashAttention backend; each requirement is checked when the drafter loads.

```bash
uv venv --python 3.12 .venv-vllm
source .venv-vllm/bin/activate
uv pip install "vllm==0.31.*" "git+https://github.com/al-jshen/speculators@flashmap"

# Convert the release checkpoint to the speculators format (64k-token draft vocabulary by default)
hf download al-jshen/FlashMap-Qwen3-8B --local-dir FlashMap-Qwen3-8B
speculators convert ./FlashMap-Qwen3-8B --algorithm flashmap --verifier Qwen/Qwen3-8B \
    --output-path ./FlashMap-Qwen3-8B-speculators

vllm serve ./FlashMap-Qwen3-8B-speculators
```

For sampling at `temperature > 0`, pass the speculative config explicitly, so that drafts are sampled from
FlashMap's own distribution:

```bash
vllm serve Qwen/Qwen3-8B --speculative-config '{"model": "./FlashMap-Qwen3-8B-speculators",
    "method": "dflash", "num_speculative_tokens": 16, "draft_sample_method": "probabilistic"}'
```

The [vLLM guide](https://github.com/al-jshen/speculators/blob/flashmap/docs/user_guide/algorithms/flashmap.md)
covers the requirements, options and full benchmarks.

### Sending requests

Both servers expose an OpenAI-compatible API, SGLang on port 30000 and vLLM on port 8000. The results below
were measured with Qwen3's thinking mode turned off.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")  # vLLM: port 8000
response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}],
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```


## Training

FlashMap is trained in two stages: a flow over one-hot token blocks, then a flow map distilled from it. The
recipes are in the speculators fork
([Training FlashMap](https://github.com/al-jshen/speculators/blob/flashmap/docs/user_guide/algorithms/flashmap_training.md))
and the SpecForge fork ([FlashMap](https://github.com/al-jshen/SpecForge/blob/flashmap/docs/sections/concepts/FlashMap.md)).
[`scripts/convert_checkpoint.py`](scripts/convert_checkpoint.py) turns a trained checkpoint into a release
folder for this package ([details](docs/details.md#converting-checkpoints)).



## License

The code is released under the [MIT license](LICENSE). The model weights are released under Apache-2.0,
following Qwen3.

FlashMap builds on [DFlash](https://github.com/z-lab/dflash),
[Flow Map Language Models](https://arxiv.org/abs/2602.16813), [DeepSpec](https://github.com/deepseek-ai/DeepSpec),
[SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm),
[speculators](https://github.com/vllm-project/speculators) and [SpecForge](https://github.com/sgl-project/SpecForge).
