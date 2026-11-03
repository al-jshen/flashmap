# FlashMap: Flow Map Speculative Decoding

**[Project page](https://jshen.net/flashmap)** · **[Paper](https://arxiv.org/abs/2610.24365)** · **[Models](https://huggingface.co/collections/j-shen/flashmap-6ad0feb955bdc30db59769fa)**

FlashMap is a drafter for lossless speculative decoding. It proposes a whole block of K tokens in a single pass. Unlike discrete diffusion drafters, which predict each token of the block independently and ignore inter-token dependencies, FlashMap uses a flow map ([Flow Map Language Models](https://arxiv.org/abs/2602.16813)), which jumps from Gaussian noise to the whole block in a single step and can capture dependencies between its tokens. On Qwen3-8B, FlashMap decodes up to 6x faster than the target alone.

This repository is a small, standalone implementation for Hugging Face `transformers`. For fast serving, use
the [SGLang](#serve-with-sglang) or [vLLM](#serve-with-vllm) integration.

## Models

| Drafter | Target | Drafter parameters |
|---|---|---|
| [j-shen/FlashMap-Qwen3-8B](https://huggingface.co/j-shen/FlashMap-Qwen3-8B) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 500M |
| [j-shen/FlashMap-Qwen3-4B](https://huggingface.co/j-shen/FlashMap-Qwen3-4B) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 475M |
| [j-shen/FlashMap-Qwen3.8-27B-FP8](https://huggingface.co/j-shen/FlashMap-Qwen3.8-27B-FP8) | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 770M |

The Qwen3.8 drafter drafts over a draft vocabulary (below) and is served with vLLM or SGLang only: the `transformers`
implementation undoes rejected drafts by cropping the target's cache, which Qwen3.8's linear-attention layers do not
support, and `spec_generate` refuses such targets.

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install flashmap
```

The package needs only `torch` and `transformers>=4.56`; serving takes the `sglang` or `vllm` extra (below). It
is tested with torch 2.9 and 2.13 and with transformers 4.57, 5.10, 5.16 and 5.20.

## Quickstart

```python
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda().eval()
draft = AutoModel.from_pretrained("j-shen/FlashMap-Qwen3-8B", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True
)["input_ids"].cuda()

output = draft.spec_generate(target, input_ids, max_new_tokens=512, temperature=0.0, num_steps=1)
print(tokenizer.decode(output.sequences[0, input_ids.shape[1]:], skip_special_tokens=True))
print(f"acceptance length: {output.acceptance_length:.2f}")
```

`num_steps=2` drafts with two jumps of the flow map (trading speed for accuracy). 
The model code also ships with the weights, so `AutoModel.from_pretrained(..., trust_remote_code=True)` works without this
package. 

## Serve with SGLang

The SGLang integration is a plugin in this package (the `sglang` extra), which SGLang loads automatically.

```bash
uv venv --python 3.12 .venv-sglang
source .venv-sglang/bin/activate
uv pip install "sglang==0.5.20" "flashmap[sglang]"

sglang serve --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DSPARK --speculative-draft-model-path j-shen/FlashMap-Qwen3-8B
```

[Our SpecForge fork](https://github.com/al-jshen/SpecForge/tree/flashmap) trains FlashMap with SGLang's
training library, and its `specforge export --to hf` writes a trained map as a release folder that
`flashmap[sglang]` serves as it is.

## Serve with vLLM

The vLLM integration is a plugin in this package (the `vllm` extra), which vLLM loads automatically.

```bash
uv venv --python 3.12 .venv-vllm
source .venv-vllm/bin/activate
uv pip install "vllm==0.32.*" "flashmap[vllm]"

vllm serve Qwen/Qwen3-8B --speculative-config '{"model": "j-shen/FlashMap-Qwen3-8B", "num_speculative_tokens": 16}'
```

The plugin supports vLLM 0.31 and 0.32. For sampling at `temperature > 0`, add `"draft_sample_method": "probabilistic"`
to the speculative config.

Prefix caching works with both integrations: vLLM's (`--enable-prefix-caching`) and SGLang's radix cache, which is on
by default. On a hybrid target such as Qwen3.8, vLLM caches in blocks sized to the target's linear-attention state (864
tokens for Qwen3.8-27B-FP8 with its drafter) and, as for every drafter that reads the target's hidden states,
recomputes the last cached block, so a shared prefix starts to be reused only after about two blocks.

Drafters you train yourself ([Training](#training)) are served the same way once converted to the release
format, including drafters whose DiT uses a SwiGLU MLP (`"mlp_type": "swiglu"` in `config.json`; the released
drafters use GELU).

## Sending requests

Both servers expose an OpenAI-compatible API, SGLang on port 30000 and vLLM on port 8000. 

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

## Troubleshooting

**Problem:** `vllm serve` fails at startup with `/usr/local/cuda/bin/nvcc: No such file or directory`.\
**Solution:** vLLM compiles FlashInfer's sampling kernel with `nvcc`. Install the CUDA toolkit, or use PyTorch's
sampler instead: `VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ...`.

**Problem:** SGLang fails at startup with `NVCC version must be at least 12.9`.\
**Solution:** SGLang starts DeepGEMM's compiler, which needs CUDA 12.9 or newer. Install it, or turn DeepGEMM off
(it only runs FP8 matmuls): `SGLANG_DEEPGEMM_PDL=0 SGLANG_ENABLE_JIT_DEEPGEMM=0 sglang serve ...`.

**Problem:** vLLM with an FP8 target (such as Qwen3.8-27B-FP8) fails at startup with `NVCC version must be at least
12.9` or `NVCC compiler is not executable`.\
**Solution:** vLLM compiles DeepGEMM's FP8 matmuls with `nvcc` 12.9 or newer. Install it, or use CUTLASS's FP8
matmuls instead: `VLLM_USE_DEEP_GEMM=0 vllm serve ...`.

**Problem:** vLLM with a hybrid target (Qwen3.5 architecture, such as Qwen3.8-27B) fails at startup with
`max_num_seqs (1024) exceeds available Mamba cache blocks`.\
**Solution:** Each running sequence holds one block of the target's linear-attention state. Lower `--max-num-seqs`
(for example to 64) or raise `--gpu-memory-utilization`.

**Problem:** vLLM with a hybrid target runs few requests at once with a drafter (about 12 for Qwen3.8-27B-FP8 on one
H200 with 16 speculative tokens), so throughput stops growing with concurrency.\
**Solution:** vLLM keeps a copy of the target's linear-attention state for every speculative token of every request,
and pads all cache pages to the drafter's larger page. Lowering `"num_speculative_tokens"` in the speculative config
(for example to 8; the drafter still drafts its whole block) cuts the copies, at some cost in tokens per verification.

## Training

FlashMap is trained in two stages: a flow over one-hot token blocks, then a flow map distilled from it. The
recipes are in the speculators fork
([Training FlashMap](https://github.com/al-jshen/speculators/blob/flashmap/docs/user_guide/algorithms/flashmap_training.md))
and the SpecForge fork ([FlashMap](https://github.com/al-jshen/SpecForge/blob/flashmap/docs/sections/concepts/FlashMap.md)).
Both write a trained map as a release folder for this package (`speculators convert --algorithm flashmap`,
`specforge export --to hf`), and [`scripts/convert_checkpoint.py`](https://github.com/al-jshen/flashmap/blob/master/scripts/convert_checkpoint.py) converts
speculators and SpecForge checkpoints too (`python scripts/convert_checkpoint.py --help`).

A drafter can also be trained on a draft vocabulary: the target's most frequent tokens (65,536 of Qwen3.8's
248,320), over which its flow lives, noise, posterior and predictions alike. Its release lists them as the
`draft_ids` in its weights and sets `state_vocab_size` in its config (with jump grids computed at that size);
the transformers model, vLLM and SGLang draft over those tokens only, and every other token gets draft
probability 0, so speculative decoding stays lossless.

## License

The code is released under the [MIT license](https://github.com/al-jshen/flashmap/blob/master/LICENSE). The model weights are released under Apache-2.0,
following Qwen3.

FlashMap builds on [DFlash](https://github.com/z-lab/dflash),
[Flow Map Language Models](https://arxiv.org/abs/2602.16813), [DeepSpec](https://github.com/deepseek-ai/DeepSpec),
[SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm),
[speculators](https://github.com/vllm-project/speculators) and [SpecForge](https://github.com/sgl-project/SpecForge).

## Citation

```bibtex
@misc{flashmap2026,
  title         = {{FlashMap}: Flow Map Speculative Decoding},
  author        = {Shen, Jeff and Zhou, Zhejian and Lee, Ryan and May, Jonathan and Ho, Shirley},
  year          = {2026},
  eprint        = {2610.24365},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.24365}
}
```
