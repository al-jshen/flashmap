"""Acceptance length and speedup of FlashMap speculative decoding on a set of prompts.

    python examples/benchmark.py --draft j-shen/FlashMap-Qwen3-8B --target Qwen/Qwen3-8B \
        --dataset gsm8k --num-prompts 16 --max-new-tokens 512

The prompts are the first --num-prompts rows of a Hugging Face dataset (gsm8k, humaneval,
mt-bench; formatted as in DFlash's evaluation, which sampled 80 rows of each), rendered
with the target's chat template and thinking disabled, or a local JSONL file with one
{"prompt": "..."} per line. The acceptance length is the mean number of tokens committed
per target forward pass, the draft's accepted tokens plus the target's own (as in DFlash's
evaluation). With --baseline the target also decodes every prompt on its own, which gives
the speedup and, at temperature 0, checks that the outputs are the target's greedy
decoding (exactly with --dtype float32; see the note it prints in bfloat16).
"""

import argparse
import json
import time

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Dataset, subset, split, and the user turn, formatted as in DFlash's evaluation.
DATASETS = {
    "gsm8k": (
        "openai/gsm8k", "main", "test",
        lambda row: f"{row['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    ),
    "humaneval": (
        "openai/openai_humaneval", None, "test",
        lambda row: "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{row['prompt']}\n```",
    ),
    "mt-bench": ("HuggingFaceH4/mt_bench_prompts", None, "train", lambda row: row["prompt"][0]),
}


def load_prompts(name: str, num_prompts: int) -> list:
    if name.endswith(".jsonl"):
        with open(name) as handle:
            return [json.loads(line)["prompt"] for line in handle][:num_prompts]
    from datasets import load_dataset

    path, subset, split, field = DATASETS[name]
    rows = load_dataset(path, subset, split=split)
    return [field(row) for row in rows.select(range(min(num_prompts, len(rows))))]


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", default="j-shen/FlashMap-Qwen3-8B")
    parser.add_argument("--target", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="gsm8k", help="gsm8k, humaneval, mt-bench or a .jsonl file")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--baseline", action="store_true", help="also decode with the target alone")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="dtype of the target and the drafter"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    dtype = getattr(torch, args.dtype)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to("cuda").eval()
    draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True, dtype=dtype).to("cuda").eval()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    prompts = load_prompts(args.dataset, args.num_prompts)

    def encode(prompt):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"].to("cuda")

    # warm-up: CUDA kernels, the cached noise factor and time conditioning
    draft.spec_generate(target, encode(prompts[0]), max_new_tokens=32, num_steps=args.num_steps)
    target.generate(encode(prompts[0]), max_new_tokens=8, do_sample=False)

    committed = rounds = new_tokens = base_tokens = matches = 0
    spec_seconds = base_seconds = 0.0
    for prompt in prompts:
        input_ids = encode(prompt)
        output, seconds = timed(
            lambda: draft.spec_generate(
                target,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_steps=args.num_steps,
                generator=generator,
            )
        )
        committed += sum(output.acceptance_lengths)
        rounds += len(output.acceptance_lengths)
        new_tokens += output.num_new_tokens
        spec_seconds += seconds
        if args.baseline:
            if args.temperature > 0:  # sample from the target's full distribution, as spec_generate does
                options = dict(do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0)
            else:
                options = dict(do_sample=False, temperature=1.0)
            baseline, seconds = timed(lambda: target.generate(input_ids, max_new_tokens=args.max_new_tokens, **options))
            base_tokens += baseline.shape[1] - input_ids.shape[1]
            base_seconds += seconds
            matches += torch.equal(baseline, output.sequences)

    print(f"{args.dataset}: {len(prompts)} prompts, {new_tokens} tokens, T={args.temperature}, steps={args.num_steps}")
    print(f"acceptance length {committed / max(rounds, 1):.2f}, {new_tokens / spec_seconds:.1f} tok/s")
    if args.baseline:
        base_speed = base_tokens / base_seconds
        print(f"target alone {base_speed:.1f} tok/s, speedup {new_tokens / spec_seconds / base_speed:.2f}x")
        if args.temperature == 0:
            print(f"identical to the target's greedy decoding: {matches}/{len(prompts)} prompts")
            if matches < len(prompts) and args.dtype == "bfloat16":
                print(
                    "  In bfloat16 the target's batched verification pass and its token-by-token decoding round\n"
                    "  differently, so the two can part where the target's top two logits tie to within bfloat16\n"
                    "  precision. Run with --dtype float32 for an exact check."
                )


if __name__ == "__main__":
    main()
