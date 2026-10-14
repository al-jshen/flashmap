"""Chat with a Qwen3 model accelerated by FlashMap speculative decoding.

    python examples/generate.py --draft al-jshen/FlashMap-Qwen3-8B --target Qwen/Qwen3-8B \
        --prompt "How many positive whole-number divisors does 196 have?"
"""

import argparse
import time

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", default="al-jshen/FlashMap-Qwen3-8B", help="hub id or local folder")
    parser.add_argument("--target", default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt", default="How many positive whole-number divisors does 196 have?")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=1, help="flow-map jumps per draft (1 or 2)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()
    draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True
    )["input_ids"].to(device)

    # warm-up: CUDA kernels, the cached noise factor and time conditioning
    draft.spec_generate(target, input_ids, max_new_tokens=16, num_steps=args.num_steps)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = draft.spec_generate(
        target,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_steps=args.num_steps,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    print(tokenizer.decode(output.sequences[0, input_ids.shape[1]:], skip_special_tokens=True))
    print(
        f"\n{output.num_new_tokens} tokens in {seconds:.2f}s "
        f"({output.num_new_tokens / seconds:.1f} tok/s), acceptance length {output.acceptance_length:.2f}"
    )


if __name__ == "__main__":
    main()
