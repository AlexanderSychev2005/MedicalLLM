"""
Quick sanity-check inference: load the base model + a LoRA checkpoint, run it on a
handful of test.jsonl examples (one per task by default) and print input/generated/
reference side by side. Not a metrics run - see the benchmark reference doc for what
to actually score once there's a checkpoint worth scoring properly.

Run this on the devcloud box (same GPU/environment as training) - the test set is
MIMIC-derived, same reason it can't go through a third-party API.
"""
import argparse
import json
import random
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint-N directory")
    parser.add_argument("--test_path", type=str, default="data/processed/hybrid/test.jsonl")
    parser.add_argument("--per_task", type=int, default=1, help="How many examples to sample per task")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading base model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    print(f"Attaching LoRA adapter from {args.checkpoint}...")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    by_task = defaultdict(list)
    with open(args.test_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            by_task[ex.get("task", "?")].append(ex)

    random.seed(args.seed)
    for task, examples in sorted(by_task.items()):
        sample = random.sample(examples, min(args.per_task, len(examples)))
        for ex in sample:
            messages = ex["messages"]
            prompt_messages = messages[:2]  # system + user, model generates the assistant turn
            reference = messages[2]["content"]

            inputs = tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)

            print(f"\n{'=' * 20} {task} {'=' * 20}")
            print(f"USER:\n{prompt_messages[1]['content'][:500]}")
            print(f"\nGENERATED:\n{generated}")
            print(f"\nREFERENCE:\n{reference[:500]}")


if __name__ == "__main__":
    main()
