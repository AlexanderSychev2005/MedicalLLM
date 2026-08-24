"""
Batched inference over the full test.jsonl split, saved to a file for later
comparison (across checkpoints, or against the base untuned model). Not scoring
here - just producing generated vs reference pairs to score/eyeball afterward.

Run on the devcloud box - test.jsonl is MIMIC-derived, same as training data.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a checkpoint-N directory. Omit to eval the base model with no adapter.")
    parser.add_argument("--test_path", type=str, default="data/processed/hybrid/test.jsonl")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    # DiReCT/discharge_ie reference outputs regularly run past 400 tokens (long
    # evidence-citation chains / full JSON) and were getting cut off mid-answer.
    parser.add_argument("--max_new_tokens", type=int, default=900)
    args = parser.parse_args()

    print(f"Loading base model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"  # required for batched decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if args.checkpoint:
        print(f"Attaching LoRA adapter from {args.checkpoint}...")
        model = PeftModel.from_pretrained(model, args.checkpoint)
    else:
        print("No checkpoint given - evaluating the base model as-is.")
    model.eval()

    examples = [json.loads(line) for line in open(args.test_path, encoding="utf-8")]
    print(f"Running inference on {len(examples)} test examples, batch_size={args.batch_size}...")

    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for i, batch in enumerate(batched(examples, args.batch_size)):
            prompts = [tokenizer.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True) for ex in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    # Greedy decoding on a mid-training checkpoint can fall into
                    # repeating itself (saw a 15-item near-identical list on
                    # med_case_reasoning) - discourage looping without adding
                    # sampling randomness.
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=4,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for ex, generated in zip(batch, generated_texts):
                out_f.write(json.dumps({
                    "task": ex.get("task", "?"),
                    "user": ex["messages"][1]["content"],
                    "generated": generated,
                    "reference": ex["messages"][2]["content"],
                }, ensure_ascii=False) + "\n")
            out_f.flush()

            done = min((i + 1) * args.batch_size, len(examples))
            print(f"  {done}/{len(examples)} done")

    print(f"Done! Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
