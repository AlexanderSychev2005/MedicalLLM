import torch
from unsloth import FastLanguageModel
import json
import random
import argparse

# MI300X Constraints
max_seq_length = 8192
dtype = torch.bfloat16
load_in_4bit = True

def load_examples():
    """Loads a mix of examples from the different datasets."""
    random.seed(42)
    examples = []
    
    # 1. MedAlign (Expert style) - from our untouched test set
    with open("processed_data/hybrid/test_medalign.jsonl", "r", encoding="utf-8") as f:
        medalign_data = [json.loads(line) for line in f.readlines()]
        # Take 5 random examples
        for item in random.sample(medalign_data, 5):
            examples.append({
                "source": "MedAlign (Real World)",
                "messages": item["messages"]
            })
            
    # 2. Extract Asclepius and MedQA from their pure test sets!
    asclepius_pool = []
    with open("processed_data/asclepius/test.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50: break
            asclepius_pool.append(json.loads(line))
            
    medqa_pool = []
    with open("processed_data/medqa/test.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50: break
            medqa_pool.append(json.loads(line))
                
    for item in random.sample(asclepius_pool, 5):
        examples.append({
            "source": "Asclepius (Synthetic EHR)",
            "messages": item["messages"]
        })
        
    for item in random.sample(medqa_pool, 5):
        examples.append({
            "source": "MedQA (Theory/USMLE)",
            "messages": item["messages"]
        })
        
    return examples

def generate_response(model, tokenizer, messages):
    # Format the prompt using ChatML, but STOP before the assistant's answer
    prompt = tokenizer.apply_chat_template(
        messages[:-1], # Pass both system and user messages!
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Clean up the prompt from the response
    return response.split("assistant\n")[-1].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", type=int, default=350, help="Checkpoint to benchmark against Base model")
    args = parser.parse_args()

    print("Loading test examples...")
    examples = load_examples()
    
    print("\n--- 1. Loading Base Model ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    results = []
    for i, ex in enumerate(examples):
        print(f"Base Model processing {ex['source']} ({i+1}/{len(examples)})...")
        base_out = generate_response(model, tokenizer, ex["messages"])
        
        # Save state
        results.append({
            "source": ex["source"],
            "instruction": ex["messages"][0]["content"],
            "ground_truth": ex["messages"][1]["content"],
            "base_output": base_out,
            "cp_output": ""
        })

    print(f"\n--- 2. Loading Adapter Checkpoint {args.cp} ---")
    model.load_adapter(f"output/unsloth_hybrid/checkpoint-{args.cp}", adapter_name=f"cp{args.cp}")
    model.set_adapter(f"cp{args.cp}")
    FastLanguageModel.for_inference(model)
    
    for i, ex in enumerate(examples):
        print(f"CP-{args.cp} processing {ex['source']} ({i+1}/{len(examples)})...")
        results[i]["cp_output"] = generate_response(model, tokenizer, ex["messages"])

    print("\nWriting results to Markdown...")
    with open("benchmark_results.md", "w", encoding="utf-8") as f:
        f.write("# Model Benchmark Results\n\n")
        
        for i, res in enumerate(results):
            f.write(f"## Example {i+1} [{res['source']}]\n\n")
            f.write("### Instruction (Prompt)\n")
            f.write(f"```text\n{res['instruction']}\n```\n\n")
            
            f.write("### Ground Truth (Target)\n")
            f.write(f"```text\n{res['ground_truth']}\n```\n\n")
            
            f.write("### Base Model (No Finetune)\n")
            f.write(f"```text\n{res['base_output']}\n```\n\n")
            
            f.write(f"### Checkpoint {args.cp}\n")
            f.write(f"```text\n{res['cp_output']}\n```\n\n")
            f.write("---\n\n")
            
    print("Done! Check benchmark_results.md")

if __name__ == "__main__":
    main()
