"""
Plain Transformers + PEFT + TRL training, no Unsloth.

Unsloth's own attention_dispatch.py forces the quadratic-memory SDPA "math" backend
whenever it can't find the flash-attn/xformers *packages* specifically - even when
PyTorch's own SDPA has working flash/memory-efficient backends available underneath
(confirmed directly: torch.nn.attention.sdpa_kernel(FLASH_ATTENTION/EFFICIENT_ATTENTION)
works fine in isolation on this box). This script skips Unsloth's model-loading path
entirely and sets attn_implementation="sdpa" so PyTorch picks its own best backend,
to see whether that alone fixes the OOMs Unsloth kept hitting.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/processed/hybrid/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/processed/hybrid/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/hf_hybrid")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from, or 'True' for latest")
    args = parser.parse_args()

    print(f"1. Loading {args.model_name} via plain Transformers (no Unsloth)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        # Let PyTorch's own SDPA dispatcher pick flash/mem-efficient/math itself,
        # instead of routing through Unsloth's package-presence-only check.
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("2. Setting up LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("3. Preparing chat-formatted data...")

    def format_chatml(examples):
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
        return {"text": texts}

    print(f"Loading datasets from {args.train_path} and {args.val_path}...")
    dataset_train = load_dataset("json", data_files=args.train_path, split="train")
    dataset_val = load_dataset("json", data_files=args.val_path, split="train")
    dataset_train = dataset_train.map(format_chatml, batched=True)
    dataset_val = dataset_val.map(format_chatml, batched=True)

    print("4. Configuring Trainer...")
    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        packing=False,  # start conservative - flip on once a run is confirmed stable
        logging_steps=10,
        report_to="tensorboard",  # newer TRL dropped logging_dir; defaults under output_dir,
                                   # or set TENSORBOARD_LOGGING_DIR env var to override
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=50,
        output_dir=args.output_dir,
        optim="adamw_8bit",
        seed=42,
        # Newer TRL (1.10.0) moved these from SFTTrainer's own kwargs into SFTConfig,
        # and renamed max_seq_length -> max_length (confirmed via dataclasses.fields).
        dataset_text_field="text",
        max_length=args.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # newer TRL/Trainer renamed tokenizer= to this
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        args=training_args,
    )

    print("\nStarting training (plain Transformers + PEFT + TRL, no Unsloth)...\n")

    resume_arg = None
    if args.resume:
        resume_arg = True if args.resume.lower() == "true" else args.resume
        print(f"Resuming from checkpoint: {resume_arg}")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_arg)

    print("\n5. Saving final model adapter...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done! Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
