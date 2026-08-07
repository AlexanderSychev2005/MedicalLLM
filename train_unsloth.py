from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/processed/hybrid/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/processed/hybrid/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/unsloth_hybrid")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from, or 'True' for latest")
    args = parser.parse_args()

    max_seq_length = args.max_length
    dtype = torch.bfloat16  # MI300X - native BF16, no need for FP16
    # 192GB fits the 70B model in full BF16 precision - skip 4-bit quantization noise
    # and ROCm's less mature bnb dequant kernels. ~146GB weights+LoRA, ~45GB headroom.
    load_in_4bit = False

    print("1. Loading Llama 3.1 70B via Unsloth...")
    # Full-precision repo, not the -bnb-4bit checkpoint - that one is pre-quantized and
    # can't be meaningfully loaded back out to BF16.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-70B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    print("2. Setting up LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", # The magic that saves VRAM!
    )

    print("3. Preparing ChatML Data format...")
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    
    def format_chatml(examples):
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
        return {"text": texts}

    print(f"Loading datasets from {args.train_path} and {args.val_path}...")
    dataset_train = load_dataset("json", data_files=args.train_path, split="train")
    dataset_val = load_dataset("json", data_files=args.val_path, split="train")
    
    dataset_train = dataset_train.map(format_chatml, batched=True)
    dataset_val = dataset_val.map(format_chatml, batched=True)

    print("4. Configuring Trainer for MI300X...")
    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,  # AMD MI300X loves BF16
        packing=True,  # pack short examples together, less padding waste
        logging_steps=10,
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/runs",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=50,
        output_dir=args.output_dir,
        optim="adamw_8bit", # Saves even more VRAM
        seed=42
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print("\n🚀 Starting Unsloth Training on MI300X! 🚀\n")
    
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
