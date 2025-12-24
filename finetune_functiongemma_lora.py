#!/usr/bin/env python3
"""
Finetune FunctionGemma using LoRA (Low-Rank Adaptation) for faster training on CPU.
Much faster than full finetuning while maintaining quality.
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


class FunctionGemmaDataset:
    """Custom dataset for FunctionGemma finetuning"""

    def __init__(self, data_path: str = "training_data.json", tokenizer=None, max_length: int = 128):
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Load training data
        with open(data_path, "r") as f:
            data = json.load(f)

        self.examples = data.get("train", [])
        self.val_examples = data.get("validation", [])
        print(f"✓ Loaded {len(self.examples)} training examples")
        print(f"✓ Loaded {len(self.val_examples)} validation examples")

    def format_prompt(self, query: str, output: str) -> str:
        """Format prompt in instruction-following format"""
        prompt = f"""Query: {query}
Output: {output}"""
        return prompt

    def create_hf_dataset(self, split: str = "train") -> Dataset:
        """Create HuggingFace Dataset"""
        examples = self.examples if split == "train" else self.val_examples

        texts = []
        for ex in examples:
            query = ex["input"]
            output = ex["output"]
            prompt = self.format_prompt(query, output)
            texts.append(prompt)

        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        dataset = Dataset.from_dict(encodings)
        dataset = dataset.map(self._add_labels, batched=False)

        return dataset

    def _add_labels(self, example):
        """Add labels for language modeling"""
        example["labels"] = example["input_ids"].copy()
        return example


def setup_lora_finetuning():
    """Setup and run LoRA-based finetuning"""
    print("=" * 70)
    print("FunctionGemma CSV Format Finetuning with LoRA")
    print("=" * 70)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model and tokenizer
    print("\nLoading FunctionGemma-270m-it...")
    model_name = "google/functiongemma-270m-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded: {model_name}")
    print(f"✓ Model parameters: {model.num_parameters() / 1e6:.1f}M")

    # Setup LoRA
    print("\nSetting up LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],  # Target specific modules for speed
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Create datasets
    print("\nCreating datasets...")
    dataset_loader = FunctionGemmaDataset(
        data_path="training_data.json",
        tokenizer=tokenizer,
        max_length=128,  # Shorter sequences for speed
    )

    train_dataset = dataset_loader.create_hf_dataset("train")
    val_dataset = dataset_loader.create_hf_dataset("validation")

    print(f"✓ Train dataset size: {len(train_dataset)}")
    print(f"✓ Val dataset size: {len(val_dataset)}")

    # Training arguments - optimized for CPU/speed
    output_dir = Path("./finetuned_functiongemma_lora")
    output_dir.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=2,  # Fewer epochs
        per_device_train_batch_size=4,  # Smaller batch
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,  # Slightly higher for LoRA
        weight_decay=0.01,
        warmup_steps=20,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42,
        report_to=[],
    )

    print(f"\n✓ Output directory: {output_dir}")
    print(f"✓ Training for {training_args.num_train_epochs} epochs")
    print(f"✓ Using LoRA for efficient training")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Start finetuning
    print("\n" + "=" * 70)
    print("Starting LoRA finetuning...")
    print("=" * 70 + "\n")

    trainer.train()

    # Save final model
    print("\n" + "=" * 70)
    print("Finetuning complete!")
    print("=" * 70)

    final_model_path = output_dir / "final_model"
    final_model_path.mkdir(exist_ok=True)
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print(f"✓ Final model saved to: {final_model_path}")

    # Show final metrics
    if hasattr(trainer.state, "best_metric"):
        print(f"Best eval loss: {trainer.state.best_metric:.4f}")


if __name__ == "__main__":
    setup_lora_finetuning()
