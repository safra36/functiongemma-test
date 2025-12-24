#!/usr/bin/env python3
"""
Finetune FunctionGemma to output function calls in CSV format: function_name,arg1,arg2...
Instead of XML format: <start_function_call>call:function_name<end_function_call>
"""

import json
import torch
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


class FunctionGemmaDataset:
    """Custom dataset for FunctionGemma finetuning"""

    def __init__(self, data_path: str = "training_data.json", tokenizer=None, max_length: int = 256):
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
        # Use simple format: Query -> Output
        # This trains the model to produce the output in response to the query
        prompt = f"""User Query: {query}

Function Calls (CSV format):
{output}"""
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
        """Add labels for language modeling (same as input_ids for CLM)"""
        example["labels"] = example["input_ids"].copy()
        return example


def create_datasets(tokenizer, max_length: int = 256):
    """Create train and validation datasets"""
    dataset_loader = FunctionGemmaDataset(
        data_path="training_data.json",
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_dataset = dataset_loader.create_hf_dataset("train")
    val_dataset = dataset_loader.create_hf_dataset("validation")

    print(f"\n✓ Train dataset size: {len(train_dataset)}")
    print(f"✓ Val dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_finetuning():
    """Setup and run finetuning"""
    print("=" * 70)
    print("FunctionGemma CSV Format Finetuning")
    print("=" * 70)

    # Check if CUDA is available
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

    # Create datasets
    train_dataset, val_dataset = create_datasets(tokenizer, max_length=256)

    # Training arguments
    output_dir = Path("./finetuned_functiongemma")
    output_dir.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42,
        report_to=[],  # Disable wandb/tensorboard for simplicity
    )

    print(f"\n✓ Output directory: {output_dir}")
    print(f"✓ Training for {training_args.num_train_epochs} epochs")
    print(f"✓ Batch size: {training_args.per_device_train_batch_size}")
    print(f"✓ Learning rate: {training_args.learning_rate}")

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
    print("Starting finetuning...")
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
    print("\n" + "=" * 70)
    print("Final Training Metrics:")
    print("=" * 70)
    if hasattr(trainer.state, "best_metric"):
        print(f"Best eval loss: {trainer.state.best_metric:.4f}")
    print(f"Training loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")


if __name__ == "__main__":
    setup_finetuning()
