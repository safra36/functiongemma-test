#!/usr/bin/env python3
"""
Convert training_data_responses.json into the correct format for fine-tuning
"""

import json
import random

# Load our training data
with open('training_data_responses.json', 'r') as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} examples")

# Convert to fine-tuning format
formatted_examples = []

for item in raw_data:
    # Format input: combine query + data
    input_text = f"{item['user_query']}\n\nSystem Data:\n{item['raw_data']}"

    # Output is the expected response
    output_text = item['expected_response']

    formatted_examples.append({
        "input": input_text,
        "output": output_text
    })

# Shuffle and split into train/validation (80/20)
random.seed(42)
random.shuffle(formatted_examples)

split_point = int(len(formatted_examples) * 0.8)
train_data = formatted_examples[:split_point]
val_data = formatted_examples[split_point:]

# Create final training data structure
training_data = {
    "train": train_data,
    "validation": val_data
}

# Save
with open('training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)

print(f"\n✅ Created training_data.json")
print(f"   Train: {len(train_data)} examples")
print(f"   Validation: {len(val_data)} examples")

# Print sample
print("\n" + "="*70)
print("SAMPLE TRAINING EXAMPLE:")
print("="*70)
sample = train_data[0]
print(f"\nINPUT:\n{sample['input'][:200]}...")
print(f"\nOUTPUT:\n{sample['output'][:200]}...")
print("\n" + "="*70)
print("\nNow run: python finetune_functiongemma_lora.py")
print("="*70)
