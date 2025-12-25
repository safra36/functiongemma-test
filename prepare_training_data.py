#!/usr/bin/env python3
"""
Convert training_data_responses.json into fine-tuning format
Creates both CSV and JSON formats for fine-tuning FunctionGemma
"""

import json
import csv
from pathlib import Path

# Load the training data
with open('training_data_responses.json', 'r') as f:
    training_data = json.load(f)

print(f"Loaded {len(training_data)} training examples")

# ============================================================
# Format 1: CSV (for finetune_functiongemma.py)
# ============================================================
print("\nCreating CSV format...")
csv_file = 'training_data_formatted.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(['function_name', 'user_query', 'system_data', 'expected_output'])

    # Data rows
    for item in training_data:
        writer.writerow([
            item['function_called'],
            item['user_query'],
            item['raw_data'],
            item['expected_response']
        ])

print(f"✅ Created {csv_file} with {len(training_data)} examples")

# ============================================================
# Format 2: JSONL (for modern fine-tuning)
# ============================================================
print("\nCreating JSONL format...")
jsonl_file = 'training_data_formatted.jsonl'
with open(jsonl_file, 'w', encoding='utf-8') as f:
    for item in training_data:
        # Format as prompt-completion pairs
        prompt = f"""User Query: {item['user_query']}
System Data:
{item['raw_data']}

Expected Output:"""

        entry = {
            "prompt": prompt,
            "completion": item['expected_response']
        }
        f.write(json.dumps(entry) + '\n')

print(f"✅ Created {jsonl_file} with {len(training_data)} examples")

# ============================================================
# Format 3: Text format (for training)
# ============================================================
print("\nCreating text format...")
text_file = 'training_data_formatted.txt'
with open(text_file, 'w', encoding='utf-8') as f:
    for i, item in enumerate(training_data, 1):
        f.write(f"=== Example {i} ===\n")
        f.write(f"Query: {item['user_query']}\n")
        f.write(f"Function: {item['function_called']}\n")
        f.write(f"Data:\n{item['raw_data']}\n")
        f.write(f"Response:\n{item['expected_response']}\n\n")

print(f"✅ Created {text_file}")

# ============================================================
# Format 4: JSON (for finetune_functiongemma_lora.py)
# ============================================================
print("\nCreating JSON format for LoRA...")
json_file = 'training_data_formatted.json'

# Convert to LoRA format: list of {query, output} pairs
lora_data = []
for item in training_data:
    lora_data.append({
        "query": f"{item['user_query']}\n\nSystem Data:\n{item['raw_data']}",
        "output": item['expected_response']
    })

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(lora_data, f, indent=2)

print(f"✅ Created {json_file} with {len(lora_data)} examples")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("TRAINING DATA PREPARATION COMPLETE")
print("=" * 70)
print(f"\nTotal examples: {len(training_data)}")
print("\nFiles created:")
print(f"  1. {csv_file} - For standard fine-tuning")
print(f"  2. {jsonl_file} - For modern format")
print(f"  3. {text_file} - For reference/review")
print(f"  4. {json_file} - For LoRA fine-tuning")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("\nFor standard fine-tuning:")
print("  python finetune_functiongemma.py --train-file training_data_formatted.csv")

print("\nFor LoRA fine-tuning (recommended - faster & cheaper):")
print("  python finetune_functiongemma_lora.py --train-file training_data_formatted.json")

print("\nTo expand training data:")
print("  1. Edit training_data_responses.json")
print("  2. Add more examples following the same format")
print("  3. Run this script again: python prepare_training_data.py")

print("\n" + "=" * 70)

# Print sample
print("\nSample training example:")
print("=" * 70)
sample = training_data[0]
print(f"Query: {sample['user_query']}")
print(f"Function: {sample['function_called']}")
print(f"System Data:\n{sample['raw_data']}")
print(f"\nExpected Response:\n{sample['expected_response']}")
