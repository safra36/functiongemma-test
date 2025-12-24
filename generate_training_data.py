#!/usr/bin/env python3
"""
Generate synthetic training data for FunctionGemma finetuning.
Creates (user_query, expected_output) pairs where output is in CSV format: function_name
"""

import json
import random
from typing import List, Tuple

# Define function metadata with example queries
FUNCTION_MAPPING = {
    "get_uptime_info": {
        "description": "Get system uptime information",
        "queries": [
            "How long has the system been running?",
            "What is the system uptime?",
            "When was the system started?",
            "How much uptime does the system have?",
            "Check the system uptime",
            "How long is the system up?",
            "Show me uptime info",
            "What's the uptime?",
        ]
    },
    "get_cpu_info": {
        "description": "Get CPU information",
        "queries": [
            "What are the CPU details?",
            "How many CPU cores do I have?",
            "Check CPU information",
            "What is my CPU load?",
            "Tell me about the processor",
            "How many cores?",
            "What's the CPU status?",
            "Show CPU information",
            "Processor details?",
            "CPU cores and load?",
        ]
    },
    "get_memory_info": {
        "description": "Get RAM memory and swap usage",
        "queries": [
            "How much memory am I using?",
            "What is my RAM usage?",
            "Check memory status",
            "How much free memory?",
            "Memory and swap information",
            "Tell me about RAM",
            "What's my memory usage?",
            "Show memory info",
            "RAM available?",
            "Swap usage?",
            "Memory stats?",
            "How much memory left?",
        ]
    },
    "get_disk_info": {
        "description": "Get disk storage information",
        "queries": [
            "How much disk space is available?",
            "Check disk usage",
            "Show storage information",
            "What is my disk space?",
            "How full is the disk?",
            "Disk storage details?",
            "Storage capacity?",
            "Free disk space?",
            "Disk utilization?",
        ]
    },
    "get_network_info": {
        "description": "Get network and IP information",
        "queries": [
            "What is my IP address?",
            "Show network information",
            "What is the hostname?",
            "Network details?",
            "Tell me about network",
            "Check network status",
            "What's my IP?",
            "Network configuration?",
            "Hostname information?",
            "IP address details?",
        ]
    },
    "get_system_info": {
        "description": "Get kernel version and OS information",
        "queries": [
            "What OS am I running?",
            "What is the kernel version?",
            "Show system information",
            "What Linux version?",
            "System details?",
            "OS information?",
            "Kernel version?",
            "Distribution info?",
            "What's the OS?",
        ]
    },
    "get_process_info": {
        "description": "Get running process information",
        "queries": [
            "What processes are running?",
            "Show top processes",
            "What is using the most CPU?",
            "What is using the most memory?",
            "Process information?",
            "Running processes?",
            "Top memory processes?",
            "Top CPU processes?",
            "List processes?",
        ]
    },
    "get_user_info": {
        "description": "Get user information",
        "queries": [
            "Who is the current user?",
            "What users are logged in?",
            "Show user information",
            "User details?",
            "Currently logged in users?",
            "Who's on the system?",
            "User list?",
            "Current user?",
        ]
    },
}


def generate_training_examples() -> List[Tuple[str, str]]:
    """Generate training examples: (user_query, expected_output_format)"""
    examples = []

    for func_name, func_data in FUNCTION_MAPPING.items():
        for query in func_data["queries"]:
            # Output format: function_name (CSV format with no arguments)
            output = func_name
            examples.append((query, output))

    return examples


def generate_multi_function_examples() -> List[Tuple[str, str]]:
    """Generate examples with queries that might trigger multiple functions"""
    multi_examples = []

    multi_queries = [
        ("Show me full system diagnostics", ["get_system_info", "get_cpu_info", "get_memory_info", "get_disk_info"]),
        ("Health check - CPU, memory, and disk", ["get_cpu_info", "get_memory_info", "get_disk_info"]),
        ("System load and processes", ["get_cpu_info", "get_process_info"]),
        ("Network and system info", ["get_network_info", "get_system_info"]),
        ("Complete system status", ["get_uptime_info", "get_cpu_info", "get_memory_info", "get_disk_info", "get_network_info"]),
        ("What's consuming resources?", ["get_process_info", "get_memory_info"]),
    ]

    for query, functions in multi_queries:
        # CSV format: function1,function2,function3...
        output = ",".join(functions)
        multi_examples.append((query, output))

    return multi_examples


def create_dataset_json(output_path: str = "training_data.json"):
    """Create and save training dataset as JSON"""
    single_examples = generate_training_examples()
    multi_examples = generate_multi_function_examples()

    all_examples = single_examples + multi_examples

    # Shuffle for better training
    random.shuffle(all_examples)

    # Split into train/validation
    split_idx = int(len(all_examples) * 0.8)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    dataset = {
        "train": [
            {"input": q, "output": o} for q, o in train_data
        ],
        "validation": [
            {"input": q, "output": o} for q, o in val_data
        ],
        "stats": {
            "total": len(all_examples),
            "train": len(train_data),
            "validation": len(val_data),
            "functions": list(FUNCTION_MAPPING.keys()),
            "num_functions": len(FUNCTION_MAPPING),
        }
    }

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✓ Training data saved to {output_path}")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Train: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"\n  Example entries:")
    for i, (q, o) in enumerate(all_examples[:5]):
        print(f"    {i+1}. Query: '{q}'")
        print(f"       Output: '{o}'")

    return dataset


if __name__ == "__main__":
    create_dataset_json("training_data.json")
