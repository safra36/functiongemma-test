"""
FunctionGemma-270m-it Setup Guide
This model is specifically designed for function calling (tool use)
"""

from transformers import AutoProcessor, AutoModelForCausalLM
import json

# ============================================================
# STEP 1: INSTALLATION
# ============================================================
# Run these commands in terminal:
# pip install torch
# pip install transformers
# pip install huggingface-hub

# ============================================================
# STEP 2: LOGIN TO HUGGING FACE (if needed for gated model)
# ============================================================
# huggingface-cli login
# Then paste your token from: https://huggingface.co/settings/tokens

# ============================================================
# STEP 3: LOAD MODEL
# ============================================================
print("Loading FunctionGemma-270m-it...")
processor = AutoProcessor.from_pretrained("google/functiongemma-270m-it", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it", dtype="auto", device_map="auto")
print("✓ Model loaded successfully!\n")

# ============================================================
# EXAMPLE 1: Simple Function Call - Get Weather
# ============================================================
print("="*70)
print("EXAMPLE 1: Get Temperature (Simple)")
print("="*70)

weather_function = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                },
            },
            "required": ["location"],
        },
    }
}

messages_1 = [
    {
        "role": "developer",
        "content": "You are a helpful assistant that can call functions to help the user."
    },
    {
        "role": "user", 
        "content": "What's the temperature in London?"
    }
]

inputs_1 = processor.apply_chat_template(
    messages_1, 
    tools=[weather_function], 
    add_generation_prompt=True, 
    return_dict=True, 
    return_tensors="pt"
)

out_1 = model.generate(
    **inputs_1.to(model.device), 
    pad_token_id=processor.eos_token_id, 
    max_new_tokens=128
)

output_1 = processor.decode(out_1[0][len(inputs_1["input_ids"][0]):], skip_special_tokens=True)
print(f"User: What's the temperature in London?")
print(f"Model Output:\n{output_1}\n")

# ============================================================
# EXAMPLE 2: Multiple Functions - Search & File Operations
# ============================================================
print("="*70)
print("EXAMPLE 2: Multiple Functions (Search + File Operations)")
print("="*70)

search_function = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information on the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    }
}

file_function = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "The path to the file",
                },
            },
            "required": ["filepath"],
        },
    }
}

messages_2 = [
    {
        "role": "developer",
        "content": "You are a helpful assistant with access to search and file reading tools."
    },
    {
        "role": "user", 
        "content": "Search for Python tutorials online"
    }
]

inputs_2 = processor.apply_chat_template(
    messages_2, 
    tools=[search_function, file_function], 
    add_generation_prompt=True, 
    return_dict=True, 
    return_tensors="pt"
)

out_2 = model.generate(
    **inputs_2.to(model.device), 
    pad_token_id=processor.eos_token_id, 
    max_new_tokens=128
)

output_2 = processor.decode(out_2[0][len(inputs_2["input_ids"][0]):], skip_special_tokens=True)
print(f"User: Search for Python tutorials online")
print(f"Model Output:\n{output_2}\n")

# ============================================================
# EXAMPLE 3: Parallel Functions (Multiple calls at once)
# ============================================================
print("="*70)
print("EXAMPLE 3: Parallel Function Calls")
print("="*70)

calculator_function = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The operation to perform (add, subtract, multiply, divide)",
                },
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "The numbers to operate on",
                },
            },
            "required": ["operation", "numbers"],
        },
    }
}

messages_3 = [
    {
        "role": "developer",
        "content": "You are a calculator assistant. Use the calculate function to solve math problems."
    },
    {
        "role": "user", 
        "content": "What is 10 + 5 and 20 * 3?"
    }
]

inputs_3 = processor.apply_chat_template(
    messages_3, 
    tools=[calculator_function], 
    add_generation_prompt=True, 
    return_dict=True, 
    return_tensors="pt"
)

out_3 = model.generate(
    **inputs_3.to(model.device), 
    pad_token_id=processor.eos_token_id, 
    max_new_tokens=128
)

output_3 = processor.decode(out_3[0][len(inputs_3["input_ids"][0]):], skip_special_tokens=True)
print(f"User: What is 10 + 5 and 20 * 3?")
print(f"Model Output:\n{output_3}\n")

# ============================================================
# EXAMPLE 4: Custom Tools - Agent Framework
# ============================================================
print("="*70)
print("EXAMPLE 4: Custom Persian Content Tools")
print("="*70)

persian_tools = [
    {
        "type": "function",
        "function": {
            "name": "translate_to_persian",
            "description": "Translate English text to Persian",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "English text to translate",
                    },
                },
                "required": ["text"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_persian_text",
            "description": "Process and analyze Persian text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Persian text to process",
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform: analyze, summarize, or extract",
                    },
                },
                "required": ["text", "operation"],
            },
        }
    }
]

messages_4 = [
    {
        "role": "developer",
        "content": "You are a Persian language specialist. Use translation and text processing tools."
    },
    {
        "role": "user", 
        "content": "Translate 'Hello, my name is Ali' to Persian"
    }
]

inputs_4 = processor.apply_chat_template(
    messages_4, 
    tools=persian_tools, 
    add_generation_prompt=True, 
    return_dict=True, 
    return_tensors="pt"
)

out_4 = model.generate(
    **inputs_4.to(model.device), 
    pad_token_id=processor.eos_token_id, 
    max_new_tokens=128
)

output_4 = processor.decode(out_4[0][len(inputs_4["input_ids"][0]):], skip_special_tokens=True)
print(f"User: Translate 'Hello, my name is Ali' to Persian")
print(f"Model Output:\n{output_4}\n")

# ============================================================
# EXAMPLE 5: Multi-turn Function Calling
# ============================================================
print("="*70)
print("EXAMPLE 5: Multi-turn Conversation")
print("="*70)

database_function = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Query a database for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL or natural language query",
                },
            },
            "required": ["query"],
        },
    }
}

messages_5 = [
    {
        "role": "developer",
        "content": "You are a database assistant. Use the query_database function to retrieve data."
    },
    {
        "role": "user", 
        "content": "Get all users from the database"
    },
    {
        "role": "assistant",
        "content": '<start_function_call>call:query_database{query:<escape>SELECT * FROM users<escape>}<end_function_call>'
    },
    {
        "role": "user", 
        "content": "Now filter for users from Iran"
    }
]

inputs_5 = processor.apply_chat_template(
    messages_5, 
    tools=[database_function], 
    add_generation_prompt=True, 
    return_dict=True, 
    return_tensors="pt"
)

out_5 = model.generate(
    **inputs_5.to(model.device), 
    pad_token_id=processor.eos_token_id, 
    max_new_tokens=128
)

output_5 = processor.decode(out_5[0][len(inputs_5["input_ids"][0]):], skip_special_tokens=True)
print(f"User: Now filter for users from Iran")
print(f"Model Output:\n{output_5}\n")

print("="*70)
print("✓ All examples completed!")
print("="*70)