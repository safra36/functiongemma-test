#!/usr/bin/env python3
"""
System Diagnosis Agent - Using Fine-tuned FunctionGemma
Two-Pass architecture with fine-tuned Pass 2 model
"""

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torch
import re
from typing import Dict, Any, List, Tuple
from peft import PeftModel

# Use Windows-compatible functions
from app_windows import (
    get_uptime_info, get_cpu_info, get_memory_info, get_disk_info,
    get_network_info, get_system_info, get_process_info, get_user_info,
)

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# ============================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================
FUNCTIONS = {
    "get_uptime_info": get_uptime_info,
    "get_cpu_info": get_cpu_info,
    "get_memory_info": get_memory_info,
    "get_disk_info": get_disk_info,
    "get_network_info": get_network_info,
    "get_system_info": get_system_info,
    "get_process_info": get_process_info,
    "get_user_info": get_user_info,
}

FUNCTION_DESCRIPTIONS = {
    "get_uptime_info": "Get system uptime",
    "get_cpu_info": "Get CPU info and load",
    "get_memory_info": "Get RAM/swap usage",
    "get_disk_info": "Get disk usage",
    "get_network_info": "Get network/IP info",
    "get_system_info": "Get OS/kernel info",
    "get_process_info": "Get running processes",
    "get_user_info": "Get user info",
}


# ============================================================
# CONSOLE FUNCTION - Prints to user
# ============================================================
def console(message: str):
    """Print message to user"""
    print(f"\n{'='*70}")
    print(f"AI: {message}")
    print(f"{'='*70}\n")


# ============================================================
# AGENT WITH FINE-TUNED MODEL
# ============================================================
class FineTunedAgent:
    def __init__(self):
        print("Loading FunctionGemma (Pass 1)...")
        self.processor = AutoProcessor.from_pretrained("google/functiongemma-270m-it")
        self.model_pass1 = AutoModelForCausalLM.from_pretrained(
            "google/functiongemma-270m-it",
            dtype=torch.float32,
            device_map=DEVICE,
            attn_implementation="eager",
        )
        print("✓ Pass 1 model loaded")

        print("\nLoading Fine-tuned FunctionGemma (Pass 2)...")
        # Load base model
        self.model_pass2_base = AutoModelForCausalLM.from_pretrained(
            "google/functiongemma-270m-it",
            dtype=torch.float32,
            device_map=DEVICE,
            attn_implementation="eager",
        )

        # Load LoRA adapters
        try:
            self.model_pass2 = PeftModel.from_pretrained(
                self.model_pass2_base,
                "finetuned_functiongemma_lora/final_model"
            )
            print("✓ Fine-tuned model (LoRA) loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load fine-tuned model: {e}")
            print("Falling back to base model...")
            self.model_pass2 = self.model_pass2_base

        print("\n✓ Both models ready!\n")

    def build_tools_pass1(self) -> List[Dict]:
        """Tools for Pass 1: Only diagnostic functions"""
        tools = []
        for name, desc in FUNCTION_DESCRIPTIONS.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            })
        return tools

    def build_tools_pass2(self) -> List[Dict]:
        """Tools for Pass 2: Only console function"""
        return [{
            "type": "function",
            "function": {
                "name": "console",
                "description": "Output your analysis to the user. Call this with your response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Your analysis/response message",
                        }
                    },
                    "required": ["message"],
                },
            },
        }]

    def call_model(self, model, messages: List[Dict], tools: List[Dict]) -> str:
        """Call model and return response"""
        inputs = self.processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=self.processor.eos_token_id,
                max_new_tokens=256,
            )

        response = self.processor.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        return response

    def parse_function_call(self, response: str) -> Tuple[str, Dict]:
        """Parse function name and params from response"""
        patterns = [
            r'call:(\w+)(\{.*?\})?',
            r'call\s+(\w+)(\{.*?\})?',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                func_name = match.group(1)
                params_str = match.group(2) or "{}"

                # Handle FunctionGemma's <escape> format
                escape_match = re.search(r'message:<escape>(.*?)<escape>', params_str, re.DOTALL)
                if escape_match:
                    return func_name, {"message": escape_match.group(1).strip()}

                # Handle JSON format
                json_match = re.search(r'"message"\s*:\s*"([^"]*)"', params_str)
                if json_match:
                    return func_name, {"message": json_match.group(1)}

                try:
                    params = eval(params_str)
                except:
                    params = {}
                return func_name, params

        return None, {}

    def execute_function(self, func_name: str) -> str:
        """Execute diagnostic function"""
        if func_name not in FUNCTIONS:
            return f"Error: Unknown function {func_name}"

        try:
            result = FUNCTIONS[func_name]()
            if isinstance(result, tuple):
                return f"{result[0]}:\n{result[1]}"
            elif isinstance(result, list):
                output = ""
                for item in result:
                    if isinstance(item, tuple):
                        output += f"{item[0]}:\n{item[1]}\n\n"
                return output
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def process(self, user_input: str):
        """
        Two-pass processing:
        Pass 1: User input → FunctionGemma → pick function
        Pass 2: Result → Fine-tuned FunctionGemma → console() with natural response
        """

        # ========== PASS 1: Pick function ==========
        print("🔄 Pass 1: Selecting function...")

        messages_pass1 = [
            {
                "role": "developer",
                "content": "You are a system diagnosis assistant. Based on the user query, call the appropriate diagnostic function."
            },
            {"role": "user", "content": user_input}
        ]

        response1 = self.call_model(self.model_pass1, messages_pass1, self.build_tools_pass1())
        func_name, _ = self.parse_function_call(response1)

        if not func_name:
            console("I couldn't determine which function to call.")
            return

        print(f"📞 Calling: {func_name}")

        # Execute the function
        result = self.execute_function(func_name)
        print(f"📊 Got result ({len(result)} chars)")

        # ========== PASS 2: Analyze with fine-tuned model ==========
        print("🔄 Pass 2: Generating natural response (fine-tuned)...")

        result_truncated = result[:800] if len(result) > 800 else result

        messages_pass2 = [
            {
                "role": "developer",
                "content": f"""You are a helpful system assistant.
Summarize this system data for the user.

DATA:
{result_truncated}

Call console() with a helpful summary of the above data. Include the actual numbers and values from the data."""
            },
            {"role": "user", "content": f"Summarize this data: {result_truncated[:200]}"}
        ]

        response2 = self.call_model(self.model_pass2, messages_pass2, self.build_tools_pass2())
        func_name2, params = self.parse_function_call(response2)

        if func_name2 == "console" and "message" in params:
            console(params["message"])
        else:
            # Fallback
            console(f"Here's the system data:\n{result_truncated[:500]}")

    def run(self):
        """Interactive loop"""
        print("=" * 70)
        print("System Diagnosis Agent (Fine-tuned)")
        print("=" * 70)
        print("Type 'exit' to quit\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    break
                self.process(user_input)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    agent = FineTunedAgent()
    agent.run()
