#!/usr/bin/env python3
"""
Interactive System Diagnosis Chat using FunctionGemma
Uses the FunctionGemma model to determine which system diagnosis functions to call
based on natural language user requests.
"""

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import json
import re
from typing import Dict, Callable, Any, List
from app import (
    get_uptime_info,
    get_cpu_info,
    get_memory_info,
    get_disk_info,
    get_network_info,
    get_system_info,
    get_process_info,
    get_user_info,
)


# ============================================================
# FUNCTION DEFINITIONS FOR THE MODEL
# ============================================================
AVAILABLE_FUNCTIONS = {
    "get_uptime_info": {
        "type": "function",
        "function": {
            "name": "get_uptime_info",
            "description": "Get the system uptime information",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "callable": get_uptime_info,
    },
    "get_cpu_info": {
        "type": "function",
        "function": {
            "name": "get_cpu_info",
            "description": "Get CPU information including number of cores and load average",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "callable": get_cpu_info,
    },
    "get_memory_info": {
        "type": "function",
        "function": {
            "name": "get_memory_info",
            "description": "Get RAM memory usage, swap usage, and available memory in MB/GB. Call this for RAM, memory, or swap questions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "callable": get_memory_info,
    },
    "get_disk_info": {
        "type": "function",
        "function": {
            "name": "get_disk_info",
            "description": "Get disk usage information",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "callable": get_disk_info,
    },
    "get_network_info": {
        "type": "function",
        "function": {
            "name": "get_network_info",
            "description": "Get network information including hostname, IP addresses, and socket statistics",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "callable": get_network_info,
    },
    "get_system_info": {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get Linux kernel version and OS distribution name only. Do NOT call for RAM/memory questions - use get_memory_info instead.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "callable": get_system_info,
    },
    "get_process_info": {
        "type": "function",
        "function": {
            "name": "get_process_info",
            "description": "Get process information including top CPU and memory consuming processes",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "callable": get_process_info,
    },
    "get_user_info": {
        "type": "function",
        "function": {
            "name": "get_user_info",
            "description": "Get user information including current user and logged in users",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "callable": get_user_info,
    },
}


class SystemDiagnosisChat:
    """Interactive chat interface for system diagnosis using FunctionGemma"""

    def __init__(self):
        """Initialize the models and processors"""
        print("Loading FunctionGemma-270m-it model...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                "google/functiongemma-270m-it", device_map="auto"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/functiongemma-270m-it", dtype="auto", device_map="auto"
            )
            print("✓ FunctionGemma model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(
                "Make sure you have installed: pip install torch transformers huggingface-hub"
            )
            raise

        # Load Phi-3-mini for friendly message generation
        print("Loading Phi-3-mini-4k-instruct for friendly responses...")
        try:
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/phi-3-mini-4k-instruct",
                trust_remote_code=True
            )
            # Set pad token to avoid warnings
            if self.gemma_tokenizer.pad_token is None:
                self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token

            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-3-mini-4k-instruct",
                device_map="auto",
                trust_remote_code=True
            )
            print("✓ Phi-3-mini model loaded successfully!\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not load Phi-3-mini model: {e}")
            print("Continuing with raw output only...\n")
            self.gemma_tokenizer = None
            self.gemma_model = None

        self.conversation_history = []

    def get_tools_list(self) -> List[Dict]:
        """Get list of function definitions for the model"""
        return [
            {"type": "function", "function": f["function"]}
            for f in AVAILABLE_FUNCTIONS.values()
        ]

    def format_function_results(self, result: Any) -> str:
        """Format function results for display"""
        if isinstance(result, tuple):
            title, content = result
            return f"\n{title}:\n{content}"
        elif isinstance(result, list):
            formatted = ""
            for item in result:
                if isinstance(item, tuple):
                    title, content = item
                    formatted += f"\n{title}:\n{content}\n"
                else:
                    formatted += f"\n{item}"
            return formatted
        return str(result)

    def execute_function_call(self, func_name: str) -> str:
        """Execute a function call and return the result"""
        if func_name not in AVAILABLE_FUNCTIONS:
            return f"Error: Unknown function '{func_name}'"

        try:
            func = AVAILABLE_FUNCTIONS[func_name]["callable"]
            result = func()
            formatted_result = self.format_function_results(result)
            return formatted_result
        except Exception as e:
            return f"Error executing {func_name}: {str(e)}"

    def generate_friendly_message(self, user_query: str, raw_data: str) -> str:
        """Generate a user-friendly message using Phi-3-mini model"""
        if not self.gemma_model or not self.gemma_tokenizer:
            return f"System data retrieved for: {user_query}"

        try:
            # Format prompt for Phi-3-mini chat format
            prompt = f"""You are a helpful system administrator. Summarize the following system data in 1-2 clear, natural sentences.

System Data:
{raw_data[:400]}

Summary:"""

            inputs = self.gemma_tokenizer(prompt, return_tensors="pt")

            # Generate response with Phi-3-mini settings
            outputs = self.gemma_model.generate(
                **inputs.to(self.gemma_model.device),
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.gemma_tokenizer.eos_token_id,
                eos_token_id=self.gemma_tokenizer.eos_token_id,
            )

            # Extract only the generated part
            generated_ids = outputs[0][len(inputs["input_ids"][0]):]
            response = self.gemma_tokenizer.decode(generated_ids, skip_special_tokens=True)

            print(f"[DEBUG] Phi-3-mini raw response: {repr(response)}")

            friendly_msg = response.strip()

            print(f"[DEBUG] Extracted message: {repr(friendly_msg)} (length: {len(friendly_msg)})")

            # If response is too short or empty, return a generic message
            if not friendly_msg or len(friendly_msg) < 10:
                print(f"[DEBUG] Message too short, using fallback")
                return f"System data retrieved for: {user_query}"

            return friendly_msg

        except Exception as e:
            print(f"⚠️  Error: {e}")
            return f"System data retrieved for: {user_query}"

    def parse_function_call(self, model_output: str) -> List[str]:
        """Parse function calls from model output"""
        # Look for function call patterns in various formats
        patterns = [
            r"<start_function_call>call:(\w+)(?:\{[^}]*\})?<end_function_call>",  # XML format
            r"call:(\w+)(?:\{[^}]*\})?",  # call:function_name{} format
            r"call\s+(\w+)(?:\{[^}]*\})?",  # call function_name{} format (with space)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, model_output)
            if matches:
                print(f"[DEBUG] Parsed functions from pattern '{pattern}': {matches}")
                return matches

        print(f"[DEBUG] No function calls matched. Output was: {repr(model_output)}")
        return []

    def generate_response(self, user_message: str) -> Dict[str, Any]:
        """Generate response using FunctionGemma and execute functions"""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Prepare messages for the model
        messages = [
            {
                "role": "developer",
                "content": """You are a Linux system diagnosis assistant with access to diagnostic functions.

IMPORTANT: Use these functions based on user questions:
- "RAM", "memory", "swap" → call get_memory_info
- "OS", "kernel", "distribution", "Linux version" → call get_system_info
- "CPU", "processor", "cores" → call get_cpu_info
- "disk", "storage", "space" → call get_disk_info
- "network", "IP", "hostname" → call get_network_info
- "processes", "running apps" → call get_process_info
- "user", "logged in" → call get_user_info
- "uptime", "running time" → call get_uptime_info

Always call the MOST SPECIFIC function that matches the user's question.""",
            }
        ] + self.conversation_history

        # Get model response
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tools=self.get_tools_list(),
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            outputs = self.model.generate(
                **inputs.to(self.model.device),
                pad_token_id=self.processor.eos_token_id,
                max_new_tokens=256,
            )

            response = self.processor.decode(
                outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
            )

            # Debug: Show what FunctionGemma returned
            print(f"[DEBUG] FunctionGemma raw response: {repr(response)}")

        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "summary": f"Error generating response: {str(e)}",
                "functions_called": [],
                "results": [],
            }

        # Parse and execute function calls
        function_names = self.parse_function_call(response)
        results = []

        if function_names:
            for func_name in function_names:
                result = self.execute_function_call(func_name)
                results.append({"function": func_name, "result": result})

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})

            # Prepare raw data for friendly message generation
            raw_data = ""
            for item in results:
                raw_data += f"{item['function']}: {item['result']}\n"

            # Generate friendly message if Phi-3-mini is available
            if self.gemma_model:
                friendly_msg = self.generate_friendly_message(user_message, raw_data)
            else:
                friendly_msg = None

            # Format output with friendly message first, then technical details
            output = "\n" + "=" * 70 + "\n"

            if friendly_msg:
                output += f"💬 Assistant: {friendly_msg}\n"
                output += "-" * 70 + "\n"

            for item in results:
                output += f"📋 Function Executed: {item['function']}\n"
                output += f"📊 Result:\n{item['result']}\n"
                output += "=" * 70 + "\n"

            return {
                "response": response,
                "summary": output,
                "functions_called": function_names,
                "results": results,
            }
        else:
            # No function calls detected
            self.conversation_history.append({"role": "assistant", "content": response})
            summary = f"Model Response: {response}\n\n[Note: No function calls were detected in the response]"
            return {
                "response": response,
                "summary": summary,
                "functions_called": [],
                "results": [],
            }

    def display_available_functions(self):
        """Display available functions to the user"""
        print("\n" + "=" * 70)
        print("AVAILABLE SYSTEM DIAGNOSIS FUNCTIONS:")
        print("=" * 70)
        for name, func_info in AVAILABLE_FUNCTIONS.items():
            description = func_info["function"]["description"]
            print(f"  • {name}: {description}")
        print("=" * 70 + "\n")

    def run_interactive_chat(self):
        """Run the interactive chat loop"""
        print("\n" + "=" * 70)
        print("🔧 System Diagnosis Interactive Chat")
        print("=" * 70)
        print("Chat with the AI to ask about your system. Type 'exit' or 'quit' to stop.")
        self.display_available_functions()

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\n👋 Goodbye! Thanks for using System Diagnosis Chat.\n")
                    break

                if user_input.lower() == "help":
                    self.display_available_functions()
                    continue

                print("\n🔧 Processing...\n")
                result = self.generate_response(user_input)

                # Display the summary with function and results
                print(result["summary"])

                # Show what functions were called
                if result["functions_called"]:
                    print(f"✅ Successfully called: {', '.join(result['functions_called'])}")

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!\n")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


def main():
    """Main entry point"""
    try:
        chat = SystemDiagnosisChat()
        chat.run_interactive_chat()
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
