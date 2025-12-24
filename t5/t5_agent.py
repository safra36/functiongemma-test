from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import subprocess
import os
import re
from typing import Dict, List, Tuple

# Load model once
print("Loading T5Gemma-2 model...")
processor = AutoProcessor.from_pretrained("google/t5gemma-2-270m-270m")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-270m-270m")

# Define available tools with their signatures
TOOLS_DEFINITION = """
AVAILABLE TOOLS:
1. Search(query: string) -> string
   - Purpose: Search for information
   - Returns: search results
   
2. OpenFile(path: string) -> string
   - Purpose: Read file contents
   - Returns: file contents or error
   
3. Bash(command: string) -> string
   - Purpose: Execute bash command
   - Returns: command output or error

OUTPUT FORMAT: tool_name(arg1,arg2,...)|tool_name(arg1)|result_text
Example: Search(python tutorials)|OpenFile(/etc/hosts)|Results here
"""

# Tool implementations
def search_tool(query: str) -> str:
    """Simulated search - in production use real API"""
    # For demo, return a formatted response
    return f"Search results for '{query}': Found 5 results about {query}"

def open_file_tool(path: str) -> str:
    """Read file contents"""
    try:
        with open(path, 'r') as f:
            content = f.read()
        return f"File '{path}' contents:\n{content[:500]}"  # First 500 chars
    except Exception as e:
        return f"Error reading '{path}': {str(e)}"

def bash_tool(command: str) -> str:
    """Execute bash command safely"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        output = result.stdout + result.stderr
        return f"Command '{command}' output:\n{output[:500]}"  # First 500 chars
    except subprocess.TimeoutExpired:
        return f"Command '{command}' timed out"
    except Exception as e:
        return f"Error executing '{command}': {str(e)}"

# Tool registry
TOOLS = {
    "Search": search_tool,
    "OpenFile": open_file_tool,
    "Bash": bash_tool,
}

def parse_tool_calls(text: str) -> List[Tuple[str, str]]:
    """
    Parse tool calls from model output
    Format: ToolName(arg1,arg2,...)
    """
    pattern = r'(\w+)\(([^)]*)\)'
    matches = re.findall(pattern, text)
    return matches

def execute_tool(tool_name: str, args: str) -> str:
    """Execute a tool with given arguments"""
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}"
    
    # Clean up args (remove spaces, handle multiple args)
    args_list = [arg.strip() for arg in args.split(',')]
    
    try:
        if len(args_list) == 1:
            return TOOLS[tool_name](args_list[0])
        else:
            return f"Error: {tool_name} takes exactly 1 argument, got {len(args_list)}"
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

def agent_prompt(user_request: str) -> str:
    """Create a prompt that forces tool use"""
    return f"""{TOOLS_DEFINITION}

USER REQUEST: {user_request}

You MUST respond in this format:
ToolName(argument)|ToolName(argument)|Final answer or summary

If no tools needed, explain why.
Choose tools to complete the task efficiently.
Keep arguments comma-separated and concise."""

def run_agent(user_request: str) -> str:
    """Run the agent with forced tool calling"""
    print(f"\n{'='*60}")
    print(f"USER REQUEST: {user_request}")
    print(f"{'='*60}\n")
    
    # Create prompt
    prompt = agent_prompt(user_request)
    
    # Get model response
    print("Generating response with T5Gemma-2...")
    inputs = processor(text=prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Model Response:\n{response}\n")
    
    # Parse and execute tool calls
    tool_calls = parse_tool_calls(response)
    
    if tool_calls:
        print(f"Detected {len(tool_calls)} tool call(s):")
        results = []
        for tool_name, args in tool_calls:
            print(f"  - {tool_name}({args})")
            result = execute_tool(tool_name, args)
            results.append(f"{tool_name}({args})|{result}")
        
        # Format output: comma-separated
        output = ",".join(results)
        print(f"\nTool Execution Results:")
        print(output)
        return output
    else:
        print("No tool calls detected in response")
        return response

# Example usage
if __name__ == "__main__":
    # Test requests
    test_requests = [
        "Check what files are in the current directory",
        "Get system information using bash",
        "Search for information about Persian language models",
    ]
    
    for request in test_requests:
        result = run_agent(request)
        print(f"\nFinal Output:\n{result}\n")
        print("\n" + "="*60 + "\n")