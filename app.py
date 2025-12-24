#!/usr/bin/env python3
"""
Simple Linux System Diagnosis Tool - Functional Programming Style
Runs various system commands and displays diagnostic information
"""

import subprocess
import sys
from typing import Tuple, List, Dict, Callable
from datetime import datetime


def run_command(cmd: str, timeout: int = 5) -> str:
    """
    Execute a bash command and return its output.
    
    Args:
        cmd: The bash command to run
        timeout: Command timeout in seconds
        
    Returns:
        Command output or error message
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {str(e)}"


def format_section(title: str, content: str) -> str:
    """Format a diagnostic section with title and content."""
    separator = "=" * 60
    return f"\n{separator}\n  {title}\n{separator}\n{content}"


def print_section(title: str, content: str) -> None:
    """Print a formatted section to stdout."""
    print(format_section(title, content))


def get_uptime_info() -> Tuple[str, str]:
    """Get system uptime information."""
    return ("System Uptime", run_command("uptime -p"))


def get_cpu_info() -> List[Tuple[str, str]]:
    """Get CPU information."""
    return [
        ("CPU Cores", run_command("nproc")),
        ("CPU Load Average", run_command("top -bn1")),
    ]


def get_memory_info() -> Tuple[str, str]:
    """Get memory usage information."""
    return ("Memory Usage", run_command("free -h"))


def get_disk_info() -> List[Tuple[str, str]]:
    """Get disk usage information."""
    return [
        ("Disk Usage", run_command("df -h")),
        ("Root Filesystem Size", run_command("du -sh / 2>/dev/null")),
    ]


def get_network_info() -> List[Tuple[str, str]]:
    """Get network information."""
    return [
        ("Hostname", run_command("hostname")),
        ("IP Addresses", run_command("ip addr show | grep 'inet '")),
        ("Socket Statistics", run_command("ss -s")),
    ]


def get_system_info() -> List[Tuple[str, str]]:
    """Get system information."""
    return [
        ("Kernel Information", run_command("uname -a")),
        ("OS Information", run_command("cat /etc/os-release | grep PRETTY_NAME")),
    ]


def get_process_info() -> List[Tuple[str, str]]:
    """Get process information."""
    return [
        ("Total Processes", run_command("ps aux | wc -l")),
        ("Top Memory Consuming Processes", run_command("ps aux --sort=-%mem")),
        ("Top CPU Consuming Processes", run_command("ps aux --sort=-%cpu")),
    ]


def get_user_info() -> List[Tuple[str, str]]:
    """Get user information."""
    return [
        ("Current User", run_command("whoami")),
        ("Logged In Users", run_command("who")),
    ]


def flatten_results(results_list: List[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in results_list for item in sublist]


def collect_diagnostics() -> List[Tuple[str, str]]:
    """Collect all diagnostic information."""
    return [get_uptime_info()] + \
           get_cpu_info() + \
           [get_memory_info()] + \
           get_disk_info() + \
           get_network_info() + \
           get_system_info() + \
           get_process_info() + \
           get_user_info()


def print_header() -> None:
    """Print report header."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔍 System Diagnosis Report - {timestamp}")


def print_footer() -> None:
    """Print report footer."""
    separator = "=" * 60
    print(f"\n{separator}\n  Report Complete\n{separator}\n")


def display_results(results: List[Tuple[str, str]]) -> None:
    """Display all diagnostic results."""
    for title, content in results:
        print_section(title, content)


def export_to_file(results: List[Tuple[str, str]], filename: str = "diagnosis_report.txt") -> None:
    """Export diagnostic results to a file."""
    try:
        with open(filename, 'w') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"System Diagnosis Report - {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            
            for title, content in results:
                f.write(f"\n{title}\n")
                f.write("-" * 40 + "\n")
                f.write(content + "\n")
        
        print(f"✅ Report exported to {filename}")
    except Exception as e:
        print(f"❌ Failed to export report: {e}", file=sys.stderr)


def validate_platform() -> bool:
    """Check if running on Linux."""
    if sys.platform not in ['linux', 'linux2']:
        print("❌ This tool requires Linux", file=sys.stderr)
        return False
    return True


def main() -> None:
    """Main entry point."""
    if not validate_platform():
        sys.exit(1)
    
    print_header()
    results = collect_diagnostics()
    display_results(results)
    print_footer()
    
    # Optional: Export to file
    # export_to_file(results, "system_diagnosis.txt")


if __name__ == "__main__":
    main()