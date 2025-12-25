#!/usr/bin/env python3
"""
Windows System Diagnosis Tool
Uses Windows commands and Python psutil for cross-platform compatibility
"""

import subprocess
import sys
import os
import platform
from typing import Tuple, List
from datetime import datetime

# Try to import psutil for better cross-platform support
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: Install psutil for better results: pip install psutil")


def run_command(cmd: str, timeout: int = 10) -> str:
    """Execute a command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout.strip()
        if not output and result.stderr:
            return f"Error: {result.stderr.strip()}"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {str(e)}"


def get_uptime_info() -> Tuple[str, str]:
    """Get system uptime"""
    if HAS_PSUTIL:
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return ("System Uptime", f"Up {days} days, {hours} hours, {minutes} minutes (since {boot_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        return ("System Uptime", run_command("net statistics server | findstr Statistics"))


def get_cpu_info() -> List[Tuple[str, str]]:
    """Get CPU information"""
    results = []

    if HAS_PSUTIL:
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_freq = psutil.cpu_freq()

        results.append(("CPU Cores", f"Physical: {cpu_physical}, Logical: {cpu_count}"))
        results.append(("CPU Usage", f"{cpu_percent}%"))
        if cpu_freq:
            results.append(("CPU Frequency", f"Current: {cpu_freq.current:.0f} MHz, Max: {cpu_freq.max:.0f} MHz"))
    else:
        results.append(("CPU Info", run_command("wmic cpu get name,numberofcores,numberoflogicalprocessors /format:list")))

    return results


def get_memory_info() -> Tuple[str, str]:
    """Get memory usage"""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        def format_bytes(b):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if b < 1024:
                    return f"{b:.1f} {unit}"
                b /= 1024

        info = f"""RAM:
  Total: {format_bytes(mem.total)}
  Used: {format_bytes(mem.used)} ({mem.percent}%)
  Available: {format_bytes(mem.available)}

Swap:
  Total: {format_bytes(swap.total)}
  Used: {format_bytes(swap.used)} ({swap.percent}%)
  Free: {format_bytes(swap.free)}"""
        return ("Memory Usage", info)
    else:
        return ("Memory Usage", run_command("systeminfo | findstr Memory"))


def get_disk_info() -> List[Tuple[str, str]]:
    """Get disk usage"""
    results = []

    if HAS_PSUTIL:
        partitions = psutil.disk_partitions()
        for p in partitions:
            try:
                usage = psutil.disk_usage(p.mountpoint)
                def format_bytes(b):
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if b < 1024:
                            return f"{b:.1f} {unit}"
                        b /= 1024

                info = f"{p.device} ({p.mountpoint}): {format_bytes(usage.used)}/{format_bytes(usage.total)} ({usage.percent}% used)"
                results.append((f"Disk {p.device}", info))
            except:
                pass
    else:
        results.append(("Disk Info", run_command("wmic logicaldisk get size,freespace,caption")))

    return results if results else [("Disk Info", "No disk information available")]


def get_network_info() -> List[Tuple[str, str]]:
    """Get network information"""
    results = []

    # Hostname
    hostname = platform.node()
    results.append(("Hostname", hostname))

    if HAS_PSUTIL:
        # Network interfaces
        addrs = psutil.net_if_addrs()
        for iface, addr_list in addrs.items():
            for addr in addr_list:
                if addr.family.name == 'AF_INET':  # IPv4
                    results.append((f"IP ({iface})", addr.address))
                    break
    else:
        results.append(("IP Address", run_command("ipconfig | findstr IPv4")))

    return results


def get_system_info() -> List[Tuple[str, str]]:
    """Get OS/system information"""
    results = []

    results.append(("OS", f"{platform.system()} {platform.release()}"))
    results.append(("Version", platform.version()))
    results.append(("Architecture", platform.machine()))
    results.append(("Processor", platform.processor()))

    return results


def get_process_info() -> List[Tuple[str, str]]:
    """Get process information"""
    results = []

    if HAS_PSUTIL:
        processes = list(psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']))
        total = len(processes)
        results.append(("Total Processes", str(total)))

        # Top memory consumers
        top_mem = sorted(processes, key=lambda p: p.info.get('memory_percent', 0) or 0, reverse=True)[:5]
        mem_info = "\n".join([f"  {p.info['name']}: {p.info.get('memory_percent', 0):.1f}% RAM" for p in top_mem])
        results.append(("Top Memory Processes", mem_info))

        # Top CPU consumers
        top_cpu = sorted(processes, key=lambda p: p.info.get('cpu_percent', 0) or 0, reverse=True)[:5]
        cpu_info = "\n".join([f"  {p.info['name']}: {p.info.get('cpu_percent', 0):.1f}% CPU" for p in top_cpu])
        results.append(("Top CPU Processes", cpu_info))
    else:
        results.append(("Process Count", run_command("tasklist | find /c /v \"\"")))

    return results


def get_user_info() -> List[Tuple[str, str]]:
    """Get user information"""
    results = []

    # Current user
    user = os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))
    results.append(("Current User", user))

    # Logged in users
    if HAS_PSUTIL:
        users = psutil.users()
        if users:
            user_list = "\n".join([f"  {u.name} (from {u.host or 'local'}, started {datetime.fromtimestamp(u.started).strftime('%H:%M')})" for u in users])
            results.append(("Logged In Users", user_list))
        else:
            results.append(("Logged In Users", "(No other sessions)"))
    else:
        results.append(("Logged In Users", run_command("query user")))

    return results
