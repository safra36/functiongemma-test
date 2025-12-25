#!/usr/bin/env python3
"""
Generate a massive, high-quality training dataset for FunctionGemma
~1000 examples covering all diagnostic functions with varied queries and scenarios
"""

import json
import random

# ============================================================
# QUERY VARIATIONS - Different ways users might ask
# ============================================================

MEMORY_QUERIES = [
    "How much RAM am I using?", "Tell me about my memory usage", "Memory status",
    "Check memory", "What's my RAM usage?", "Show me memory info",
    "How's my system memory?", "Memory check", "RAM usage please",
    "What's my memory situation?", "Check RAM", "Memory stats",
    "How much memory is available?", "Show RAM", "Memory information",
    "What's using my RAM?", "How full is my RAM?", "RAM status",
    "Check system memory", "Memory report", "How's the memory?",
    "RAM check", "Memory usage stats", "What about memory?",
    "Show memory usage", "How much RAM is free?", "Memory details",
    "Check available memory", "What's my memory like?", "RAM report",
]

CPU_QUERIES = [
    "How many CPU cores do I have?", "What's my CPU status?", "Tell me about CPU",
    "CPU usage", "Check CPU", "How's my processor?",
    "Show CPU info", "CPU stats", "What's my CPU load?",
    "Processor information", "CPU check", "How many cores?",
    "What CPU do I have?", "CPU status please", "Show processor info",
    "How's the CPU doing?", "Check processor", "CPU details",
    "What's my CPU usage?", "Processor status", "Show me CPU",
    "CPU report", "How loaded is my CPU?", "Processor check",
    "What about the CPU?", "CPU information", "Check my processor",
    "How's CPU performance?", "Show CPU usage", "Processor stats",
]

DISK_QUERIES = [
    "How much disk space is left?", "Show disk usage", "Disk space check",
    "Check disk space", "What's my storage like?", "Show storage",
    "How full is my disk?", "Disk status", "Storage check",
    "How much space do I have?", "Check hard drive", "Disk usage",
    "What's my disk usage?", "Storage status", "Show disk info",
    "How's my storage?", "Disk space status", "Check storage space",
    "What about disk space?", "Show me storage", "Disk report",
    "Storage information", "How much free space?", "Disk check",
    "What's my storage situation?", "Show hard drive usage", "Storage report",
    "Disk space info", "Check free space", "How's the disk?",
]

UPTIME_QUERIES = [
    "How long has my system been running?", "Show uptime", "How long is uptime?",
    "Check uptime", "System uptime", "How long since reboot?",
    "When did I last restart?", "Show system uptime", "Uptime check",
    "How long has this been up?", "System runtime", "Uptime status",
    "How long running?", "When was last boot?", "Uptime info",
    "Check system uptime", "How long since boot?", "Show me uptime",
    "System uptime check", "How long up?", "Last reboot time",
    "When did system start?", "Uptime report", "System run time",
    "How long has it been running?", "Boot time", "Show runtime",
    "How long since startup?", "System start time", "Uptime details",
]

SYSTEM_QUERIES = [
    "What operating system am I running?", "Show system info", "Tell me about my system",
    "System information", "What OS?", "Check OS version",
    "Show OS info", "What system am I on?", "OS details",
    "System details", "What's my OS?", "Operating system info",
    "Show system details", "What version of Windows?", "System specs",
    "OS information", "Check system info", "What's my system?",
    "Show operating system", "System version", "OS check",
    "What system is this?", "Show me system info", "System report",
    "OS status", "What's running?", "Computer information",
    "System configuration", "Show system specs", "What OS version?",
]

PROCESS_QUERIES = [
    "What processes are running?", "Show running processes", "Tell me about processes",
    "Process list", "What's running?", "Check processes",
    "Show me processes", "Running programs", "What apps are open?",
    "Process information", "What's using resources?", "Show tasks",
    "Task list", "What programs are running?", "Process check",
    "Show active processes", "What's running on my system?", "Running tasks",
    "Process status", "Show applications", "What's consuming resources?",
    "Check running programs", "Process report", "Active processes",
    "What tasks are active?", "Show running apps", "Process details",
    "What's using CPU?", "Running applications", "Show me what's running",
]

USER_QUERIES = [
    "Who am I logged in as?", "Show user information", "Who is logged in?",
    "Current user", "What's my username?", "Check user",
    "Who's logged on?", "User info", "What user am I?",
    "Show logged in users", "User check", "Who am I?",
    "Current logged in user", "Show users", "User status",
    "What's my user account?", "Check logged in users", "Who's using the system?",
    "User details", "Show current user", "User information",
    "Who's on the system?", "Check user sessions", "What user?",
    "Show me user info", "Active users", "User session info",
    "Who's connected?", "Current user info", "User report",
]

NETWORK_QUERIES = [
    "Get network info", "What is my IP address?", "Show network information",
    "Network status", "What's my IP?", "Check network",
    "Show IP address", "Network details", "What's my hostname?",
    "Network check", "Show network config", "IP info",
    "What's my network setup?", "Network configuration", "Show me network info",
    "IP address check", "Network information", "What's my computer name?",
    "Check IP", "Show hostname", "Network report",
    "What network am I on?", "Show network status", "Computer name",
    "Network settings", "IP status", "Show me my IP",
    "Network details please", "What's my network?", "Hostname check",
]

# ============================================================
# SCENARIO GENERATORS - Create realistic system states
# ============================================================

def generate_memory_scenarios():
    """Generate diverse memory usage scenarios"""
    scenarios = []

    # Healthy scenarios (20-60% usage)
    for _ in range(50):  # Increased from 40
        total = random.choice([8.0, 16.0, 32.0, 64.0])
        percent = random.uniform(20, 60)
        used = round(total * (percent / 100), 1)
        available = round(total - used, 1)

        swap_total = random.choice([4.0, 8.0, 16.0, 24.0])
        swap_percent = random.uniform(0, 15)
        swap_used = round(swap_total * (swap_percent / 100), 1)
        swap_free = round(swap_total - swap_used, 1)

        scenarios.append({
            "total": total, "used": used, "percent": round(percent, 1),
            "available": available, "swap_total": swap_total,
            "swap_used": swap_used, "swap_percent": round(swap_percent, 1),
            "swap_free": swap_free, "status": "healthy"
        })

    # Warning scenarios (60-85% usage)
    for _ in range(40):  # Increased from 30
        total = random.choice([8.0, 16.0, 32.0, 64.0])
        percent = random.uniform(60, 85)
        used = round(total * (percent / 100), 1)
        available = round(total - used, 1)

        swap_total = random.choice([4.0, 8.0, 16.0, 24.0])
        swap_percent = random.uniform(15, 50)
        swap_used = round(swap_total * (swap_percent / 100), 1)
        swap_free = round(swap_total - swap_used, 1)

        scenarios.append({
            "total": total, "used": used, "percent": round(percent, 1),
            "available": available, "swap_total": swap_total,
            "swap_used": swap_used, "swap_percent": round(swap_percent, 1),
            "swap_free": swap_free, "status": "warning"
        })

    # Critical scenarios (85-99% usage)
    for _ in range(40):  # Increased from 30
        total = random.choice([8.0, 16.0, 32.0])
        percent = random.uniform(85, 99)
        used = round(total * (percent / 100), 1)
        available = round(total - used, 1)

        swap_total = random.choice([4.0, 8.0, 16.0])
        swap_percent = random.uniform(50, 95)
        swap_used = round(swap_total * (swap_percent / 100), 1)
        swap_free = round(swap_total - swap_used, 1)

        scenarios.append({
            "total": total, "used": used, "percent": round(percent, 1),
            "available": available, "swap_total": swap_total,
            "swap_used": swap_used, "swap_percent": round(swap_percent, 1),
            "swap_free": swap_free, "status": "critical"
        })

    return scenarios

def generate_cpu_scenarios():
    """Generate diverse CPU scenarios"""
    scenarios = []

    cpu_configs = [
        (2, 4), (4, 8), (6, 12), (8, 16), (10, 20), (12, 24), (14, 20), (16, 32)
    ]

    # Low usage (0-30%)
    for _ in range(40):  # Increased from 30
        physical, logical = random.choice(cpu_configs)
        usage = round(random.uniform(0, 30), 1)
        base_freq = random.choice([1800, 2000, 2300, 2400, 2600, 3000])
        max_freq = base_freq + random.randint(1000, 2000)
        current_freq = random.randint(base_freq, base_freq + 500)

        scenarios.append({
            "physical": physical, "logical": logical, "usage": usage,
            "current_freq": current_freq, "max_freq": max_freq,
            "status": "idle"
        })

    # Moderate usage (30-70%)
    for _ in range(50):  # Increased from 40
        physical, logical = random.choice(cpu_configs)
        usage = round(random.uniform(30, 70), 1)
        base_freq = random.choice([1800, 2000, 2300, 2400, 2600, 3000])
        max_freq = base_freq + random.randint(1000, 2000)
        current_freq = random.randint(base_freq + 200, max_freq - 200)

        scenarios.append({
            "physical": physical, "logical": logical, "usage": usage,
            "current_freq": current_freq, "max_freq": max_freq,
            "status": "moderate"
        })

    # High usage (70-100%)
    for _ in range(40):  # Increased from 30
        physical, logical = random.choice(cpu_configs)
        usage = round(random.uniform(70, 100), 1)
        base_freq = random.choice([1800, 2000, 2300, 2400, 2600, 3000])
        max_freq = base_freq + random.randint(1000, 2000)
        current_freq = random.randint(max_freq - 500, max_freq)

        scenarios.append({
            "physical": physical, "logical": logical, "usage": usage,
            "current_freq": current_freq, "max_freq": max_freq,
            "status": "heavy"
        })

    return scenarios

def generate_disk_scenarios():
    """Generate diverse disk usage scenarios"""
    scenarios = []

    # Healthy scenarios (0-70% usage)
    for _ in range(40):
        c_total = random.choice([250, 500, 1000, 2000])
        c_percent = random.uniform(20, 70)
        c_used = round(c_total * (c_percent / 100), 1)

        d_total = random.choice([500, 1000, 2000, 4000])
        d_percent = random.uniform(10, 60)
        d_used = round(d_total * (d_percent / 100), 1)

        scenarios.append({
            "c_total": c_total, "c_used": c_used, "c_percent": round(c_percent, 1),
            "d_total": d_total, "d_used": d_used, "d_percent": round(d_percent, 1),
            "status": "healthy"
        })

    # Warning scenarios (70-90% usage)
    for _ in range(30):
        c_total = random.choice([250, 500, 1000])
        c_percent = random.uniform(70, 90)
        c_used = round(c_total * (c_percent / 100), 1)

        d_total = random.choice([500, 1000, 2000])
        d_percent = random.uniform(70, 90)
        d_used = round(d_total * (d_percent / 100), 1)

        scenarios.append({
            "c_total": c_total, "c_used": c_used, "c_percent": round(c_percent, 1),
            "d_total": d_total, "d_used": d_used, "d_percent": round(d_percent, 1),
            "status": "warning"
        })

    # Critical scenarios (90-99% usage)
    for _ in range(30):
        c_total = random.choice([250, 500])
        c_percent = random.uniform(90, 99)
        c_used = round(c_total * (c_percent / 100), 1)

        d_total = random.choice([500, 1000])
        d_percent = random.uniform(90, 99)
        d_used = round(d_total * (d_percent / 100), 1)

        scenarios.append({
            "c_total": c_total, "c_used": c_used, "c_percent": round(c_percent, 1),
            "d_total": d_total, "d_used": d_used, "d_percent": round(d_percent, 1),
            "status": "critical"
        })

    return scenarios

def generate_uptime_scenarios():
    """Generate diverse uptime scenarios"""
    scenarios = []

    # Recent boots (< 1 day)
    for _ in range(30):  # Increased from 25
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        scenarios.append({
            "days": 0, "hours": hours, "minutes": minutes,
            "status": "fresh"
        })

    # Short uptimes (1-7 days)
    for _ in range(30):  # Increased from 25
        days = random.randint(1, 7)
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        scenarios.append({
            "days": days, "hours": hours, "minutes": minutes,
            "status": "recent"
        })

    # Medium uptimes (7-30 days)
    for _ in range(30):  # Increased from 25
        days = random.randint(7, 30)
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        scenarios.append({
            "days": days, "hours": hours, "minutes": minutes,
            "status": "stable"
        })

    # Long uptimes (30+ days)
    for _ in range(30):  # Increased from 25
        days = random.randint(30, 180)
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        scenarios.append({
            "days": days, "hours": hours, "minutes": minutes,
            "status": "very_stable"
        })

    return scenarios

# ============================================================
# RESPONSE GENERATORS - Create natural language responses
# ============================================================

def generate_memory_response(scenario, query):
    """Generate natural language response for memory"""
    total = scenario["total"]
    used = scenario["used"]
    percent = scenario["percent"]
    available = scenario["available"]
    swap_used = scenario["swap_used"]
    swap_total = scenario["swap_total"]
    swap_percent = scenario["swap_percent"]
    status = scenario["status"]

    if status == "healthy":
        intros = [
            f"You have {total} GB of RAM with {used} GB in use ({percent}%). ",
            f"Your system has {total} GB of total RAM. Currently using {used} GB ({percent}%). ",
            f"RAM usage looks good at {percent}%. You're using {used} GB out of {total} GB. ",
            f"Memory is healthy. {used} GB of {total} GB RAM is in use ({percent}%). ",
        ]
        middles = [
            f"You still have {available} GB available. ",
            f"{available} GB remains free for applications. ",
            f"There's {available} GB of free memory available. ",
            f"You have {available} GB of headroom left. ",
        ]
        endings = [
            f"Swap usage is minimal at {swap_used} GB out of {swap_total} GB ({swap_percent}%). Your memory situation is healthy.",
            f"Swap is barely used ({swap_used} GB of {swap_total} GB), which is excellent.",
            f"Very low swap usage at {swap_percent}% - your system isn't under memory pressure.",
            f"Swap usage is only {swap_used} GB ({swap_percent}%), indicating good memory health.",
        ]

    elif status == "warning":
        intros = [
            f"Your RAM usage is getting high at {percent}%. ",
            f"Memory is at {percent}% capacity ({used} GB of {total} GB). ",
            f"You're using {used} GB out of {total} GB RAM ({percent}%). ",
            f"RAM usage is elevated - {percent}% of {total} GB is in use. ",
        ]
        middles = [
            f"Only {available} GB remains available. ",
            f"You have {available} GB of RAM left. ",
            f"Just {available} GB of free memory remaining. ",
            f"{available} GB is all that's left available. ",
        ]
        endings = [
            f"Swap usage is at {swap_percent}% ({swap_used} GB of {swap_total} GB). Consider closing some applications.",
            f"Your swap is being used more at {swap_used} GB ({swap_percent}%). You might want to free up some memory.",
            f"Swap usage has increased to {swap_percent}%. Closing unused programs would help.",
            f"With {swap_used} GB of swap in use ({swap_percent}%), you should consider freeing up memory.",
        ]

    else:  # critical
        intros = [
            f"WARNING: Your system is running critically low on memory! ",
            f"ALERT: RAM usage is at {percent}% - very high! ",
            f"CRITICAL: Only {available} GB of RAM available out of {total} GB! ",
            f"DANGER: Memory is nearly exhausted at {percent}% usage! ",
        ]
        middles = [
            f"You're using {used} GB out of {total} GB, leaving only {available} GB free. ",
            f"RAM: {used} GB used, {available} GB free ({percent}% full). ",
            f"Only {available} GB remains from {total} GB total. ",
            f"{used} GB in use, just {available} GB left. ",
        ]
        endings = [
            f"Swap is also heavily used at {swap_percent}% ({swap_used} GB of {swap_total} GB). Close applications immediately!",
            f"Your swap usage is critical at {swap_used} GB ({swap_percent}%). Urgent action needed - close programs now!",
            f"Swap is at {swap_percent}% - system is under severe memory pressure. Free up memory immediately!",
            f"With swap at {swap_used} GB ({swap_percent}%), your system may become unstable. Close applications now!",
        ]

    response = random.choice(intros) + random.choice(middles) + random.choice(endings)
    return response

def generate_cpu_response(scenario, query):
    """Generate natural language response for CPU"""
    physical = scenario["physical"]
    logical = scenario["logical"]
    usage = scenario["usage"]
    current = scenario["current_freq"]
    max_freq = scenario["max_freq"]
    status = scenario["status"]

    if status == "idle":
        intros = [
            f"Your CPU is running smoothly with {physical} physical cores and {logical} logical threads. ",
            f"You have a {physical}-core processor ({logical} threads total). ",
            f"CPU configuration: {physical} physical cores, {logical} logical threads. ",
            f"Processor has {physical} cores with {logical} threads available. ",
        ]
        middles = [
            f"CPU usage is very low at {usage}%, so you have plenty of processing power available. ",
            f"Currently using only {usage}% of CPU capacity - lots of headroom. ",
            f"Usage is minimal at {usage}%, leaving plenty of resources free. ",
            f"Just {usage}% CPU load - your processor is mostly idle. ",
        ]
        endings = [
            f"Running at {current} MHz (can boost to {max_freq} MHz when needed).",
            f"Current frequency: {current} MHz, max boost: {max_freq} MHz.",
            f"Clock speed is {current} MHz, with a maximum of {max_freq} MHz available.",
            f"Frequency: {current} MHz out of {max_freq} MHz max.",
        ]

    elif status == "moderate":
        intros = [
            f"Your {physical}-core CPU ({logical} threads) is moderately loaded. ",
            f"CPU has {physical} physical cores and {logical} logical threads. ",
            f"Processor: {physical} cores, {logical} threads total. ",
            f"You have {physical} cores ({logical} threads) available. ",
        ]
        middles = [
            f"Current usage is {usage}%, which is moderate. ",
            f"CPU load is at {usage}% - handling current workload well. ",
            f"Running at {usage}% capacity - reasonable load. ",
            f"Usage: {usage}% - moderate activity level. ",
        ]
        endings = [
            f"Frequency is {current} MHz out of {max_freq} MHz max. System is responsive.",
            f"Clock speed: {current} MHz (max: {max_freq} MHz). Performance is good.",
            f"Running at {current} MHz, can boost to {max_freq} MHz if needed.",
            f"Current: {current} MHz, maximum: {max_freq} MHz. Performing well.",
        ]

    else:  # heavy
        intros = [
            f"Your CPU is under heavy load! ",
            f"Processor is working hard - {usage}% usage. ",
            f"CPU is heavily loaded at {usage}%. ",
            f"High CPU usage detected: {usage}%. ",
        ]
        middles = [
            f"All {physical} cores ({logical} threads) are being utilized. ",
            f"Your {physical}-core processor ({logical} threads) is near capacity. ",
            f"{physical} cores and {logical} threads are actively processing. ",
            f"The {physical} cores ({logical} threads total) are working hard. ",
        ]
        endings = [
            f"Frequency is maxed at {current} MHz (limit: {max_freq} MHz). Check what's consuming CPU resources.",
            f"Running at {current} MHz, near the {max_freq} MHz maximum. Consider checking running processes.",
            f"Clock speed: {current} MHz out of {max_freq} MHz. You may want to identify heavy processes.",
            f"Boosted to {current} MHz (max: {max_freq} MHz). Something is consuming significant CPU.",
        ]

    response = random.choice(intros) + random.choice(middles) + random.choice(endings)
    return response

def generate_disk_response(scenario, query):
    """Generate natural language response for disk"""
    c_total = scenario["c_total"]
    c_used = scenario["c_used"]
    c_percent = scenario["c_percent"]
    d_total = scenario["d_total"]
    d_used = scenario["d_used"]
    d_percent = scenario["d_percent"]
    status = scenario["status"]

    c_free = c_total - c_used
    d_free = d_total - d_used

    if status == "healthy":
        intros = [
            f"Your disk space looks healthy. ",
            f"Storage situation is good. ",
            f"You have plenty of disk space available. ",
            f"Disk usage is at comfortable levels. ",
        ]
        c_parts = [
            f"C: drive is {c_percent}% full ({c_used} GB of {c_total} GB used, {c_free} GB free). ",
            f"C: drive has {c_free} GB free out of {c_total} GB ({c_percent}% used). ",
            f"Your C: drive is using {c_used} GB of {c_total} GB ({c_percent}%), leaving {c_free} GB available. ",
            f"C: drive: {c_used} GB used, {c_free} GB free ({c_total} GB total, {c_percent}% usage). ",
        ]
        d_parts = [
            f"D: drive is {d_percent}% full ({d_used} GB of {d_total} GB used, {d_free} GB free). ",
            f"D: drive has {d_free} GB free out of {d_total} GB ({d_percent}% used). ",
            f"Your D: drive is using {d_used} GB of {d_total} GB ({d_percent}%), leaving {d_free} GB available. ",
            f"D: drive: {d_used} GB used, {d_free} GB free ({d_total} GB total, {d_percent}% usage). ",
        ]
        endings = [
            "No storage concerns at this time.",
            "Everything looks good.",
            "Plenty of room for new files.",
            "Storage is in excellent shape.",
        ]

    elif status == "warning":
        intros = [
            f"Your disk space is getting limited. ",
            f"Storage space is running low. ",
            f"Disk usage is elevated. ",
            f"You're running low on storage. ",
        ]
        c_parts = [
            f"C: drive is {c_percent}% full with {c_free} GB remaining out of {c_total} GB. ",
            f"C: drive has only {c_free} GB left ({c_percent}% used of {c_total} GB total). ",
            f"Your C: drive is at {c_percent}% capacity - {c_used} GB used, {c_free} GB free. ",
            f"C: drive: {c_percent}% full ({c_used} GB of {c_total} GB), {c_free} GB available. ",
        ]
        d_parts = [
            f"D: drive is {d_percent}% full with {d_free} GB remaining out of {d_total} GB. ",
            f"D: drive has only {d_free} GB left ({d_percent}% used of {d_total} GB total). ",
            f"Your D: drive is at {d_percent}% capacity - {d_used} GB used, {d_free} GB free. ",
            f"D: drive: {d_percent}% full ({d_used} GB of {d_total} GB), {d_free} GB available. ",
        ]
        endings = [
            "Consider cleaning up unnecessary files soon.",
            "You should free up some space when possible.",
            "Deleting unused files would help.",
            "Time to do some cleanup or add storage.",
        ]

    else:  # critical
        intros = [
            f"CRITICAL: Your disk space is almost exhausted! ",
            f"WARNING: Storage is critically low! ",
            f"ALERT: Disk space is nearly full! ",
            f"DANGER: You're running out of disk space! ",
        ]
        c_parts = [
            f"C: drive is {c_percent}% full with only {c_free} GB remaining out of {c_total} GB! ",
            f"C: drive: {c_used} GB used out of {c_total} GB ({c_percent}% full), just {c_free} GB left! ",
            f"Your C: drive is critically full at {c_percent}% - only {c_free} GB available! ",
            f"C: drive has just {c_free} GB free ({c_percent}% of {c_total} GB used)! ",
        ]
        d_parts = [
            f"D: drive is {d_percent}% full with only {d_free} GB remaining out of {d_total} GB! ",
            f"D: drive: {d_used} GB used out of {d_total} GB ({d_percent}% full), just {d_free} GB left! ",
            f"Your D: drive is critically full at {d_percent}% - only {d_free} GB available! ",
            f"D: drive has just {d_free} GB free ({d_percent}% of {d_total} GB used)! ",
        ]
        endings = [
            "Delete files immediately or add more storage!",
            "Urgent action required - free up space now!",
            "System may malfunction if storage fills completely. Clean up urgently!",
            "Critical: Delete unnecessary files or upgrade storage immediately!",
        ]

    response = random.choice(intros) + random.choice(c_parts) + random.choice(d_parts) + random.choice(endings)
    return response

def generate_uptime_response(scenario, query):
    """Generate natural language response for uptime"""
    days = scenario["days"]
    hours = scenario["hours"]
    minutes = scenario["minutes"]
    status = scenario["status"]

    # Format uptime string
    if days == 0:
        if hours == 0:
            uptime_str = f"{minutes} minutes"
        else:
            uptime_str = f"{hours} hours and {minutes} minutes"
    elif days == 1:
        uptime_str = f"1 day, {hours} hours, and {minutes} minutes"
    else:
        uptime_str = f"{days} days, {hours} hours, and {minutes} minutes"

    if status == "fresh":
        responses = [
            f"Your system was just started {uptime_str} ago. This is a fresh boot.",
            f"System has been running for {uptime_str}. Very recent startup.",
            f"Uptime is {uptime_str}. Your computer was recently restarted.",
            f"The system booted {uptime_str} ago. Fresh session.",
            f"Your computer has been up for {uptime_str}. Just started recently.",
        ]

    elif status == "recent":
        responses = [
            f"Your system has been running for {uptime_str}. Recent boot.",
            f"Uptime: {uptime_str}. System is running well since last restart.",
            f"The system has been up for {uptime_str}. Still relatively fresh.",
            f"Running continuously for {uptime_str}. Good uptime.",
            f"System uptime is {uptime_str}. Stable since last boot.",
        ]

    elif status == "stable":
        responses = [
            f"Your system has been running continuously for {uptime_str}. Very stable!",
            f"Impressive uptime of {uptime_str}. Your system is quite stable.",
            f"The system has been up for {uptime_str} - excellent stability.",
            f"Running for {uptime_str} without restart. Great uptime!",
            f"System uptime: {uptime_str}. Very stable operation.",
        ]

    else:  # very_stable
        responses = [
            f"Exceptional uptime: {uptime_str}! Your system is extremely stable.",
            f"Your system has been running for {uptime_str} - outstanding stability!",
            f"Remarkable uptime of {uptime_str}. Excellent system stability!",
            f"The system has been up for {uptime_str} continuously. Impressive!",
            f"Amazing uptime: {uptime_str}. Your system is rock solid. Consider a reboot for updates though.",
        ]

    return random.choice(responses)

# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

def generate_dataset():
    """Generate the complete massive dataset"""
    print("Generating massive training dataset...")
    print("=" * 70)

    dataset = []

    # Generate Memory examples (~150 examples)
    print("Generating memory examples...")
    memory_scenarios = generate_memory_scenarios()
    for i, scenario in enumerate(memory_scenarios):
        query = random.choice(MEMORY_QUERIES)

        raw_data = f"""Memory Usage:
RAM:
  Total: {scenario['total']} GB
  Used: {scenario['used']} GB ({scenario['percent']}%)
  Available: {scenario['available']} GB

Swap:
  Total: {scenario['swap_total']} GB
  Used: {scenario['swap_used']} GB ({scenario['swap_percent']}%)
  Free: {scenario['swap_free']} GB"""

        response_text = generate_memory_response(scenario, query)
        expected_response = f"call:console{{message:<escape>{response_text}<escape>}}"

        dataset.append({
            "user_query": query,
            "function_called": "get_memory_info",
            "raw_data": raw_data,
            "expected_response": expected_response
        })

    print(f"  [OK] Created {len(memory_scenarios)} memory examples")

    # Generate CPU examples (~150 examples)
    print("Generating CPU examples...")
    cpu_scenarios = generate_cpu_scenarios()
    for scenario in cpu_scenarios:
        query = random.choice(CPU_QUERIES)

        raw_data = f"""CPU Cores:
Physical: {scenario['physical']}, Logical: {scenario['logical']}

CPU Usage:
{scenario['usage']}%

CPU Frequency:
Current: {scenario['current_freq']} MHz, Max: {scenario['max_freq']} MHz"""

        response_text = generate_cpu_response(scenario, query)
        expected_response = f"call:console{{message:<escape>{response_text}<escape>}}"

        dataset.append({
            "user_query": query,
            "function_called": "get_cpu_info",
            "raw_data": raw_data,
            "expected_response": expected_response
        })

    print(f"  [OK] Created {len(cpu_scenarios)} CPU examples")

    # Generate Disk examples (~150 examples)
    print("Generating disk examples...")
    disk_scenarios = generate_disk_scenarios()
    for scenario in disk_scenarios:
        query = random.choice(DISK_QUERIES)

        raw_data = f"""Disk C:\\:
C:\\ (C:\\): {scenario['c_used']} GB/{scenario['c_total']} GB ({scenario['c_percent']}% used)

Disk D:\\:
D:\\ (D:\\): {scenario['d_used']} GB/{scenario['d_total']} GB ({scenario['d_percent']}% used)"""

        response_text = generate_disk_response(scenario, query)
        expected_response = f"call:console{{message:<escape>{response_text}<escape>}}"

        dataset.append({
            "user_query": query,
            "function_called": "get_disk_info",
            "raw_data": raw_data,
            "expected_response": expected_response
        })

    print(f"  [OK] Created {len(disk_scenarios)} disk examples")

    # Generate Uptime examples (~100 examples)
    print("Generating uptime examples...")
    uptime_scenarios = generate_uptime_scenarios()
    for scenario in uptime_scenarios:
        query = random.choice(UPTIME_QUERIES)

        # Create date string (mock)
        raw_data = f"System Uptime:\nUp {scenario['days']} days, {scenario['hours']} hours, {scenario['minutes']} minutes"

        response_text = generate_uptime_response(scenario, query)
        expected_response = f"call:console{{message:<escape>{response_text}<escape>}}"

        dataset.append({
            "user_query": query,
            "function_called": "get_uptime_info",
            "raw_data": raw_data,
            "expected_response": expected_response
        })

    print(f"  [OK] Created {len(uptime_scenarios)} uptime examples")

    # Generate System Info examples (~100 examples)
    print("Generating system info examples...")
    os_versions = [
        ("Windows 10 Home", "10.0.19045", "19045"),
        ("Windows 10 Pro", "10.0.19044", "19044"),
        ("Windows 11 Home", "10.0.22631", "22631"),
        ("Windows 11 Pro", "10.0.22621", "22621"),
        ("Windows 11 Pro", "10.0.22000", "22000"),
    ]
    processors = [
        "Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz",
        "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz",
        "Intel(R) Core(TM) i5-13600K CPU @ 3.50GHz",
        "AMD Ryzen 5 5600X @ 3.70GHz",
        "AMD Ryzen 7 5800X @ 3.80GHz",
        "AMD Ryzen 9 5950X @ 3.40GHz",
        "Intel(R) Core(TM) i9-12900K CPU @ 3.20GHz",
    ]

    for _ in range(100):
        query = random.choice(SYSTEM_QUERIES)
        os_name, version, build = random.choice(os_versions)
        processor = random.choice(processors)

        raw_data = f"""OS:
{os_name}

Version:
{version} Build {build}

Architecture:
AMD64

Processor:
{processor}"""

        # Generate varied responses
        responses = [
            f"You're running {os_name} (Build {build}) on AMD64 architecture with a {processor}. This is a solid configuration for most tasks.",
            f"Your system is {os_name} version {version} (Build {build}) running on {processor}. Good setup for everyday computing.",
            f"OS: {os_name} Build {build}. Processor: {processor}. Architecture: AMD64. A capable system.",
            f"You have {os_name} (Build {build}) with {processor} on AMD64. This handles most workloads well.",
            f"System: {os_name} {version} on {processor}. Modern configuration with good performance.",
        ]

        response_text = random.choice(responses)
        expected_response = f"call:console{{message:<escape>{response_text}<escape>}}"

        dataset.append({
            "user_query": query,
            "function_called": "get_system_info",
            "raw_data": raw_data,
            "expected_response": expected_response
        })

    print(f"  [OK] Created 100 system info examples")

    # Generate Process Info examples (~150 examples)
    print("Generating process examples...")
    process_configs = [
        # (total_processes, [(process_name, ram_mb)], [(process_name, cpu_percent)])
        (85, [("explorer.exe", 280), ("firefox.exe", 1200), ("vscode.exe", 450)],
         [("firefox.exe", 22), ("vscode.exe", 8), ("system", 3)]),
        (127, [("explorer.exe", 450), ("chrome.exe", 1850), ("discord.exe", 520)],
         [("chrome.exe", 35), ("discord.exe", 12), ("vscode.exe", 7)]),
        (156, [("explorer.exe", 320), ("python.exe", 2800), ("docker-desktop.exe", 1500)],
         [("python.exe", 45), ("docker-desktop.exe", 18), ("explorer.exe", 2)]),
        (92, [("explorer.exe", 310), ("slack.exe", 850), ("spotify.exe", 420)],
         [("slack.exe", 15), ("spotify.exe", 8), ("chrome.exe", 12)]),
        (143, [("explorer.exe", 380), ("teams.exe", 980), ("outlook.exe", 650)],
         [("teams.exe", 28), ("outlook.exe", 14), ("chrome.exe", 19)]),
    ]

    for _ in range(180):  # Increased from 150
        query = random.choice(PROCESS_QUERIES)
        total, mem_procs, cpu_procs = random.choice(process_configs)

        # Add some randomization
        total = total + random.randint(-10, 10)
        mem_str = "\n".join([f"  {name}: {mb + random.randint(-50, 50)} MB" for name, mb in mem_procs])
        cpu_str = "\n".join([f"  {name}: {pct + random.uniform(-3, 3):.1f}% CPU" for name, pct in cpu_procs])

        raw_data = f"""Total Processes:
{total}

Top Memory Processes:
{mem_str}

Top CPU Processes:
{cpu_str}"""

        # Generate varied responses
        top_mem = mem_procs[0]
        top_cpu = cpu_procs[0]

        responses = [
            f"You have {total} processes running. {top_mem[0].replace('.exe', '').capitalize()} is using the most RAM at around {top_mem[1]} MB, while {top_cpu[0].replace('.exe', '')} is consuming {top_cpu[1]}% CPU.",
            f"Currently {total} processes are active. Top memory user: {top_mem[0]} ({top_mem[1]} MB). Top CPU: {top_cpu[0]} ({top_cpu[1]}%).",
            f"System has {total} running processes. {top_mem[0]} dominates RAM usage at {top_mem[1]} MB, and {top_cpu[0]} is using {top_cpu[1]}% CPU.",
            f"There are {total} processes running. {top_cpu[0]} is the heaviest on CPU at {top_cpu[1]}%, while {top_mem[0]} uses {top_mem[1]} MB of RAM.",
        ]

        response_text = random.choice(responses)
        expected_response = f"call:console{{message:<escape>{response_text}<escape>}}"

        dataset.append({
            "user_query": query,
            "function_called": "get_process_info",
            "raw_data": raw_data,
            "expected_response": expected_response
        })

    print(f"  [OK] Created 180 process examples")

    # Generate User Info examples (~100 examples)
    print("Generating user info examples...")
    usernames = ["ammob", "john.smith", "administrator", "user", "dev-user", "jane.doe", "admin"]

    for _ in range(100):
        query = random.choice(USER_QUERIES)
        username = random.choice(usernames)

        # 60% single user, 40% multiple users
        if random.random() < 0.6:
            raw_data = f"""Current User:
{username}

Logged In Users:
  (No other sessions)"""

            responses = [
                f"You are logged in as '{username}'. You're the only user currently logged into this system.",
                f"Current user: {username}. No other active sessions on this computer.",
                f"You're logged in as {username}. This is the only active user session.",
                f"Username: {username}. You're the sole user logged into the system right now.",
            ]

        else:
            # Multiple users
            other_user = random.choice([u for u in usernames if u != username])
            ip = f"192.168.1.{random.randint(100, 200)}"
            time1 = f"{random.randint(8, 16):02d}:{random.randint(0, 59):02d}"
            time2 = f"{random.randint(8, 16):02d}:{random.randint(0, 59):02d}"

            raw_data = f"""Current User:
{username}

Logged In Users:
  {username} (from local, started {time1})
  {other_user} (from {ip}, started {time2})"""

            responses = [
                f"You are logged in as '{username}' locally since {time1}. There's also '{other_user}' connected remotely from {ip} who logged in at {time2}.",
                f"Current user: {username} (local, since {time1}). Additionally, {other_user} is connected from {ip} since {time2}. Two active sessions total.",
                f"You're {username} on a local session (started {time1}). User {other_user} is also logged in remotely from {ip} (started {time2}).",
            ]

        response_text = random.choice(responses)
        expected_response = f"call:console{{message:<escape>{response_text}<escape>}}"

        dataset.append({
            "user_query": query,
            "function_called": "get_user_info",
            "raw_data": raw_data,
            "expected_response": expected_response
        })

    print(f"  [OK] Created 100 user info examples")

    # Generate Network Info examples (~100 examples)
    print("Generating network examples...")
    hostnames = [
        "DESKTOP-GAMING", "LAPTOP-USER", "WORKSTATION-PRO", "PC-HOME",
        "DEV-MACHINE", "OFFICE-PC", "HOME-DESKTOP", "WORK-LAPTOP"
    ]

    for _ in range(100):
        query = random.choice(NETWORK_QUERIES)
        hostname = random.choice(hostnames)

        # 40% single connection, 60% multiple connections
        if random.random() < 0.4:
            conn_type = random.choice(["Ethernet", "WiFi"])
            if random.random() < 0.5:
                ip = f"192.168.1.{random.randint(10, 200)}"
                network_type = "home"
            else:
                ip = f"10.0.0.{random.randint(10, 200)}"
                network_type = "corporate"

            raw_data = f"""Hostname:
{hostname}

IP ({conn_type}):
{ip}"""

            if network_type == "home":
                responses = [
                    f"Your computer hostname is '{hostname}'. IP address on {conn_type}: {ip}. You're on a local home network.",
                    f"System name: {hostname}. {conn_type} IP: {ip}. Connected to local network.",
                    f"Hostname: {hostname}. Your {conn_type} connection has IP {ip} (local network).",
                ]
            else:
                responses = [
                    f"Your computer is '{hostname}'. {conn_type} IP: {ip}. You appear to be on a corporate/institutional network.",
                    f"System: {hostname}. IP on {conn_type}: {ip} (corporate network range).",
                    f"Hostname: {hostname}. {conn_type} connection: {ip}. This looks like a company network.",
                ]

        else:
            # Dual connection
            eth_ip = f"192.168.1.{random.randint(10, 200)}"
            wifi_ip = f"192.168.1.{random.randint(10, 200)}"

            raw_data = f"""Hostname:
{hostname}

IP (Ethernet):
{eth_ip}

IP (WiFi):
{wifi_ip}"""

            responses = [
                f"Your system is named '{hostname}'. You have two active connections: Ethernet on {eth_ip} and WiFi on {wifi_ip}. Both are on the local network.",
                f"Hostname: {hostname}. Dual connections detected - Ethernet: {eth_ip}, WiFi: {wifi_ip}. Both on local network.",
                f"Computer name: {hostname}. Two network interfaces active: {eth_ip} (Ethernet) and {wifi_ip} (WiFi).",
            ]

        response_text = random.choice(responses)
        expected_response = f"call:console{{message:<escape>{response_text}<escape>}}"

        dataset.append({
            "user_query": query,
            "function_called": "get_network_info",
            "raw_data": raw_data,
            "expected_response": expected_response
        })

    print(f"  [OK] Created 100 network examples")

    print("=" * 70)
    print(f"Total examples generated: {len(dataset)}")

    return dataset

# ============================================================
# RUN GENERATOR
# ============================================================

if __name__ == "__main__":
    dataset = generate_dataset()

    # Save to file
    output_file = "training_data_responses_massive.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSUCCESS: Saved {len(dataset)} examples to {output_file}")

    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE EXAMPLES:")
    print("=" * 70)
    for i in range(3):
        sample = random.choice(dataset)
        print(f"\n[Example {i+1}]")
        print(f"Query: {sample['user_query']}")
        print(f"Function: {sample['function_called']}")
        print(f"Response: {sample['expected_response'][:150]}...")
