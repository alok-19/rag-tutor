import re

def expand_query(query: str) -> str:
    """Expand common acronyms dynamically depending on active keywords in the user query."""
    expanded = query
    query_lower = query.lower()
    
    abbreviations = {
        "pcb": "Process Control Block",
        "sjf": "Shortest Job First",
        "fcfs": "First Come First Served",
        "tlb": "Translation Lookaside Buffer",
        "lru": "Least Recently Used",
        "fifo": "First In First Out",
        "ipc": "Interprocess Communication",
        "srtf": "Shortest Remaining Time First",
        "dma": "Direct Memory Access",
        "dns": "Domain Name System",
        "tcp": "Transmission Control Protocol",
        "udp": "User Datagram Protocol",
        "sql": "Structured Query Language",
        "acid": "Atomicity Consistency Isolation Durability",
    }
    
    added_expansions = []
    for abbr, full in abbreviations.items():
        if re.search(rf"\b{abbr}\b", query_lower):
            if full.lower() not in query_lower:
                added_expansions.append(f"{abbr} ({full})")
                
    if added_expansions:
        expanded = f"{query} - " + ", ".join(added_expansions)
        
    return expanded
