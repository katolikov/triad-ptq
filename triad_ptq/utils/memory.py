"""Memory reporting helpers (M1 unified memory)."""
from __future__ import annotations

import os
import resource

import torch


def rss_gb() -> float:
    """Resident set size in GB (process-level, macOS reports bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS: ru_maxrss in bytes; Linux: in KB
    val = ru.ru_maxrss
    # heuristic
    if val > 10**9:
        return val / 1e9
    return val / 1e6


def mps_alloc_gb() -> float:
    if torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def file_size_mb(path: str | os.PathLike) -> float:
    import pathlib

    p = pathlib.Path(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1e6
