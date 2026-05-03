"""Timing helpers — MPS-aware."""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


def mps_sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


@contextmanager
def timed(label: str = ""):
    mps_sync()
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
    mps_sync()


def time_block(fn, *args, **kwargs) -> tuple[float, object]:
    mps_sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    mps_sync()
    return time.perf_counter() - t0, out
