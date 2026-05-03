"""TRIAD-PTQ: Trace-Router-Interaction-Aware Decomposition for PTQ.

Public API: :func:`optimize`.
"""

__all__ = ["optimize"]


def optimize(*args, **kwargs):  # pragma: no cover - lazy proxy
    from .api import optimize as _opt

    return _opt(*args, **kwargs)

__version__ = "0.1.0"
