"""Device selection and op-availability helpers for M1/MPS."""
from __future__ import annotations

import os
import warnings

import torch


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warn_no_silent_fallback() -> None:
    """Warn if PyTorch will silently fall back to CPU on missing MPS ops."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        warnings.warn(
            "PYTORCH_ENABLE_MPS_FALLBACK=1 is set; some ops will silently run on CPU. "
            "TRIAD-PTQ prefers explicit CPU dispatch instead.",
            stacklevel=2,
        )


def safe_eigh(mat: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh with explicit CPU dispatch on MPS.

    PyTorch >=2.4..2.11 does not implement eigh on MPS. We move to CPU
    explicitly (not via the silent fallback) and move results back. This is
    a per-layer offline op on a small d x d matrix, not in the inference path.
    """
    dev = mat.device
    A = mat.double().detach().cpu()
    A = A + eps * torch.eye(A.size(-1), dtype=A.dtype)
    L, U = torch.linalg.eigh(A)
    return L.to(dev, dtype=torch.float32), U.to(dev, dtype=torch.float32)
