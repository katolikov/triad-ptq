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
    # MPS does not support float64; move to CPU first, then upcast.
    A = mat.detach().cpu().double()
    A = A + eps * torch.eye(A.size(-1), dtype=A.dtype)
    L, U = torch.linalg.eigh(A)
    return L.to(dev, dtype=torch.float32), U.to(dev, dtype=torch.float32)


def safe_cholesky_inverse(
    H: torch.Tensor,
    *,
    percdamp: float = 0.01,
    upper: bool = True,
    cpu_threshold_dim: int | None = 4096,
) -> torch.Tensor:
    """Numerically-safe Cholesky-based inverse of a symmetric PSD matrix.

    Used by GPTQ-style solvers (`triad_ptq/core/gptq_solver.py`) where the
    per-layer Hessian H = X^T X must be inverted. On MPS, large d × d
    cholesky_inverse can OOM (TinyLlama-1.1B 2048×2048 hit ~20 GiB on M1
    8 GB; documented in README "Limitations"). The 4090 calibration host
    has plenty of memory but `cholesky_inverse` is occasionally numerically
    unhappy on CUDA fp16/bf16 inputs — promoting to fp32 on CPU when the
    GPU path fails gives a deterministic fallback.

    Behaviour
    ---------
    1. Add a `percdamp · mean(diag(H))` ridge to the diagonal (standard
       GPTQ damping).
    2. Try `torch.linalg.cholesky(...).cholesky_inverse(upper=upper)` on
       the input device.
    3. On RuntimeError (OOM, non-PSD, NaN) or when ``H.size(-1) >=
       cpu_threshold_dim`` *and* the input device is MPS, retry on CPU in
       float64, returning to the original device + dtype.

    The CPU fallback is **off the inference path** — it runs once per
    layer during calibration. The extra GPU↔CPU transfer is negligible
    compared to the eigh dispatch already required by v1's TRIAD basis.

    Returns
    -------
    H_inv : torch.Tensor, same shape, device, and dtype as ``H``.
    """
    if H.ndim != 2 or H.size(0) != H.size(1):
        raise ValueError(f"safe_cholesky_inverse: expected square 2-D, got {tuple(H.shape)}")

    dev_in = H.device
    dtype_in = H.dtype
    d = H.size(-1)

    def _ridge(A: torch.Tensor) -> torch.Tensor:
        ridge = percdamp * torch.diagonal(A).mean()
        return A + ridge * torch.eye(A.size(-1), dtype=A.dtype, device=A.device)

    # Optional fast path on the input device (skip if dim is large + MPS).
    skip_gpu = (
        cpu_threshold_dim is not None
        and d >= cpu_threshold_dim
        and dev_in.type == "mps"
    )
    if not skip_gpu:
        try:
            A = _ridge(H)
            L = torch.linalg.cholesky(A, upper=upper)
            H_inv = torch.cholesky_inverse(L, upper=upper)
            if torch.isnan(H_inv).any() or torch.isinf(H_inv).any():
                raise RuntimeError("safe_cholesky_inverse: NaN/Inf in result on input device")
            return H_inv
        except RuntimeError:
            pass  # fall through to CPU path

    # CPU fallback in float64 for numerical safety.
    A = _ridge(H.detach().to(device="cpu", dtype=torch.float64))
    L = torch.linalg.cholesky(A, upper=upper)
    H_inv = torch.cholesky_inverse(L, upper=upper)
    return H_inv.to(device=dev_in, dtype=dtype_in)
