"""Phase C — Offline block-diagonal random sign-flip + permutation rotation.

Reference: arXiv:2511.04214 ("Block Rotation is All You Need for MXFP4
Quantization"). Replaces v1's R1 Hadamard (triad_ptq/core/rotate.py).

The rotation R = Π · diag(ε) is block-diagonal with block size G (the
target group size), so it respects group-32 / group-64 alignment used by
the per-group INT4 quantizer. This means:

  * No online Hadamard at inference (zero kernel changes).
  * Group boundaries are preserved (the per-group max is invariant).
  * The fold into the preceding RMSNorm is exactly equivariant: any
    forward-cosine deviation from 1.0 indicates a fold bug.

Status
------
NOT IMPLEMENTED — Phase C placeholder. Calling `apply_sign_perm_to_llama`
raises NotImplementedError.
"""
from __future__ import annotations

IMPLEMENTED = False


def apply_sign_perm_to_llama(*args, **kwargs):
    raise NotImplementedError(
        "Phase C (sign_perm rotation) is not yet implemented; "
        "v2 currently delegates to v1's R1 Hadamard for rotation."
    )
