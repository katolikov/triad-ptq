"""Phase E — Channel-grained INT8 mixed precision for super-weights.

Replaces v1's FP16 sparse super-weights (which would require a sparse FP16
kernel at inference). Top-1.5% of OUTPUT channels by κ_j get stored as
INT8 instead of INT4, in the SAME packed tensor with a per-output-channel
1-bit indicator.

v2.0 release uses Option A (no kernel change): super-channels become a
small FP16 sub-tensor with the same per-group scales; the runtime treats
them as a separate small FP16 GEMV. <0.5% decode-time overhead.

Optional add-on: per-weight FP16 override for true outliers
(Yu et al. arXiv:2411.07191) — single FP16 scalar in lm_head metadata.

Status
------
NOT IMPLEMENTED — Phase E placeholder.
"""
from __future__ import annotations

IMPLEMENTED = False


def select_super_channels(*args, **kwargs):
    raise NotImplementedError(
        "Phase E (channel-INT8 super-weights) is not yet implemented; "
        "v2 currently delegates to v1's FP16 super-weight residual path."
    )
