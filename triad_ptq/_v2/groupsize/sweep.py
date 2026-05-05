"""Phase G — Hardware-aware group-size sweep.

Runs the full v2 calibration pipeline at G ∈ {32, 64, 128} on TinyLlama-1.1B
and SmolLM-360M, then sets G = 64 as the new recommended default for
Xclipse 950 (gated on measured Mali decode tok/s ≥ G=32).

Rationale: G=64 halves FP16 scale traffic and aligns with the Xclipse 950
native subgroup size of 64 (wave64). If Mali measurement shows G=64 is
slower, the decision is reverted with measured data documented in
docs/decisions/015-group-size-default.md.

Status
------
NOT IMPLEMENTED — Phase G placeholder.
"""
from __future__ import annotations

IMPLEMENTED = False
GROUP_SIZES_TO_SWEEP = (32, 64, 128)
RECOMMENDED_DEFAULT = 64  # subject to Phase G empirical confirmation
