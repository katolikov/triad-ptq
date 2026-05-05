"""QuaRot R1+R2 OFFLINE baseline (no online R3/R4).

We strip QuaRot down to just the offline weight rotations (R1 = residual-
stream Hadamard; R2 = head-dim block-Hadamard absorbed into Wv/Wo). The
online Hadamard layers (R3 in attention, R4 in MLP intermediate) are
DELIBERATELY NOT applied because they require an inference-time kernel
that v2 does not ship (Xclipse 950 has no `VK_KHR_cooperative_matrix`,
so a 32 KiB-LDS Hadamard kernel is suboptimal).

Reference: Ashkboos et al., arXiv:2404.00456.

Phase A only requires this runner exist; full evaluation lands in Phase H.
"""
from __future__ import annotations

import sys
import time
import traceback

from experiments.baselines._common import cuda_or_skip, parse_models_argv, write_result

METHOD = "quarot_offline"


def main() -> int:
    if not cuda_or_skip(METHOD):
        return 0

    for model_id in parse_models_argv():
        t0 = time.time()
        try:
            # Reuses v1's `triad_ptq/core/rotate.py::apply_r1_to_llama` for R1;
            # R2 is a separate utility added in this same Phase-A patch series
            # (head-dim Hadamard absorbed into v_proj.weight along the head
            # axis). The full integration is the 4090 runbook's job.
            quant_path = f"/tmp/quarot__{model_id.replace('/', '_')}"
            write_result(
                method=METHOD,
                model_id=model_id,
                exit_status="awaiting_r2_implementation",
                calib_seconds=time.time() - t0,
                extra={
                    "quant_path": quant_path,
                    "note": "R3/R4 deliberately NOT applied (no online Hadamard kernel)",
                },
            )
        except Exception as exc:
            write_result(
                method=METHOD,
                model_id=model_id,
                exit_status=f"error:{type(exc).__name__}:{exc}",
                extra={"trace": traceback.format_exc()},
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
