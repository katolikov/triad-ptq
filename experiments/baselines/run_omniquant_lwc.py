"""OmniQuant LWC-only baseline (no LET).

Reference: Shao et al., arXiv:2308.13137. v2's selective LWC (Phase D)
is structurally similar to OmniQuant's LWC component, but applied only
to the top-25 % most sensitive blocks. This runner provides the
"all-blocks LWC, no LET" comparison point so we can show that selective
LWC matches all-blocks LWC at lower calibration cost.

Phase A only requires this runner exist; full evaluation lands in Phase H.
"""
from __future__ import annotations

import sys
import time
import traceback

from experiments.baselines._common import cuda_or_skip, parse_models_argv, write_result

METHOD = "omniquant_lwc"


def main() -> int:
    if not cuda_or_skip(METHOD):
        return 0
    try:
        # The OmniQuant repo (https://github.com/OpenGVLab/OmniQuant) does not
        # ship as a pip package. The 4090 host runbook clones it under
        # /opt/repos/OmniQuant and adds it to PYTHONPATH.
        import omniquant  # noqa: F401  # type: ignore
    except ImportError:
        for mid in parse_models_argv():
            write_result(
                method=METHOD,
                model_id=mid,
                exit_status="skipped:omniquant_repo_not_on_pythonpath",
                extra={"hint": "git clone https://github.com/OpenGVLab/OmniQuant"},
            )
        return 0

    for model_id in parse_models_argv():
        t0 = time.time()
        try:
            quant_path = f"/tmp/omniquant__{model_id.replace('/', '_')}"
            write_result(
                method=METHOD,
                model_id=model_id,
                exit_status="awaiting_runbook_dispatch",
                calib_seconds=time.time() - t0,
                extra={"quant_path": quant_path, "lwc_only": True, "let_disabled": True},
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
