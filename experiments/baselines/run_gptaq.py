"""GPTAQ baseline (Chen et al., arXiv:2504.02692).

Uses the official GPTAQ repo
(https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ) — NOT the
v1 in-tree port (`triad_ptq/core/gptaq_asym.py`), which exists for
Phase-2 ablation of the in-pipeline transfer and is not the standalone
quantizer.

Phase A only requires this runner exist; full evaluation lands in Phase H.
"""
from __future__ import annotations

import sys
import time
import traceback

from experiments.baselines._common import cuda_or_skip, parse_models_argv, write_result

METHOD = "gptaq_official"


def main() -> int:
    if not cuda_or_skip(METHOD):
        return 0
    try:
        # The official repo exposes a CLI `python main.py --model ... --bits 4`;
        # we shell out to it from the 4090 host runbook. Importing here just to
        # confirm the package is on PYTHONPATH (clone of the upstream repo).
        import gptaq  # noqa: F401  # type: ignore
    except ImportError:
        for mid in parse_models_argv():
            write_result(
                method=METHOD,
                model_id=mid,
                exit_status="skipped:gptaq_repo_not_on_pythonpath",
                extra={"hint": "git clone https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ"},
            )
        return 0

    for model_id in parse_models_argv():
        t0 = time.time()
        try:
            quant_path = f"/tmp/gptaq__{model_id.replace('/', '_')}"
            write_result(
                method=METHOD,
                model_id=model_id,
                exit_status="awaiting_runbook_dispatch",
                calib_seconds=time.time() - t0,
                extra={"quant_path": quant_path},
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
