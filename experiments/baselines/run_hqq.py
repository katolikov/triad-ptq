"""HQQ baseline (Half-Quadratic Quantization, Mobius Labs).

Reference: https://github.com/mobiusml/hqq. HQQ is a fast no-data PTQ
that shows up in many recent W4 benchmarks; we include it for parity with
the literature.

Phase A only requires this runner exist; full evaluation lands in Phase H.
"""
from __future__ import annotations

import sys
import time
import traceback

from experiments.baselines._common import cuda_or_skip, parse_models_argv, write_result

METHOD = "hqq"


def main() -> int:
    if not cuda_or_skip(METHOD):
        return 0
    try:
        from hqq.core.quantize import HQQLinear  # noqa: F401  # type: ignore
    except ImportError:
        for mid in parse_models_argv():
            write_result(
                method=METHOD,
                model_id=mid,
                exit_status="skipped:hqq_not_installed",
                extra={"hint": "pip install hqq"},
            )
        return 0

    for model_id in parse_models_argv():
        t0 = time.time()
        try:
            quant_path = f"/tmp/hqq__{model_id.replace('/', '_')}"
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
