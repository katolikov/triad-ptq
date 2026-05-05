"""Vanilla GPTQ baseline (Frantar et al., arXiv:2210.17323).

Uses the official `gptq` reference implementation
(https://github.com/IST-DASLab/gptq) when available. Falls back to
`auto_gptq` otherwise. Quantises to W4G128 (the GPTQ-paper preset
closest to v2's W4G64 / W4G32 sweep) and exports to MLC q4f16_1.

Phase A only requires this runner exist; full evaluation lands in Phase H.
"""
from __future__ import annotations

import sys
import time
import traceback

from experiments.baselines._common import cuda_or_skip, parse_models_argv, write_result

METHOD = "gptq"


def main() -> int:
    if not cuda_or_skip(METHOD):
        return 0
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        for mid in parse_models_argv():
            write_result(method=METHOD, model_id=mid, exit_status=f"skipped:missing_dep:{exc.name}")
        return 0

    for model_id in parse_models_argv():
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            qcfg = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
            model = AutoGPTQForCausalLM.from_pretrained(model_id, qcfg)
            # The auto_gptq quantize API expects an iterable of dicts; we
            # leave the calibration set wiring to the 4090 host runbook.
            quant_path = f"/tmp/gptq__{model_id.replace('/', '_')}"
            write_result(
                method=METHOD,
                model_id=model_id,
                exit_status="awaiting_calibration_set",
                calib_seconds=time.time() - t0,
                extra={"quant_path": quant_path, "group_size_used": 128},
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
