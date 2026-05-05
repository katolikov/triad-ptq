"""Real `autoawq` baseline (NOT the M1-native AWQ-like reimplementation).

Runs the official `autoawq` quantize step on each requested model, then
exports to MLC q4f16_1 via `mlc_llm convert_weight + compile`. WT2 PPL
is evaluated on the post-export checkpoint.

Notes
-----
* `autoawq` is CUDA-only at inference (its `awq_inference_engine`
  extension does not build on M1). On the M1 dev host this script is a
  no-op that writes `exit_status: skipped:no_cuda`.
* On the 4090 calibration host, the script will:
    1. Quantize on CUDA via `autoawq`'s `AutoAWQForCausalLM`.
    2. Save the AWQ checkpoint.
    3. Convert to MLC q4f16_1.
    4. Run WT2 PPL eval.
* Compare against `triad_ptq.baselines.awq.awq_like_quantize`, which is
  the M1-native faithful reimplementation that v1 ships. v2 stops using
  the M1 reimplementation as a primary baseline (per the design doc).
"""
from __future__ import annotations

import sys
import time
import traceback

from experiments.baselines._common import cuda_or_skip, parse_models_argv, write_result

METHOD = "autoawq"


def main() -> int:
    if not cuda_or_skip(METHOD):
        return 0

    try:
        from awq import AutoAWQForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        for mid in parse_models_argv():
            write_result(method=METHOD, model_id=mid, exit_status=f"skipped:missing_dep:{exc.name}")
        return 0

    model_ids = parse_models_argv()
    for model_id in model_ids:
        t0 = time.time()
        try:
            quant_path = f"/tmp/autoawq__{model_id.replace('/', '_')}"
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)
            quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
            model.quantize(tok, quant_config=quant_config)
            model.save_quantized(quant_path)
            tok.save_pretrained(quant_path)
            calib = time.time() - t0
            # PPL eval and MLC export are deferred to a follow-up patch:
            # Phase A only requires the runner exists and can dispatch.
            write_result(
                method=METHOD,
                model_id=model_id,
                exit_status="quantized:awaiting_eval",
                calib_seconds=calib,
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
