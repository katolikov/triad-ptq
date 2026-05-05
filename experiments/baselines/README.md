# v2 baseline runners (Phase A)

Each script in this directory wraps a published quantization baseline and
emits a JSON under `results/baselines/<method>__<model>.json`. The schema
is defined in `_common.py::write_result`.

## Status (Phase A — plumbing only)

| Method            | Script                  | Repo                                                      |
|-------------------|-------------------------|-----------------------------------------------------------|
| `autoawq`         | `run_autoawq.py`        | `pip install autoawq`                                     |
| `gptq`            | `run_gptq.py`           | `pip install auto_gptq` (or upstream IST-DASLab/gptq)     |
| `gptaq_official`  | `run_gptaq.py`          | https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ  |
| `quarot_offline`  | `run_quarot_offline.py` | in-tree (uses v1 R1 + new R2 helper)                      |
| `omniquant_lwc`   | `run_omniquant_lwc.py`  | https://github.com/OpenGVLab/OmniQuant                    |
| `hqq`             | `run_hqq.py`            | `pip install hqq`                                         |

All runners short-circuit to `exit_status: "skipped:no_cuda"` on the M1
dev host; the real numbers are collected on the 4090 calibration host
during Phase H. Phase A only gates on the runners EXISTING and being
syntactically importable. The 4090-host runbook (`docs/runbook-4090.md`,
not yet written) covers actual dispatch.

## Why not the M1-native AWQ-like reimplementation?

The v1 README compares against `triad_ptq.baselines.awq.awq_like_quantize`,
a faithful M1-native reimpl of AWQ's per-channel search. v2 stops doing
that because (a) it is not the literature baseline, and (b) it OOMed on
TinyLlama-1.1B at the 21-grid step (documented in v1 limitations). The
official `autoawq` runs on the 4090 without that issue.
