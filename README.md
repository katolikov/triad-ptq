# TRIAD-PTQ

**Weight-only post-training quantization for edge LLMs**, byte-compatible
with MLC-LLM `q4f16_1` so no inference kernels change.

> **Status:** v1.0.0 production · **v2.0.0-alpha (SPECTRA-Q)** on
> `v2-spectra` (155 unit tests pass on M1).
> v2's measured model-and-device PPL / decode-tps come from a separate
> runbook on a CUDA host + Galaxy Z Flip7
> ([ADR-017](docs/decisions/017-h2-h4-hardware-deferred.md)).
> The session-3 N=3 numbers below are the current published headline
> until the runbook overwrites them; see "Limits" for what they actually
> support.

## What it does

```python
from triad_ptq import optimize

optimize(model, bits=4, calibration=calib, algorithm='v2',
         group_size=64, rotation='sign_perm')
```

Output is a packed weight that loads as MLC `q4f16_1`. The compiled
OpenCL device code object is **bit-identical** to the community
baseline — verified by `tools/verify_kernel_identity.sh`. Zero kernel
changes at inference.

## v1 → v2 in one table

| Component                    | v1                                  | v2 (SPECTRA-Q, alpha)                                            |
|------------------------------|-------------------------------------|------------------------------------------------------------------|
| Sensitivity router           | KFAC + noise-injection probe        | **Squisher Fisher diagonal** ([2507.18807](https://arxiv.org/abs/2507.18807)) |
| Bit allocator                | Trace watershed (collapses to uniform) | Retired — Squisher feeds α scheduling                         |
| Rotation                     | Full-d R1 Hadamard                  | **Block-diagonal sign+perm** at G ([2511.04214](https://arxiv.org/abs/2511.04214)) |
| Smoothing exponent β         | Closed-form β\* (eq. 5)             | **Learnable per-block β**, 100 Adam BRECQ steps                  |
| Super-weights                | FP16 sparse residual                | **Channel-grained INT8** (top-1.5 % output channels, single bundle) |
| GPTAQ α                      | Fixed 0.5                           | **ρ-weighted** α = min(0.8, σ(c · log ρ))                        |
| LWC                          | –                                   | **Selective** OmniQuant LWC on top-25 % most sensitive blocks    |
| Bench protocol               | N=3                                 | **N=10 + paired-t** ([ADR-014](docs/decisions/014-bench-protocol-n10.md)) |

The only **original** v2 contributions: ρ-weighted α scheduling and the
channel-INT8 packing format. Everything else is engineering of published
methods (QuaRot, GPTAQ, AWQ, BRECQ, OmniQuant, Squisher, Yu et al.). Full
matrix in [`docs/v2-design.md`](docs/v2-design.md).

## Hardware

| Role             | Component                                                               |
|------------------|-------------------------------------------------------------------------|
| Calibration      | Apple M1 Pro 16 GB (v1; v2 on small models)                              |
| Calibration      | RTX 4090 24 GB (v2 runbook for ≥ TinyLlama-1.1B)                         |
| Inference target | Galaxy Z Flip7 / Exynos 2500 / Xclipse 950 (RDNA wave64, no `VK_KHR_cooperative_matrix`, 32 KiB LDS) |
| Inference runtime | MLC-LLM `q4f16_1`, OpenCL via `libSGPUOpenCL.so`                       |

Probe artefacts: [`docs/probe/SUMMARY.md`](docs/probe/SUMMARY.md).

## Install

```bash
uv sync --no-dev
uv add --dev pytest pytest-xdist tabulate
```

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from triad_ptq import optimize
from triad_ptq.eval.calib import build_wikitext_calib

tok   = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM-135M", torch_dtype=torch.float32
).to("mps").eval()

calib = build_wikitext_calib(tok, n_samples=32, seq_len=1024,
                             device=torch.device("mps"))

# v1 (production, M1-native):
optimize(model, bits=4, calibration=calib, group_size=64)

# v2 (alpha; runs end-to-end on a CUDA host):
# optimize(model, bits=4, calibration=calib, algorithm='v2',
#          group_size=64, rotation='sign_perm')
```

## Reproducing

```bash
make test         # 155 pass + 5 skip on M1
make smoke        # SmolLM-135M end-to-end (~5 min)
make sweep_llm    # SmolLM-135M, SmolLM-360M, TinyLlama-1.1B
make sweep_cnn    # MobileNetV2, EfficientNet-B0, MobileViT-S on ImageNetV2 (5K)
```

## Results — v1, WikiText-2 PPL on M1

| Model           | Method   | Bits | PPL ↓     | Calib s |
|-----------------|----------|------|-----------|---------|
| SmolLM-135M     | FP32     |  32  | 18.87     |   0     |
| SmolLM-135M     | RTN      |   4  | 26.60     |   1     |
| SmolLM-135M     | AWQ-like |   4  | 23.85     |  26     |
| **SmolLM-135M** | **TRIAD**|   4  | **21.56** | 213     |
| SmolLM-360M     | FP32     |  32  | 14.07     |   0     |
| SmolLM-360M     | RTN      |   4  | 17.29     |   4     |
| SmolLM-360M     | AWQ-like |   4  | 16.60     |  54     |
| **SmolLM-360M** | **TRIAD**|   4  | **15.79** | 843     |

TinyLlama-1.1B v1 OOM'd on M1 at the GPTQ Cholesky step; v2 + the
4090 host fixes that (Phase A6 + ADR-017).

### Phase-2 GPTAQ asymmetric calibration (v1)

| Variant                                                  | PPL    | Δ vs baseline       |
|----------------------------------------------------------|--------|---------------------|
| TRIAD-INT4 baseline                                      | 21.149 | reference           |
| GPTAQ asym (full transfer)                               | 25.218 | +4.069 (regression) |
| GPTAQ asym (scope-limit, α=1.0)                          | 22.033 | +0.884              |
| **GPTAQ asym (scope-limit, α=0.5) — DEFAULT**            | **20.627** | **−0.523 ✓**    |

Source: [`results/tables/smollm135_gptaq_smoke.json`](results/tables/smollm135_gptaq_smoke.json),
[ADR-010](docs/decisions/010-gptaq-phase-2.md). v2 replaces the fixed
α=0.5 with the ρ-weighted schedule.

### v2 alpha — synthetic-fixture measurements

These are **unit-test artefacts archived during the v2 build**, not
model-on-device claims. Cited per the working rule "every numerical
claim must cite an experiment file":

| Phase | Acceptance gate                                       | Measured                            | File |
|-------|-------------------------------------------------------|-------------------------------------|------|
| B     | Squisher↔Hutchinson Pearson r ≥ 0.7                   | mean **0.813**, min 0.751 (5 seeds) | [phase_b_squisher_correlation.json](results/v2/phase_b_squisher_correlation.json) |
| C     | Forward cosine ≥ 0.99999 after rotation+fold          | min **0.99999952** (3 seeds × 2 kinds) | [phase_c_rotation_forward_equivalence.json](results/v2/phase_c_rotation_forward_equivalence.json) |
| D     | BRECQ loss decreases over 100 Adam steps              | mean **−6.3 %** (5 seeds)            | [phase_d_learnable_beta.json](results/v2/phase_d_learnable_beta.json) |
| E     | INT8 sub-MSE < INT4 sub-MSE in same bundle            | ratio **≈ 290×** (4 fixtures, G=64)  | [phase_e_channel_int8.json](results/v2/phase_e_channel_int8.json) |
| F     | α(ρ=1) = 0.5 exactly (matches v1 default)             | 0.500000000000                      | [phase_f_gptaq_rho_alpha.json](results/v2/phase_f_gptaq_rho_alpha.json) |
| G     | Static disk-MB ratio G64/G32 ≤ 0.92 (H4 #5)           | **≈ 0.945** (3 fixtures) — gate fails on disk-only basis; on-device measurement pending ([ADR-015](docs/decisions/015-group-size-default.md)) | [phase_g_groupsize_sweep_static.json](results/v2/phase_g_groupsize_sweep_static.json) |

The v2 model-and-device file
[`results/v2/v2_full_sweep.json`](results/v2/v2_full_sweep.json) is a
**partially populated** — the 2026-05-06 N=10 device bench below is
recorded; v2 calibration + baselines + reference comparison still need
the runbook.

## On-device bench — v2.0.0-alpha N=10 (2026-05-06, Z Flip7)

Real measurement on the unfolded Galaxy Z Flip7 / Xclipse 950, TRIAD-INT4
v1 bundle (the only bundle currently staged on the device). Driven by
`tools/bench_android.sh 1 10 1 60` (N=10 + paired-t protocol per
[ADR-014](docs/decisions/014-bench-protocol-n10.md)). Source:
[`results/v2/device_bench/2026-05-06_z_flip7_triad_int4_summary.json`](results/v2/device_bench/2026-05-06_z_flip7_triad_int4_summary.json).

|                                              | tok/s              | N             |
|----------------------------------------------|--------------------|---------------|
| Decode mean (mixed completion lengths)       | **31.40 ± 4.80**   | 6 (of 10)     |
| Decode mean — short completions (<200 tok)   | **35.36**          | 2             |
| Decode mean — long completions (≥500 tok)    | **29.43**          | 4             |
| Prefill mean                                 | 15.00 ± 0.39       | 6             |
| Session-3 single-prompt headline (superseded)| 35.56 (N=3)        | 3             |

**Finding**: decode tok/s drops from ~36 (cold-cache, short generation)
to ~24–25 (sustained, 600+ tokens) as thermal throttling engages. The
session-3 N=3 value sampled only the short regime. The honest published
claim is now:

> **TRIAD-INT4 sustained decode on Xclipse 950: 31.4 ± 4.8 tok/s**
> (short ≈ 35 tok/s, long ≈ 29 tok/s).

The session-3 "+2.7 % vs ref" claim is **dropped** because (a) at N=3 it
wasn't statistically distinguishable from zero, and (b) the reference
q4f16_1 bundle is not currently staged on this device — a paired-t
comparison requires the runbook to push it first.

## What v2 does NOT claim

* No algorithmic decode speedup beyond G=32 → G=64 bandwidth savings
  ([ADR-015](docs/decisions/015-group-size-default.md), provisional).
* No paired-t comparison vs the q4f16_1 community reference (ref
  bundle not staged on the device; runbook owns the staging).
* No v2-vs-v1 model-on-device comparison — v2 calibration of TinyLlama-
  1.1B requires the 4090 host.
* No PPL number for v2 on any real model — see deferred items in
  [`results/v2/v2_full_sweep.json`](results/v2/v2_full_sweep.json).

## Limits & caveats

* `torch.linalg.eigh` is not implemented on MPS (PyTorch 2.11). v1
  dispatches to CPU via `triad_ptq.utils.device.safe_eigh`; this
  dominates v1 calibration time at ~5–10 s per d=2048 layer.
* TinyLlama-1.1B v1 OOM'd on M1 at the GPTQ Cholesky-inverse step;
  resolved by `safe_cholesky_inverse` (Phase A6) which falls back to
  CPU/fp64 for d ≥ 4096 on MPS. v2 uses this automatically.
* `autoawq` inference is CUDA-only — its quantize step runs on M1 but
  the produced checkpoint requires CUDA. v2 invokes it from
  `experiments/baselines/run_autoawq.py` on the 4090 host
  (`exit_status: skipped:no_cuda` on M1).
* The v0.3.0-session3 device bench was N=3 — superseded by v2's N=10
  protocol ([ADR-014](docs/decisions/014-bench-protocol-n10.md)).
  When the v2 N=10 bench runs, the result lands in
  `results/v2/device_bench/`.

## ADRs

| #   | Subject                                                                  |
|-----|--------------------------------------------------------------------------|
| 001 | `truncated_eigh` rejected (rank deficiency)                              |
| 002 | Vulkan-on-Android isn't an MLC preset; OpenCL on Xclipse 950             |
| 003 | MLC source build deferred; nightly wheels                                |
| 004 | TRIAD-folded HF-safetensors → `mlc_llm convert_weight`                   |
| 005 | Custom MLCChat APK with static system_lib                                |
| 006 | Phase-5 12 % decode gap was N=1 noise; N≥3 mandatory                     |
| 007 | `clip_search` lowers eval PPL but breaks on-device generation            |
| 008 | Rebase chain to `main` for `v0.2.0-alpha`                                |
| 009 | Phase-0.4 baseline reproduction deferred                                 |
| 010 | Phase-2 GPTAQ asymmetric calibration (scope-limit + α=0.5 = PPL win)     |
| 011 | Phase-1 q4f16_0 export option                                             |
| 012 | Phase-4 offline R1 Hadamard pre-rotation                                  |
| 013 | On-device bench runner via patched MLCChat + JSON-over-logcat            |
| 014 | **v2:** N=3 → N=10 + paired-t                                            |
| 015 | **v2:** group-size default gated on Mali measurement (provisional)       |
| 016 | **v2:** closed-form β\* init deferred — different basis                  |
| 017 | **v2:** full eval matrix (H2–H4) deferred to runbook                     |

## Repository layout

```
triad_ptq/
  core/        v1: calibration, allocator, router, grid, gptq_solver, rotate
  baselines/   rtn.py, awq.py (M1 reimpl)
  eval/        ppl.py, calib.py, vision.py, generate.py
  utils/       device.py (safe_eigh, safe_cholesky_inverse)
  _v2/
    rotation/sign_perm.py
    router/{squisher,hutchinson_check}.py
    transform/learnable_beta.py
    lwc/selective.py
    superweight/channel_int8.py
    calib/gptaq_rho_alpha.py
    groupsize/sweep.py
    pipeline.py
experiments/
  baselines/   run_{autoawq,gptq,gptaq,quarot_offline,omniquant_lwc,hqq}.py
docs/
  decisions/   ADR-001..017
  v2-design.md  SPECTRA-Q canonical design
  probe/        Xclipse 950 Vulkan + OpenCL probe JSONs
results/
  tables/       v1 sweep JSONs
  v2/           v2 synthetic-fixture results + v2_full_sweep.json placeholder
  device_bench/ N=3 v0.3.0-session3 + N=10 v2 (when produced)
tests/v2/      Phase A..G unit tests + integration smoke
tools/
  bench_android.sh         on-device runner (N=10 + paired-t)
  verify_kernel_identity.sh OpenCL device-code md5 check
```

## Citation

```bibtex
@misc{katolikov2026triad,
  author       = {Artem Katolikov},
  title        = {TRIAD-PTQ: Trace--Router--Interaction-Aware Decomposition for
                  Post-Training Quantization of Edge-Class Neural Networks},
  year         = {2026},
  howpublished = {Preprint, May 2026},
  note         = {DOI: pending}
}

@misc{katolikov2026spectraq,
  author       = {Artem Katolikov},
  title        = {{SPECTRA-Q}: Squisher Fisher routing, learnable per-block beta,
                  and channel-INT8 super-weights for byte-compatible MLC q4f16\_1
                  deployment on edge GPUs},
  year         = {2026},
  howpublished = {Preprint, v2.0.0-alpha branch \texttt{v2-spectra}, May 2026},
  note         = {Companion to TRIAD-PTQ v1.0.0; DOI: pending}
}
```

## License

Apache 2.0 — see `LICENSE`.
