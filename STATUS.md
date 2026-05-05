# TRIAD-PTQ Exynos session status

Wall clock:  ~2h15m  (start 2026-05-05 01:30 IDT, this report 2026-05-05 03:45 IDT)
Outcome:     **partial**  (Phase 0–4 complete; Phase 5 device bench blocked
             on the deferred Phase-1 MLC Android runtime install)

## Headline numbers

|                                   | value      | acceptance       | status |
|---                                |---         |---               |---     |
| WikiText-2 PPL, FP16 (M1, CPU)    | 10.882     | (anchor)         |        |
| WikiText-2 PPL, TRIAD-INT4 (M1)   | 11.477     | ≤ FP16 + 1.0     | **PASS** (+0.595) |
| Decode tok/s on Xclipse 950       | (not measured) | ≥ 25         | **N/M** — needs MLC runtime install |
| Peak GPU MB during decode         | (not measured) | ≤ 1200 MB     | **N/M** — needs MLC runtime install |
| MLC q4f16_1 bundle size (disk)    | 778 MB     | (informational)  |        |
| Calibration peak MPS (driver hwm) | 12.19 GB   | "no OOM on M1"   | **PASS** (was OOM before) |

PPL eval window is 4088 tokens / seq=512 / CPU-fp32 dequantised forward.
The session prompt's reference figure of 8.45 PPL FP16 is from a larger
window (likely full WT2 test, ~280k tokens). We anchor against the
matched-window figure 10.882 for the +1.0 acceptance because the absolute
PPL on a small window is ~25% above full-corpus PPL on TinyLlama and the
delta is what the acceptance criterion is about.

## Completed

- **Phase 0 — Device inventory.**  Galaxy Z Flip7 / Exynos 2500 (s5e9955)
  identified. GPU is **Xclipse 950 (AMD RDNA-based, "sgpu" kernel
  driver)**, not Mali. Vulkan ICD `vulkan.samsung.so` exposes
  `VK_AMD_*` + `VK_SEC_amigo_profiling`. Full probe in
  `experiments/device_profile.txt`.  GPU mismatch with the prompt
  recorded in `docs/decisions/002-exynos-2500-xclipse-gpu.md`.

- **Phase 1 — DEFERRED.**  Reason in
  `docs/decisions/003-phase1-mlc-build-deferred.md`: building MLC-LLM
  Android runtime from source needs ~10–15 GB free disk and is a
  multi-hour open-ended toolchain task; combined with the GPU mismatch
  this would only produce a single-device reference number, not the
  Mali-G715/G725 throughput band the prompt expects. The published
  `MLCChat-release.apk` is a one-line `adb install` for the user.

- **Phase 2 — Streaming compile_model + Cholesky-OOM fix.**  The
  `dict-of-everything` retention pattern in `triad_ptq/compile.py`
  (per-layer A, U, W_prime, kappa all alive simultaneously, ~18 GB at
  TinyLlama scale) was replaced by a strict per-layer streaming loop
  with `a_device='cpu'` so Gram matrices live on host RAM. Adds
  `compute_kappa_topk` in `triad_ptq/core/router.py` so global super-
  weight selection is also a streaming pass. `gptq_solver` drops H/L
  references eagerly (3 → 2 (n,n) tensors at peak).
  - SmolLM-135M smoke: 21.506 PPL (was 21.521; within run noise).
  - TinyLlama-1.1B: full TRIAD INT4 calibration completes
    (155 layers in 1295 s on M1 16 GB unified, was OOM before).
  - 18 tests pass (15 baseline + 3 streaming-memory).

- **Phase 3 — TinyLlama calibration on M1.**  ~26 min wall clock.
  `experiments/13_tinyllama_phase3.py` saves `state_dict` to
  `/tmp/triad-tinyllama-int4/model.pt` (4.5 GB) immediately after
  compile, before any eval that could swap-thrash. PPL eval on CPU.
  Output `results/triad_tinyllama_int4_m1.json`.

- **Phase 4 — MLC q4f16_1 export.**  `triad_ptq/export/mlc.py` folds the
  activation-side rotation `U / Lam_pow_beta` and the stored sparse
  super-weight residuals into the dense weight, re-quantises at
  `group_size=32`, packs INT4 codes into `uint32` lanes (8 codes per
  lane), interleaves per-group fp16 (scale, zero) pairs.
  `experiments/14_export_mlc.py` runs the loader (re-attaches
  TriadLinear modules into a fresh HF skeleton) and emits the bundle:
  - `/tmp/triad-tinyllama-int4-mlc/mlc-chat-config.json`
  - `/tmp/triad-tinyllama-int4-mlc/ndarray-cache.json`  (356 records)
  - `/tmp/triad-tinyllama-int4-mlc/params_shard_0.bin`  (778 MB)
  - tokenizer files copied from the HF snapshot
  - `compile.sh` next to the bundle for the manual final-mile step
  - 3 export tests pass.

- **Phase 5 (partial) — comparison.md generator + FP16 baseline.**
  `experiments/15_exynos_compare.py` reads the four result JSONs and
  writes `results/exynos_comparison.md`. The TRIAD row's PPL cell is
  populated from the M1 number; throughput / GPU-MB cells stay em-
  dashed with a STATUS pointer.

## Blocked at

- **Phase 5 device bench.**  Needs the MLC Android runtime APK plus a
  built/compiled `.so` for Xclipse 950 — both deferred to manual user
  steps per ADR-003. The bundle ready to ship is at
  `/tmp/triad-tinyllama-int4-mlc/`. Manual procedure when you wake up:

  ```bash
  # 1. Install the published MLCChat APK (one click)
  #    https://llm.mlc.ai/docs/deploy/android.html (Get MLC Chat App)

  # 2. Free disk + clone MLC for the compile step
  #    (~10 GB free needed; ~/.cache/huggingface is currently 16 GB; consider
  #    moving SmolVLM2 (8.4 GB) elsewhere first)
  git clone https://github.com/mlc-ai/mlc-llm /tmp/mlc-llm
  cd /tmp/mlc-llm && git submodule update --init --recursive
  pip install mlc-llm-nightly  # or build from source per android/README.md

  # 3. Compile (per the compile.sh emitted by Phase 4)
  bash /tmp/triad-tinyllama-int4-mlc/compile.sh

  # 4. Push + bench
  adb push /tmp/triad-tinyllama-int4-mlc /sdcard/Download/triad-tinyllama
  adb shell run-as ai.mlc.mlcchat ./mlc_chat_bench \
    --model /sdcard/Download/triad-tinyllama \
    --prompt-len 64 --gen-len 64 --batch 1 --warmup 4 --runs 10

  # 5. Re-run Phase 5 to refresh the table
  uv run python experiments/15_exynos_compare.py
  ```

- **Phase 6 (NPU INT4 path).**  N/A on this device. The plan invokes a
  `exynos_nn_compile` userland binary which is not present on the stock
  Galaxy Z Flip7 retail build (Android 16, non-rooted). Documented in
  ADR-002.

## Recommended first action when you wake up

1. Open `results/exynos_comparison.md` — the PPL acceptance column
   already shows PASS (+0.595 ≤ +1.0). The two on-device rows are
   stubs.

2. Free ~10 GB on the laptop (the largest item is
   `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct`
   at 8.4 GB; relocate or delete if you don't need it for other work).

3. Run the four shell commands in "Blocked at" above. Step 3
   (`compile.sh`) is the high-risk one; if it breaks the most likely
   fix is checking the MLC q4f16_1 layout against
   `mlc_llm/quantization/group_quantization.py::quantize_weight_int4`
   in the version you installed and adjusting the
   `_pack_int4_uint32_lanes` / `_interleave_scale_zero_q4f16_1`
   helpers in `triad_ptq/export/mlc.py`. Both helpers are 5-line
   functions with explicit unit tests, so deviations are local.

## Final numbers

```
Method                            Bits  WT2-PPL  Tok/s  Peak-GPU-MB  Disk-MB
FP16 (CPU, matched window)         16    10.882    —       —            —
MLC q4f16_1 (community baseline)    4      —       —       —            —
TRIAD-INT4 (this work, M1 quality)  4    11.477    —       —          778

PPL delta TRIAD vs FP16:           +0.595        (acceptance ≤ +1.0)  PASS
```

## Branches

| Branch                          | State | Last commit |
|---                              |---    |---          |
| `feat/exynos-baseline`          | local | `95e0226` Phase 0 + ADR-003 |
| `feat/exynos-cholesky-fix`      | local | `6bbe816` Phase 2 + smoke results |
| `feat/exynos-calib-tinyllama`   | local | `573fe33` Phase 3 calib + checkpoint |
| `feat/exynos-mlc-export`        | local | `95acc07` Phase 4 MLC q4f16_1 |
| `feat/exynos-bench`             | local | `8375826` Phase 5 partial |

Each branch is a strict superset of the previous one (linear history).
**No branches were pushed to origin** because the four success criteria
require Phases 0–5 fully complete with all three on-device acceptances
met; we have one of three (PPL) and the other two are mechanically
unmeasurable until the manual MLC Android runtime install. This matches
the autonomous-mode rule: "If any of the above does not hold: do not
push. Leave everything local."

## Tests

```
uv run pytest -q
.....................   21 passed
```

## ADRs added this session

- `docs/decisions/002-exynos-2500-xclipse-gpu.md` — GPU is Xclipse 950
  not Mali; Vulkan path preserved, group_size pinned to 32 for MLC
  q4f16_1, Phase 6 deferred.
- `docs/decisions/003-phase1-mlc-build-deferred.md` — host disk +
  multi-hour toolchain risk + GPU mismatch make autonomous Phase 1
  build a poor bet; the published MLCChat APK is a one-line manual
  step.
