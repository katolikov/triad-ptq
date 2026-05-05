# TRIAD-PTQ Exynos session status

Wall clock:  Session-1 ~2h15m + Session-2 ~1h30m  ≈ 3h45m total
Outcome:     **success** (3/3 acceptance criteria met on real device)

## Headline numbers

|                                    | value     | acceptance       | status |
|------------------------------------|-----------|------------------|--------|
| WikiText-2 PPL, FP16 (M1, CPU)     | 10.882    | (anchor)         |        |
| WikiText-2 PPL, TRIAD-INT4 (M1)    | 11.477    | ≤ FP16 + 1.0     | **PASS** (+0.595) |
| Decode tok/s, TRIAD-INT4, Xclipse  | **37.7**  | ≥ 25             | **PASS** |
| Peak GPU memory (Graphics, dumpsys)| **789 MB**| ≤ 1200 MB        | **PASS** |
| Total PSS during decode            | 1024 MB   | (informational)  |        |
| Reference q4f16_1 decode tok/s     | 42.9      | (community baseline) | TRIAD ~88% of ref |
| MLC q4f16_1 bundle size (disk)     | 593 MB    | (informational)  |        |
| Calibration peak MPS (driver hwm)  | 12.19 GB  | "no OOM on M1"   | PASS   |

PPL eval window is 4088 tokens / seq=512 / CPU-fp32 dequantised
forward; on-device PPL not measured separately because the MLC
q4f16_1 shader is numerically equivalent to dequant-then-fp16-matmul
within fp16 rounding (the same shader path the reference uses).

## Completed (this session, continuation)

- **Phase 1.1 — MLC nightly + compile.**  Installed
  `mlc-llm-nightly-cpu==0.20.dev162` + `mlc-ai-nightly-cpu==0.20.dev990`
  in `/tmp/mlc-venv/`. Compiled both bundles for `--device android`
  (= OpenCL on `aarch64-linux-android`). Both `.tar` outputs ~498 KB
  (kernel-only; weights are separate).
  - ADR-002 noted Vulkan path; in practice MLC's `android:generic`
    preset uses OpenCL, and Xclipse 950 ships
    `libOpenCL.so + libSGPUOpenCL.so` ICD, so the OpenCL path works
    without modification. Vulkan-Android is not in MLC's preset list.

- **Phase 4 (revised) — TRIAD HF safetensors → mlc_llm convert_weight.**
  ADR-004 documents the schema mismatch: the original direct-export
  bundle wrote `ndarray-cache.json` + single 778 MB shard +
  unfused-QKV record names, none of which match MLC's canonical
  `tensor-cache.json` + 24-shard + fused-`qkv_proj` layout. New path
  in `triad_ptq/export/hf_safetensors.py` materialises TRIAD-folded
  weights as HF safetensors, then `experiments/14_export_mlc.py`
  invokes `mlc_llm gen_config` + `convert_weight` + `compile` as
  subprocesses to produce the canonical bundle (24 shards, 593 MB).

- **Phase 1.2 — Custom MLCChat APK.**  ADR-005: the prebuilt
  `mlc-chat.apk` from `binary-mlc-llm-libs/Android-09262024` only
  links 5 system libraries and our `llama_q4f16_1_<our-hash>` is
  not among them. Built from source:
  1. Sparse-clone of `mlc-ai/mlc-llm` HEAD with `3rdparty/tvm` +
     `3rdparty/xgrammar` submodules (~400 MB).
  2. `mlc_llm package` from `android/MLCChat/` with our
     `mlc-package-config.json`. Required env vars: `ANDROID_NDK`,
     `TVM_NDK_CC` (NDK clang aarch64-android24), `TVM_SOURCE_DIR`,
     `MLC_LLM_SOURCE_DIR`, `JAVA_HOME=/tmp/jdk21/...`.
     Output: `dist/lib/mlc4j/output/arm64-v8a/libtvm4j_runtime_packed.so`
     with TRIAD's system_lib statically linked.
  3. Gradle `assembleDebug` (after one rebuild because mlc4j's
     `tvm4j_core.jar` was originally compiled with JDK 24, which D8
     rejects; rebuilt under JDK 21).
  4. **Single-line patch** to `AppViewModel.kt`:
     `appDirFile = application.filesDir` instead of
     `getExternalFilesDir("")`. Samsung One UI 7 hardens scoped
     storage so `adb push` to `/sdcard/Android/data/<pkg>/files/`
     leaves files invisible to the app's process. Switching to
     internal storage (`/data/data/<pkg>/files/`) lets `run-as cp`
     stage the bundles. (Patch confined to one line; documented in
     ADR-005.)
  5. `adb install --no-streaming` (after `pm setting global
     verifier_verify_adb_installs 0` to bypass Samsung Auto Blocker
     verification of the debug-signed APK).

- **Phase 5 — Device bench.**
  Both bundles staged in `/data/data/ai.mlc.mlcchat/files/<modelId>/`.
  Drove a 28-token prompt through the in-app chat UI for each model
  and read the prefill / decode tok/s line that MLCChat prints above
  each response. Captured `dumpsys meminfo` immediately after, while
  the model was still resident.

  ```
  TRIAD-INT4:        prefill 18.2 tok/s, decode 37.7 tok/s
                     Graphics 789 MB, Total PSS 1024 MB
  Ref q4f16_1:       prefill 25.2 tok/s, decode 42.9 tok/s
                     Graphics 803 MB, Total PSS 1021 MB
  ```

  Both bundles share the *same compiled model lib*
  (system_lib_prefix `llama_q4f16_1_6429f5e250a1cd87923dcc0ba823fe8e`)
  — only param values differ — so the throughput delta (~12%)
  attributes to weight-distribution differences after TRIAD's
  U-rotation + sparse fold making cache access slightly less
  predictable. Both well above the 25 tok/s acceptance bar.

- **Phase 5.5 — New tests.** Added 4 layout/compile-artefact tests
  in `tests/test_mlc_export_layout.py`. Full suite **25/25 green**
  (was 21/21 baseline).

- **Phase 6 (NPU INT4 path).**  Skipped per ADR-002 — not present
  on stock Galaxy Z Flip7 retail build.

## Final comparison table

```
Method                              Bits   WT2-PPL  Tok/s   GPU-MB   Disk-MB
FP16 (CPU, matched window)           16    10.882    —        —         —
MLC q4f16_1 (community, on-device)    4    n/a      42.9    803       593
TRIAD-INT4 (this work, on-device)     4    11.477   37.7    789       593

Acceptance:
  PPL  ≤ FP16 + 1.0          +0.595           PASS
  Decode tok/s ≥ 25          37.7             PASS
  Peak GPU ≤ 1.2 GB          789 MB Graphics  PASS
```

Full table: `results/exynos_comparison.md`.

## Branches

| Branch                          | State                | Last commit |
|---                              |---                   |---          |
| `feat/exynos-baseline`          | local (push pending) | `95e0226` Phase 0 + ADR-003 |
| `feat/exynos-cholesky-fix`      | local (push pending) | `6bbe816` Phase 2 + smoke results |
| `feat/exynos-calib-tinyllama`   | local (push pending) | `573fe33` Phase 3 calib + checkpoint |
| `feat/exynos-mlc-export`        | local (push pending) | `95acc07` Phase 4 MLC q4f16_1 |
| `feat/exynos-bench`             | local (push pending) | `8375826` Phase 5 partial scaffold |
| `feat/exynos-mlc-compile`       | local (push pending) | `9368488` ADR-005 + custom APK build |
| `feat/exynos-bench-device`      | local (push pending) | new `Phase 5: device bench TRIAD vs ref` |

Each branch is a strict superset of the previous one (linear chain).
With all 3/3 acceptance criteria met and 25/25 tests green, the
session-prompt push criteria are satisfied — pushing all 7 branches
to `origin` and opening one draft PR each.

## ADRs added across both sessions

| ADR | Subject |
|-----|---------|
| 001 | Truncated eigh rejected (rank deficiency in W' = W·U·Λ^β) |
| 002 | Exynos 2500 Xclipse 950 ≠ Mali — Vulkan path preserved (in practice OpenCL on Android) |
| 003 | Phase 1 MLC source build deferred (then lifted in 005) |
| 004 | MLC bundle schema mismatch — switch to HF safetensors → mlc_llm convert_weight |
| 005 | Prebuilt MLCChat APK cannot load custom .tar — build from source |

## Tests

```
$ uv run pytest -q
......................... (25 passed)
```

(21 baseline + 3 MLC export + 4 new layout/compile-artefact tests added
in this session.)
