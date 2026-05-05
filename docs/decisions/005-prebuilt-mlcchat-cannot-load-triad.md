# ADR-005: Prebuilt MLCChat APK cannot load a custom .tar — build from source

Status: **Accepted** (user awake, explicit decision 2026-05-05 09:45)
Date:   2026-05-05
Branch: `feat/exynos-mlc-compile`

## Context

After Phase-1.2 attempted to install the prebuilt `MLCChat.apk` from
`mlc-ai/binary-mlc-llm-libs/releases/Android-09262024` and side-load
the TRIAD-TinyLlama-1.1B bundle to `/sdcard/Download/`, the app
launches successfully (screenshot:
`experiments/screenshots/mlc-app-launch.png`) but only its built-in
five models are listed:

| App-bundled model_lib prefix    | Source                           |
|---------------------------------|----------------------------------|
| `phi3_q4f16_0_<hash>`           | Phi-3.5-mini-instruct-q4f16_0    |
| `qwen2_q4f16_1_<hash>`          | Qwen2.5-1.5B-Instruct-q4f16_1    |
| `gemma2_q4f16_1_<hash>`         | gemma-2-2b-it-q4f16_1            |
| `llama_q4f16_0_<hash>`          | Llama-3.2-3B-Instruct-q4f16_0    |
| `mistral_q4f16_1_<hash>`        | Mistral-7B-Instruct-v0.3-q4f16_1 |

Our TRIAD-TinyLlama bundle compiled with prefix
`llama_q4f16_1_<hash>` (different `<hash>` because TinyLlama-1.1B
has a different model config than Llama-3.2-3B even though the
`llama` arch + `q4f16_1` quant match Mistral's quant.

These prefixes are **statically linked** into
`libtvm4j_runtime_packed.so` at app build time; the runtime
`model_lib_path_for_local_runtime` field in `mlc-app-config.json`
only resolves to a system_lib that is *already* compiled into the
APK. There is no dynamic `dlopen` of an arbitrary `.tar`.

Furthermore, MLC's `mlc_chat_bench` standalone Android binary
(referenced in earlier MLC docs) was removed in modern releases;
benching is now done either through (a) the on-device app's UI,
which only reaches statically-linked models, or (b) TVM RPC.

## Decision

Build a custom MLCChat APK that links the TRIAD-TinyLlama and the
reference TinyLlama (community q4f16_1) `.tar` files into the
packed runtime `.so`. Steps:

1. Sparse-clone `mlc-llm` with the `3rdparty/tvm` submodule — needed
   for the C++ runtime that `cmake` wraps into
   `libtvm4j_runtime_packed.so`.
2. Write a custom `android/MLCChat/mlc-package-config.json` listing
   the two TinyLlama bundles (TRIAD + ref) as local `model:` paths.
3. Run `mlc_llm package` from `android/MLCChat/`. This invokes
   `prepare_libs.py` (CMake + NDK) to produce the packed `.so`,
   copies the bundles into `dist/bundle/`, and writes
   `assets/mlc-app-config.json` for the new APK.
4. Run `./gradlew assembleRelease` to build the APK.
5. Sign with a fresh debug keystore (`apksigner` from
   `~/Library/Android/sdk/build-tools/36.0.0/`).
6. Install on the device.

The toolchain pieces required are:
- Android NDK 27.0.12077973 — present at
  `~/Library/Android/sdk/ndk/27.0.12077973`
- Android SDK build-tools 36.0.0 — present, includes `apksigner`,
  `zipalign`, `aapt`
- `cmake` — present (homebrew)
- `git` with submodule support for the mlc-llm `3rdparty/tvm`
  checkout (~1.5 GB; we accept the disk cost).

This is exactly the path ADR-003 deferred during the unattended
session. With the user awake (2026-05-05 09:45) and the toolchain
verified to be present, the deferral is lifted.

## Rejected alternatives

- **TVM RPC route.** Would work, but the `tvm_rpc.apk` versioning
  must match the TVM nightly we used (`mlc-ai-nightly-cpu
  0.20.dev990`); building or sourcing a matched RPC apk is itself
  the same effort as building the full app. Rejected.
- **Reuse one of the prebuilt model_lib slots by re-quantising
  TRIAD-TinyLlama to q4f16_0 and pretending it is Llama-3.2-3B.**
  Architectures differ (22 vs 28 layers; 2048 vs 3072 hidden);
  the system_lib hash bound to Llama-3.2-3B's exact config will
  refuse to load TinyLlama-1.1B params. Rejected.
- **Bench Qwen2.5-1.5B-q4f16_1 (a community model) as a hardware
  proxy.** Would tell us tok/s for the device pipeline at this
  size class but it is *not a TRIAD measurement*; the user
  explicitly forbade synthesising or extrapolating numbers. Rejected.

## Consequences

- ~1.5 GB extra disk for the `3rdparty/tvm` submodule.
- One more branch `feat/exynos-mlc-compile` carries the artefacts.
- If the source build fails, we land on the "stop, write STATUS.md,
  no push" outcome from the original prompt — no worse off than
  if we had stopped at the prebuilt-APK blocker.
