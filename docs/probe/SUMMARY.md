# Phase-0 device + driver capability probe — Xclipse 950 / Exynos 2500

Captured 2026-05-05 on Galaxy Z Flip7 (`SM-F766B`), Android 16 / One UI 7,
SoC `s5e9955`, GPU driver `Samsung Proprietary driver 24.0.533` (Vulkan),
`3.552, 35d62f642a` (OpenCL 3.0).

Probe binaries: `tools/build/vk_probe`, `tools/build/cl_probe`
(NDK 27, arm64-v8a, API 28, dlopen-based — see `tools/{vk,cl}_probe/main.cpp`).

Raw outputs:
* [`xclipse-950-vk.json`](xclipse-950-vk.json)
* [`xclipse-950-cl.json`](xclipse-950-cl.json)

Both probes ran on-device (`adb shell`) in 2 s, exit 0.

---

## Phase-0 acceptance checklist

| # | Capability                                                     | Y/N | Value |
|---|----------------------------------------------------------------|-----|-------|
| 1 | `subgroupSize == 32` (RDNA wave32 native)                      | **N** | **64 native**, but `VK_EXT_subgroup_size_control` advertises min=32, so wave32 is selectable via `RequireFullSubgroups + RequiredSubgroupSize=32`. |
| 2 | `VK_KHR_cooperative_matrix` supported                          | **N** | Not in extension list. No matrix intrinsics. |
| 3 | `integerDotProduct8BitPackedSignedAccelerated` (1.3 core)      | **N** | Driver advertises `shaderIntegerDotProduct=true` but **none** of the `4x8Bit` accelerated paths are flagged. The op is functionally available (slow path), just not HW-accelerated. |
| 4 | `storageBuffer16BitAccess`                                     | **Y** | + `uniformAndStorageBuffer16BitAccess`. `storagePushConstant16` = N. |
| 5 | `shaderFloat16`                                                | **Y** | Native fp16 ops in compute shaders. |
| 5 | `shaderInt8`                                                   | **Y** | Plus `storageBuffer8BitAccess` (= Y), `uniformAndStorageBuffer8BitAccess` (= Y). |
| 6 | `VK_EXT_subgroup_size_control`                                 | **Y** | min=32, max=64. `requiredSubgroupSizeStages = 0x20` (compute only). |
| – | `maxComputeSharedMemorySize` (LDS / shared)                    |       | 32 768 B (32 KiB) per workgroup. |
| – | `maxComputeWorkGroupInvocations`                               |       | 1024. |
| – | `maxComputeWorkGroupSize`                                      |       | [1024, 1024, 1024]. |
| – | OpenCL `CL_DEVICE_MAX_WORK_GROUP_SIZE`                         |       | 1024. |
| – | OpenCL `CL_DEVICE_LOCAL_MEM_SIZE`                              |       | 32 768 B (matches Vulkan). |
| – | OpenCL `CL_DEVICE_MAX_COMPUTE_UNITS`                           |       | 8. |
| – | OpenCL `CL_DEVICE_IMAGE_SUPPORT`                               |       | true (relevant for the image2d weight-pack trick from the callstack.com Adreno write-up — works here too because `cl_khr_image2d_from_buffer` is exported). |

## Driver / SoC summary

```
SoC:           Samsung Exynos 2500 (s5e9955)
GPU:           Samsung Xclipse 950 (vendorID 0x144d, deviceID 0x3600200)
Vulkan:        1.3.279
Driver:        Samsung Proprietary driver 24.0.533, git 627ec7ac0c
OpenCL:        3.0 (ICD: libOpenCL.so → libSGPUOpenCL.so)
OpenCL driver: 3.552, 35d62f642a
ro.hardware.vulkan = samsung      ro.hwui.use_vulkan = true
ro.gfx.driver.1    = com.samsung.pregpudriver.ex2500
```

## Notable extension landscape

* **AMD-derived heritage confirmed**: `VK_AMD_shader_core_properties`,
  `VK_AMD_shader_ballot`, `VK_AMD_gpu_shader_half_float`,
  `VK_AMD_gpu_shader_int16`, `VK_AMD_buffer_marker`,
  `VK_AMD_shader_explicit_vertex_parameter`. This matches RDNA-mobile
  ancestry (the Xclipse line started from AMD's RDNA2-IP licensing, here
  carried into the Xclipse 950 generation).
* **Subgroup intrinsics available**: `VK_KHR_shader_subgroup_extended_types`,
  `VK_KHR_shader_subgroup_rotate`, `VK_KHR_shader_subgroup_uniform_control_flow`,
  `VK_KHR_shader_quad_control`, `VK_EXT_shader_subgroup_ballot`,
  `VK_EXT_shader_subgroup_vote` — i.e. the building blocks for an FWHT
  butterfly via `subgroupShuffleXor` exist.
* **Acceleration structures + ray query**: `VK_KHR_acceleration_structure`,
  `VK_KHR_ray_tracing_pipeline`, `VK_KHR_ray_query` — irrelevant for LLM
  inference but interesting marketing surface.
* **Vulkan 1.3 compute target is a clean fit.** All the features TVM/MLC's
  Vulkan backend asks for at the LLM-inference path (`shaderFloat16`,
  `storageBuffer16BitAccess`, `subgroup_size_control`, `subgroupBroadcast`)
  are present.
* **No** `VK_KHR_cooperative_matrix`, **no** HW-accelerated 8-bit dot
  product. Implication: a future int8 matmul path would need to fall back
  to `subgroupShuffle`-based reductions, no MFMA-style gain.

## Implications for upcoming phases

* **Phase 7 (Vulkan backend)** is feasible from a capability standpoint:
  shaderFloat16 + 16-bit storage are present, subgroup size controllable
  to 32. The wave64 default is *different* from the prompt's assumption;
  any TVM dlight schedule we add for `target.kind.name == "vulkan"` should
  start at TS=64 (match native wave) before trying TS=32.
* **Phase 8 (online R4 FWHT)** is feasible: `subgroupShuffleXor` is
  available and 32 KiB LDS is enough for a size-128 FWHT (128 × fp16 =
  256 B → trivially fits, and 128 × fp32 = 512 B still fits).
* **No int8 MFMA / `cooperativeMatrix` path** ⇒ FP16 weights+activations
  remains the fastest path on-device. INT8 KV cache (Phase 3) would be
  bandwidth-only; arithmetic stays fp16.
* **Image2D path is open** (`cl_khr_image2d_from_buffer`). If the
  `q4f16_0` SoA swap (Phase 1) does not move the needle, the Adreno-style
  `MLC_LLM_OPENCL_USE_IMAGE2D` weight-as-texture trick is the next thing
  to try.
* **Subgroup-rotate is exposed** (`VK_KHR_shader_subgroup_rotate`) which
  is uncommon on mobile — TVM's RDNA dlight schedule could exploit it for
  cheap warp-shuffle reductions if we end up hand-tuning matmul.
