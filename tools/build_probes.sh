#!/usr/bin/env bash
# Build the Phase-0 Vulkan + OpenCL probe binaries against NDK 27, arm64-v8a,
# API 28. Static-stdlib so binaries run on any modern Android device without
# a matching libc++.

set -euo pipefail

NDK="${ANDROID_NDK:-${HOME}/Library/Android/sdk/ndk/27.0.12077973}"
TC="${NDK}/toolchains/llvm/prebuilt/darwin-x86_64"
CXX="${TC}/bin/aarch64-linux-android28-clang++"

if [[ ! -x "${CXX}" ]]; then
    echo "NDK clang++ not found at: ${CXX}" >&2
    exit 1
fi

OUT="$(cd "$(dirname "$0")" && pwd)/build"
mkdir -p "${OUT}"

# Vulkan probe — links against libvulkan from the NDK sysroot, but we dlopen()
# at runtime so we keep -ldl too.
echo "[probe] building vk_probe ..."
"${CXX}" -std=c++17 -O2 -fPIE -pie -static-libstdc++ \
    -Wno-deprecated-declarations \
    tools/vk_probe/main.cpp -ldl \
    -o "${OUT}/vk_probe"

echo "[probe] building cl_probe ..."
"${CXX}" -std=c++17 -O2 -fPIE -pie -static-libstdc++ \
    tools/cl_probe/main.cpp -ldl \
    -o "${OUT}/cl_probe"

ls -la "${OUT}"/vk_probe "${OUT}"/cl_probe
file "${OUT}"/vk_probe "${OUT}"/cl_probe
