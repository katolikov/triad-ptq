// Triad-ML / Phase 0 — Vulkan capability probe for Xclipse 950 (Exynos 2500).
// Cross-compiled with Android NDK 27 for arm64-v8a, API 28+. Loads libvulkan
// dynamically (not linked) so we can fall back gracefully on devices that
// don't expose the loader.
//
// Output: JSON on stdout — committed verbatim to docs/probe/xclipse-950-vk.json.
//
// The fields below are the ones Phase 0 SUMMARY.md needs:
//   - subgroupSize / subgroupSupportedOperations / subgroupSizeControl
//   - VK_KHR_cooperative_matrix
//   - integerDotProduct8BitPackedSignedAccelerated (Vulkan 1.3 core)
//   - storageBuffer16BitAccess, storagePushConstant16
//   - shaderFloat16, shaderInt8
//   - maxComputeSharedMemorySize
//
// Notes:
//   * vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR is loaded by name
//     because it is an instance-level extension function and not part of
//     core Vulkan 1.3.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include <vector>
#include <string>

#define LOAD(sym) ((PFN_##sym)dlsym(libvk, #sym))

static const char* qstr(const char* s) {
    static char buf[256];
    snprintf(buf, sizeof buf, "\"%s\"", s ? s : "");
    return buf;
}

int main(int argc, char** argv) {
    void* libvk = dlopen("libvulkan.so", RTLD_NOW);
    if (!libvk) {
        fprintf(stderr, "{\"error\":\"libvulkan.so dlopen failed: %s\"}\n", dlerror());
        return 2;
    }

    auto pfn_vkCreateInstance               = LOAD(vkCreateInstance);
    auto pfn_vkEnumeratePhysicalDevices     = LOAD(vkEnumeratePhysicalDevices);
    auto pfn_vkGetPhysicalDeviceProperties  = LOAD(vkGetPhysicalDeviceProperties);
    auto pfn_vkGetPhysicalDeviceProperties2 = LOAD(vkGetPhysicalDeviceProperties2);
    auto pfn_vkGetPhysicalDeviceFeatures2   = LOAD(vkGetPhysicalDeviceFeatures2);
    auto pfn_vkEnumerateDeviceExtensionProperties = LOAD(vkEnumerateDeviceExtensionProperties);
    auto pfn_vkGetInstanceProcAddr          = LOAD(vkGetInstanceProcAddr);
    auto pfn_vkDestroyInstance              = LOAD(vkDestroyInstance);

    if (!pfn_vkCreateInstance || !pfn_vkEnumeratePhysicalDevices) {
        fprintf(stderr, "{\"error\":\"vk symbols missing\"}\n");
        return 3;
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "triad-vk-probe";
    app.applicationVersion = 1;
    app.pEngineName = "triad";
    app.engineVersion = 1;
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ic{};
    ic.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ic.pApplicationInfo = &app;

    VkInstance inst;
    if (pfn_vkCreateInstance(&ic, nullptr, &inst) != VK_SUCCESS) {
        // Try downgrading apiVersion in case the loader is < 1.3.
        app.apiVersion = VK_API_VERSION_1_2;
        if (pfn_vkCreateInstance(&ic, nullptr, &inst) != VK_SUCCESS) {
            fprintf(stderr, "{\"error\":\"vkCreateInstance failed (1.3 and 1.2)\"}\n");
            return 4;
        }
    }

    uint32_t pdc = 0;
    pfn_vkEnumeratePhysicalDevices(inst, &pdc, nullptr);
    if (pdc == 0) {
        fprintf(stderr, "{\"error\":\"no physical devices\"}\n");
        pfn_vkDestroyInstance(inst, nullptr);
        return 5;
    }
    std::vector<VkPhysicalDevice> pds(pdc);
    pfn_vkEnumeratePhysicalDevices(inst, &pdc, pds.data());

    printf("{\n  \"device_count\": %u,\n  \"devices\": [\n", pdc);

    for (uint32_t pi = 0; pi < pdc; ++pi) {
        VkPhysicalDevice pd = pds[pi];

        // Properties chain.
        VkPhysicalDeviceSubgroupProperties subgroup{};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeCtl{};
        subgroupSizeCtl.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
        subgroupSizeCtl.pNext = &subgroup;

        VkPhysicalDeviceShaderIntegerDotProductProperties idotProps{};
        idotProps.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
        idotProps.pNext = &subgroupSizeCtl;

        VkPhysicalDeviceVulkan13Properties v13p{};
        v13p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        v13p.pNext = &idotProps;

        VkPhysicalDeviceVulkan12Properties v12p{};
        v12p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
        v12p.pNext = &v13p;

        VkPhysicalDeviceVulkan11Properties v11p{};
        v11p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        v11p.pNext = &v12p;

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &v11p;

        if (pfn_vkGetPhysicalDeviceProperties2) {
            pfn_vkGetPhysicalDeviceProperties2(pd, &props2);
        } else {
            VkPhysicalDeviceProperties props{};
            pfn_vkGetPhysicalDeviceProperties(pd, &props);
            props2.properties = props;
        }

        // Features chain.
        VkPhysicalDeviceShaderIntegerDotProductFeatures idotFeat{};
        idotFeat.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;

        VkPhysicalDeviceShaderFloat16Int8Features f16i8{};
        f16i8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        f16i8.pNext = &idotFeat;

        VkPhysicalDevice16BitStorageFeatures s16{};
        s16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        s16.pNext = &f16i8;

        VkPhysicalDevice8BitStorageFeatures s8{};
        s8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
        s8.pNext = &s16;

        VkPhysicalDeviceFeatures2 feat2{};
        feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feat2.pNext = &s8;

        if (pfn_vkGetPhysicalDeviceFeatures2) {
            pfn_vkGetPhysicalDeviceFeatures2(pd, &feat2);
        }

        // Extensions.
        uint32_t ec = 0;
        pfn_vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, nullptr);
        std::vector<VkExtensionProperties> exts(ec);
        pfn_vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, exts.data());

        bool has_coopmat = false;
        bool has_subgroup_size_ctl = false;
        for (auto& e : exts) {
            if (strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0) has_coopmat = true;
            if (strcmp(e.extensionName, "VK_EXT_subgroup_size_control") == 0) has_subgroup_size_ctl = true;
        }

        // Cooperative-matrix detail (KHR).
        std::string coopmat_detail = "[]";
        if (has_coopmat) {
            auto pfn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
                pfn_vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
            if (pfn) {
                uint32_t cmc = 0;
                pfn(pd, &cmc, nullptr);
                std::vector<VkCooperativeMatrixPropertiesKHR> cm(cmc);
                for (auto& p : cm) p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                pfn(pd, &cmc, cm.data());
                coopmat_detail = "[";
                for (uint32_t k = 0; k < cmc; ++k) {
                    char buf[256];
                    snprintf(buf, sizeof buf,
                        "%s{\"M\":%u,\"N\":%u,\"K\":%u,\"AType\":%d,\"BType\":%d,\"CType\":%d,\"ResultType\":%d,\"saturatingAccumulation\":%s,\"scope\":%d}",
                        k ? "," : "",
                        cm[k].MSize, cm[k].NSize, cm[k].KSize,
                        (int)cm[k].AType, (int)cm[k].BType, (int)cm[k].CType,
                        (int)cm[k].ResultType,
                        cm[k].saturatingAccumulation ? "true" : "false",
                        (int)cm[k].scope);
                    coopmat_detail += buf;
                }
                coopmat_detail += "]";
            }
        }

        // Build extension list as JSON array of strings.
        std::string ext_arr = "[";
        for (uint32_t k = 0; k < ec; ++k) {
            if (k) ext_arr += ",";
            ext_arr += "\"";
            ext_arr += exts[k].extensionName;
            ext_arr += "\"";
        }
        ext_arr += "]";

        printf("    {\n");
        printf("      \"index\": %u,\n", pi);
        printf("      \"deviceName\": %s,\n", qstr(props2.properties.deviceName));
        printf("      \"deviceType\": %u,\n", (unsigned)props2.properties.deviceType);
        printf("      \"vendorID\": \"0x%04x\",\n", props2.properties.vendorID);
        printf("      \"deviceID\": \"0x%04x\",\n", props2.properties.deviceID);
        printf("      \"apiVersion\": \"%u.%u.%u\",\n",
               VK_VERSION_MAJOR(props2.properties.apiVersion),
               VK_VERSION_MINOR(props2.properties.apiVersion),
               VK_VERSION_PATCH(props2.properties.apiVersion));
        printf("      \"driverVersion\": \"0x%08x\",\n", props2.properties.driverVersion);
        printf("      \"driverName\": %s,\n", qstr(v12p.driverName));
        printf("      \"driverInfo\": %s,\n", qstr(v12p.driverInfo));
        printf("      \"limits\": {\n");
        printf("        \"maxComputeWorkGroupInvocations\": %u,\n",
               props2.properties.limits.maxComputeWorkGroupInvocations);
        printf("        \"maxComputeSharedMemorySize\": %u,\n",
               props2.properties.limits.maxComputeSharedMemorySize);
        printf("        \"maxComputeWorkGroupSize\": [%u,%u,%u],\n",
               props2.properties.limits.maxComputeWorkGroupSize[0],
               props2.properties.limits.maxComputeWorkGroupSize[1],
               props2.properties.limits.maxComputeWorkGroupSize[2]);
        printf("        \"maxComputeWorkGroupCount\": [%u,%u,%u]\n",
               props2.properties.limits.maxComputeWorkGroupCount[0],
               props2.properties.limits.maxComputeWorkGroupCount[1],
               props2.properties.limits.maxComputeWorkGroupCount[2]);
        printf("      },\n");
        printf("      \"subgroup\": {\n");
        printf("        \"subgroupSize\": %u,\n", subgroup.subgroupSize);
        printf("        \"supportedStages\": \"0x%08x\",\n", subgroup.supportedStages);
        printf("        \"supportedOperations\": \"0x%08x\",\n", subgroup.supportedOperations);
        printf("        \"quadOperationsInAllStages\": %s\n",
               subgroup.quadOperationsInAllStages ? "true" : "false");
        printf("      },\n");
        printf("      \"subgroupSizeControl\": {\n");
        printf("        \"min\": %u,\n", subgroupSizeCtl.minSubgroupSize);
        printf("        \"max\": %u,\n", subgroupSizeCtl.maxSubgroupSize);
        printf("        \"maxComputeWorkgroupSubgroups\": %u,\n",
               subgroupSizeCtl.maxComputeWorkgroupSubgroups);
        printf("        \"requiredSubgroupSizeStages\": \"0x%08x\"\n",
               subgroupSizeCtl.requiredSubgroupSizeStages);
        printf("      },\n");
        printf("      \"integerDotProduct\": {\n");
        printf("        \"shaderIntegerDotProduct\": %s,\n",
               idotFeat.shaderIntegerDotProduct ? "true" : "false");
        // Spec field names use "4x8Bit" (four 8-bit lanes packed into a 32-bit
        // word). The Phase-0 SUMMARY just wants Y/N so we expose the canonical
        // ones plus the saturating accumulating variant.
        printf("        \"int4x8PackedSignedAccel\": %s,\n",
               idotProps.integerDotProduct4x8BitPackedSignedAccelerated ? "true" : "false");
        printf("        \"int4x8PackedUnsignedAccel\": %s,\n",
               idotProps.integerDotProduct4x8BitPackedUnsignedAccelerated ? "true" : "false");
        printf("        \"int4x8PackedMixedAccel\": %s,\n",
               idotProps.integerDotProduct4x8BitPackedMixedSignednessAccelerated ? "true" : "false");
        printf("        \"int4x8AccumSatSignedAccel\": %s\n",
               idotProps.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated ? "true" : "false");
        printf("      },\n");
        printf("      \"shaderFeatures\": {\n");
        printf("        \"shaderFloat16\": %s,\n", f16i8.shaderFloat16 ? "true" : "false");
        printf("        \"shaderInt8\": %s,\n", f16i8.shaderInt8 ? "true" : "false");
        printf("        \"storageBuffer16BitAccess\": %s,\n", s16.storageBuffer16BitAccess ? "true" : "false");
        printf("        \"uniformAndStorageBuffer16BitAccess\": %s,\n", s16.uniformAndStorageBuffer16BitAccess ? "true" : "false");
        printf("        \"storagePushConstant16\": %s,\n", s16.storagePushConstant16 ? "true" : "false");
        printf("        \"storageBuffer8BitAccess\": %s,\n", s8.storageBuffer8BitAccess ? "true" : "false");
        printf("        \"uniformAndStorageBuffer8BitAccess\": %s,\n", s8.uniformAndStorageBuffer8BitAccess ? "true" : "false");
        printf("        \"storagePushConstant8\": %s\n", s8.storagePushConstant8 ? "true" : "false");
        printf("      },\n");
        printf("      \"hasCooperativeMatrixKHR\": %s,\n", has_coopmat ? "true" : "false");
        printf("      \"cooperativeMatrixProperties\": %s,\n", coopmat_detail.c_str());
        printf("      \"hasSubgroupSizeControl\": %s,\n", has_subgroup_size_ctl ? "true" : "false");
        printf("      \"extensions\": %s\n", ext_arr.c_str());
        printf("    }%s\n", (pi + 1 == pdc) ? "" : ",");
    }

    printf("  ]\n}\n");

    pfn_vkDestroyInstance(inst, nullptr);
    dlclose(libvk);
    (void)argc; (void)argv;
    return 0;
}
