// Triad-ML / Phase 0 — OpenCL capability probe for Xclipse 950 (Exynos 2500).
// Cross-compiled for arm64-v8a, API 28+. NDK does not ship CL headers, so we
// inline the minimal Khronos cl.h subset we need (function signatures + a few
// constants). libOpenCL.so is dlopen'd dynamically (no link-time dep).
//
// Output: JSON on stdout — committed verbatim to docs/probe/xclipse-950-cl.json.

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>

// --- minimal cl.h subset (Khronos OpenCL spec, MIT-style headers --------------
// Just typedefs + the constants we need. No functional CL code is reimplemented;
// we only call into the vendor ICD at runtime.

typedef int32_t                     cl_int;
typedef uint32_t                    cl_uint;
typedef uint64_t                    cl_ulong;
typedef cl_uint                     cl_bool;
typedef cl_uint                     cl_platform_info;
typedef cl_uint                     cl_device_info;
typedef cl_ulong                    cl_device_type;
typedef struct _cl_platform_id*     cl_platform_id;
typedef struct _cl_device_id*       cl_device_id;

#define CL_DEVICE_TYPE_ALL                  0xFFFFFFFFULL
#define CL_PLATFORM_NAME                    0x0902
#define CL_PLATFORM_VENDOR                  0x0903
#define CL_PLATFORM_VERSION                 0x0901
#define CL_PLATFORM_PROFILE                 0x0900
#define CL_PLATFORM_EXTENSIONS              0x0904

#define CL_DEVICE_NAME                      0x102B
#define CL_DEVICE_VENDOR                    0x102C
#define CL_DEVICE_VERSION                   0x102F
#define CL_DRIVER_VERSION                   0x102D
#define CL_DEVICE_OPENCL_C_VERSION          0x103D
#define CL_DEVICE_EXTENSIONS                0x1030
#define CL_DEVICE_TYPE                      0x1000
#define CL_DEVICE_MAX_COMPUTE_UNITS         0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS  0x1003
#define CL_DEVICE_MAX_WORK_ITEM_SIZES       0x1005
#define CL_DEVICE_MAX_WORK_GROUP_SIZE       0x1004
#define CL_DEVICE_GLOBAL_MEM_SIZE           0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE            0x1023
#define CL_DEVICE_LOCAL_MEM_TYPE            0x1022
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE        0x1010
#define CL_DEVICE_IMAGE_SUPPORT             0x1016
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF  0x1034
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT 0x1006
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF     0x103B
#define CL_DEVICE_SVM_CAPABILITIES          0x1053
#define CL_DEVICE_HALF_FP_CONFIG            0x1033
#define CL_DEVICE_SINGLE_FP_CONFIG          0x101B
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE     0x101E
#define CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE 0x101D

typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*PFN_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*PFN_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);

static const char* qstr(const char* s) {
    static char buf[1024];
    snprintf(buf, sizeof buf, "\"%s\"", s ? s : "");
    return buf;
}

static std::string get_str(PFN_clGetDeviceInfo fn, cl_device_id d, cl_device_info q) {
    size_t n = 0;
    if (fn(d, q, 0, nullptr, &n) != 0 || n == 0) return "";
    std::string s(n, '\0');
    fn(d, q, n, s.data(), nullptr);
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}
static std::string get_pstr(PFN_clGetPlatformInfo fn, cl_platform_id p, cl_platform_info q) {
    size_t n = 0;
    if (fn(p, q, 0, nullptr, &n) != 0 || n == 0) return "";
    std::string s(n, '\0');
    fn(p, q, n, s.data(), nullptr);
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

template <typename T>
static T get_v(PFN_clGetDeviceInfo fn, cl_device_id d, cl_device_info q) {
    T v = 0;
    fn(d, q, sizeof v, &v, nullptr);
    return v;
}

int main(int argc, char** argv) {
    void* libcl = dlopen("libOpenCL.so", RTLD_NOW);
    if (!libcl) {
        fprintf(stderr, "{\"error\":\"libOpenCL.so dlopen failed: %s\"}\n", dlerror());
        return 2;
    }
    auto clGetPlatformIDs  = (PFN_clGetPlatformIDs) dlsym(libcl, "clGetPlatformIDs");
    auto clGetPlatformInfo = (PFN_clGetPlatformInfo)dlsym(libcl, "clGetPlatformInfo");
    auto clGetDeviceIDs    = (PFN_clGetDeviceIDs)   dlsym(libcl, "clGetDeviceIDs");
    auto clGetDeviceInfo   = (PFN_clGetDeviceInfo)  dlsym(libcl, "clGetDeviceInfo");
    if (!clGetPlatformIDs || !clGetPlatformInfo || !clGetDeviceIDs || !clGetDeviceInfo) {
        fprintf(stderr, "{\"error\":\"CL symbols missing\"}\n");
        return 3;
    }

    cl_uint pc = 0;
    if (clGetPlatformIDs(0, nullptr, &pc) != 0 || pc == 0) {
        fprintf(stderr, "{\"error\":\"no OpenCL platforms\"}\n");
        return 4;
    }
    std::vector<cl_platform_id> ps(pc);
    clGetPlatformIDs(pc, ps.data(), nullptr);

    printf("{\n  \"platform_count\": %u,\n  \"platforms\": [\n", pc);
    for (cl_uint i = 0; i < pc; ++i) {
        std::string pname    = get_pstr(clGetPlatformInfo, ps[i], CL_PLATFORM_NAME);
        std::string pvendor  = get_pstr(clGetPlatformInfo, ps[i], CL_PLATFORM_VENDOR);
        std::string pversion = get_pstr(clGetPlatformInfo, ps[i], CL_PLATFORM_VERSION);
        std::string pprofile = get_pstr(clGetPlatformInfo, ps[i], CL_PLATFORM_PROFILE);
        std::string pexts    = get_pstr(clGetPlatformInfo, ps[i], CL_PLATFORM_EXTENSIONS);

        cl_uint dc = 0;
        clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &dc);
        std::vector<cl_device_id> ds(dc);
        if (dc) clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_ALL, dc, ds.data(), nullptr);

        printf("    {\n");
        printf("      \"name\": %s,\n", qstr(pname.c_str()));
        printf("      \"vendor\": %s,\n", qstr(pvendor.c_str()));
        printf("      \"version\": %s,\n", qstr(pversion.c_str()));
        printf("      \"profile\": %s,\n", qstr(pprofile.c_str()));
        printf("      \"extensions\": %s,\n", qstr(pexts.c_str()));
        printf("      \"device_count\": %u,\n", dc);
        printf("      \"devices\": [\n");
        for (cl_uint j = 0; j < dc; ++j) {
            std::string nm    = get_str(clGetDeviceInfo, ds[j], CL_DEVICE_NAME);
            std::string vd    = get_str(clGetDeviceInfo, ds[j], CL_DEVICE_VENDOR);
            std::string ver   = get_str(clGetDeviceInfo, ds[j], CL_DEVICE_VERSION);
            std::string drv   = get_str(clGetDeviceInfo, ds[j], CL_DRIVER_VERSION);
            std::string clcv  = get_str(clGetDeviceInfo, ds[j], CL_DEVICE_OPENCL_C_VERSION);
            std::string exts  = get_str(clGetDeviceInfo, ds[j], CL_DEVICE_EXTENSIONS);
            cl_uint cu      = get_v<cl_uint>(clGetDeviceInfo, ds[j], CL_DEVICE_MAX_COMPUTE_UNITS);
            size_t  wgs     = get_v<size_t> (clGetDeviceInfo, ds[j], CL_DEVICE_MAX_WORK_GROUP_SIZE);
            cl_ulong gms    = get_v<cl_ulong>(clGetDeviceInfo, ds[j], CL_DEVICE_GLOBAL_MEM_SIZE);
            cl_ulong lms    = get_v<cl_ulong>(clGetDeviceInfo, ds[j], CL_DEVICE_LOCAL_MEM_SIZE);
            cl_ulong mas    = get_v<cl_ulong>(clGetDeviceInfo, ds[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            cl_bool imgs    = get_v<cl_bool> (clGetDeviceInfo, ds[j], CL_DEVICE_IMAGE_SUPPORT);
            cl_uint pvw_h   = get_v<cl_uint> (clGetDeviceInfo, ds[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF);
            cl_uint nvw_h   = get_v<cl_uint> (clGetDeviceInfo, ds[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF);
            cl_ulong svm    = get_v<cl_ulong>(clGetDeviceInfo, ds[j], CL_DEVICE_SVM_CAPABILITIES);
            cl_ulong fp16cfg= get_v<cl_ulong>(clGetDeviceInfo, ds[j], CL_DEVICE_HALF_FP_CONFIG);
            cl_ulong gcache = get_v<cl_ulong>(clGetDeviceInfo, ds[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);

            printf("        {\n");
            printf("          \"name\": %s,\n", qstr(nm.c_str()));
            printf("          \"vendor\": %s,\n", qstr(vd.c_str()));
            printf("          \"version\": %s,\n", qstr(ver.c_str()));
            printf("          \"driverVersion\": %s,\n", qstr(drv.c_str()));
            printf("          \"openCL_C_version\": %s,\n", qstr(clcv.c_str()));
            printf("          \"maxComputeUnits\": %u,\n", cu);
            printf("          \"maxWorkGroupSize\": %zu,\n", wgs);
            printf("          \"globalMemSize\": %llu,\n", (unsigned long long)gms);
            printf("          \"localMemSize\": %llu,\n", (unsigned long long)lms);
            printf("          \"maxMemAllocSize\": %llu,\n", (unsigned long long)mas);
            printf("          \"globalMemCacheSize\": %llu,\n", (unsigned long long)gcache);
            printf("          \"imageSupport\": %s,\n", imgs ? "true" : "false");
            printf("          \"preferredVectorWidthHalf\": %u,\n", pvw_h);
            printf("          \"nativeVectorWidthHalf\": %u,\n", nvw_h);
            printf("          \"svmCapabilities\": \"0x%llx\",\n", (unsigned long long)svm);
            printf("          \"halfFpConfig\": \"0x%llx\",\n", (unsigned long long)fp16cfg);
            printf("          \"extensions\": %s\n", qstr(exts.c_str()));
            printf("        }%s\n", (j + 1 == dc) ? "" : ",");
        }
        printf("      ]\n");
        printf("    }%s\n", (i + 1 == pc) ? "" : ",");
    }
    printf("  ]\n}\n");

    dlclose(libcl);
    (void)argc; (void)argv;
    return 0;
}
