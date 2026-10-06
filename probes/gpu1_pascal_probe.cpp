#include <windows.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>

typedef uint32_t (*NvEncodeAPIGetMaxSupportedVersion_t)(uint32_t* version);
typedef uint32_t (*NvOFGetMaxSupportedApiVersion_t)(uint32_t* version);

int main() {
    std::cout << "=========================================================\n";
    std::cout << " [PROBE] NVIDIA Heterogeneous ASICs & GPU 1 Pascal Probe\n";
    std::cout << "=========================================================\n\n";

    // 1. CUDA Device Enumeration
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "[-] Error enumerating CUDA devices: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "[+] Found " << deviceCount << " CUDA Devices.\n";
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "    [Device " << i << "] " << prop.name 
                  << " (Compute " << prop.major << "." << prop.minor << ")\n"
                  << "        SMs: " << prop.multiProcessorCount 
                  << " | CUDA Cores: " << (prop.major == 6 ? prop.multiProcessorCount * 128 : prop.multiProcessorCount * 64)
                  << " | VRAM: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n"
                  << "        Bus Width: " << prop.memoryBusWidth << " bits"
                  << " | L2 Cache: " << (prop.l2CacheSize / 1024) << " KB"
                  << " | Async DMA Copy Engines: " << prop.asyncEngineCount << "\n";
    }

    // 2. Target Device 1 (GTX 1050 Ti) exclusively
    int targetGpu = 1;
    if (deviceCount > 1) {
        std::cout << "\n[+] Binding exclusively to Device " << targetGpu << " (GTX 1050 Ti)...\n";
        cudaSetDevice(targetGpu);

        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        std::cout << "    Free VRAM: " << (freeMem / (1024 * 1024)) << " MB / " 
                  << (totalMem / (1024 * 1024)) << " MB\n";

        // Quick atomic 4MB bandwidth probe over PCIe
        const size_t testBytes = 4 * 1024 * 1024; // 4MB
        void* d_ptr = nullptr;
        err = cudaMalloc(&d_ptr, testBytes);
        if (err == cudaSuccess) {
            void* h_ptr = malloc(testBytes);
            memset(h_ptr, 0x5A, testBytes);

            auto startH2D = std::chrono::high_resolution_clock::now();
            cudaMemcpy(d_ptr, h_ptr, testBytes, cudaMemcpyHostToDevice);
            auto endH2D = std::chrono::high_resolution_clock::now();

            auto startD2H = std::chrono::high_resolution_clock::now();
            cudaMemcpy(h_ptr, d_ptr, testBytes, cudaMemcpyDeviceToHost);
            auto endD2H = std::chrono::high_resolution_clock::now();

            double msH2D = std::chrono::duration<double, std::milli>(endH2D - startH2D).count();
            double msD2H = std::chrono::duration<double, std::milli>(endD2H - startD2H).count();
            double bwH2D = (testBytes / (1024.0 * 1024.0 * 1024.0)) / (msH2D / 1000.0);
            double bwD2H = (testBytes / (1024.0 * 1024.0 * 1024.0)) / (msD2H / 1000.0);

            std::cout << "    [PCIe 4MB Test] H2D Latency: " << std::fixed << std::setprecision(3) << msH2D << " ms (" 
                      << bwH2D << " GB/s) | D2H Latency: " << msD2H << " ms (" << bwD2H << " GB/s)\n";

            cudaFree(d_ptr);
            free(h_ptr);
        } else {
            std::cerr << "[-] Failed cudaMalloc on Device 1: " << cudaGetErrorString(err) << "\n";
        }
    } else {
        std::cout << "[!] Only 1 GPU found. Skipping Device 1 binding.\n";
    }

    // 3. Inspect NVENC Driver API
    std::cout << "\n[+] Inspecting NVENC Driver (nvEncodeAPI64.dll)...\n";
    HMODULE hNvEnc = LoadLibraryA("nvEncodeAPI64.dll");
    if (hNvEnc) {
        auto pfnGetMaxVer = (NvEncodeAPIGetMaxSupportedVersion_t)GetProcAddress(hNvEnc, "NvEncodeAPIGetMaxSupportedVersion");
        if (pfnGetMaxVer) {
            uint32_t ver = 0;
            pfnGetMaxVer(&ver);
            uint32_t major = ver >> 4;
            uint32_t minor = ver & 0xF;
            std::cout << "    [NVENC] Loaded OK. Max Supported Version: " << major << "." << minor << " (Raw: 0x" << std::hex << ver << std::dec << ")\n";
        } else {
            std::cout << "    [NVENC] Loaded OK, but NvEncodeAPIGetMaxSupportedVersion symbol not found.\n";
        }
        FreeLibrary(hNvEnc);
    } else {
        std::cout << "[-] nvEncodeAPI64.dll failed to load.\n";
    }

    // 4. Inspect NVDEC Driver (nvcuvid.dll)
    std::cout << "\n[+] Inspecting NVDEC Driver (nvcuvid.dll)...\n";
    HMODULE hNvCuvid = LoadLibraryA("nvcuvid.dll");
    if (hNvCuvid) {
        void* pCreateDecoder = (void*)GetProcAddress(hNvCuvid, "cuvidCreateDecoder");
        void* pDecodePic = (void*)GetProcAddress(hNvCuvid, "cuvidDecodePicture");
        void* pMapFrame = (void*)GetProcAddress(hNvCuvid, "cuvidMapVideoFrame64");
        std::cout << "    [NVDEC] Loaded OK.\n"
                  << "        cuvidCreateDecoder: " << (pCreateDecoder ? "Present" : "Missing") << "\n"
                  << "        cuvidDecodePicture: " << (pDecodePic ? "Present" : "Missing") << "\n"
                  << "        cuvidMapVideoFrame64: " << (pMapFrame ? "Present" : "Missing") << "\n";
        FreeLibrary(hNvCuvid);
    } else {
        std::cout << "[-] nvcuvid.dll failed to load.\n";
    }

    // 5. Inspect NVOFA Driver (nvofapi64.dll)
    std::cout << "\n[+] Inspecting NVOFA Driver (nvofapi64.dll)...\n";
    HMODULE hNvOf = LoadLibraryA("nvofapi64.dll");
    if (hNvOf) {
        auto pfnOfVer = (NvOFGetMaxSupportedApiVersion_t)GetProcAddress(hNvOf, "NvOFGetMaxSupportedApiVersion");
        if (pfnOfVer) {
            uint32_t ofVer = 0;
            pfnOfVer(&ofVer);
            std::cout << "    [NVOFA] Loaded OK. Max API Version: 0x" << std::hex << ofVer << std::dec << "\n";
        }
        FreeLibrary(hNvOf);
    } else {
        std::cout << "[-] nvofapi64.dll failed to load.\n";
    }

    std::cout << "\n=========================================================\n";
    std::cout << " [PROBE COMPLETED SUCCESSFULLY]\n";
    std::cout << "=========================================================\n";
    return 0;
}
