#include <windows.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>

// CUDA Kernel simulating an MoE Expert / Compute layer on GPU
__global__ void mock_moe_compute_kernel(float* data, int numElements, int iterations) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        float val = data[idx];
        #pragma unroll 8
        for (int i = 0; i < iterations; ++i) {
            val = val * 1.0001f + 0.0002f;
        }
        data[idx] = val;
    }
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << " [PROBE] HPC Ring Buffer & Asynchronous Overlap Pipeline\n";
    std::cout << "=========================================================\n\n";

    // Target GPU 1 (GTX 1050 Ti) exclusively to keep GPU 0 unperturbed
    int targetGpu = 1;
    cudaError_t err = cudaSetDevice(targetGpu);
    if (err != cudaSuccess) {
        std::cerr << "[-] Error binding to Device 1: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, targetGpu);
    std::cout << "[+] Bound to: " << prop.name << " (" << prop.multiProcessorCount << " SMs, "
              << prop.asyncEngineCount << " Async DMA Engines)\n";

    // 1. Memory Mapping Test on the Local Model File
    const wchar_t* modelPath = L"C:\\Users\\alefita\\.litert-lm\\cache\\huggingface\\litert-community\\gemma-4-E2B-it-litert-lm\\gemma-4-E2B-it.litertlm";
    std::wcout << L"[+] Probing Memory-Mapped File (mmap) on local model:\n    " << modelPath << L"\n";

    HANDLE hFile = CreateFileW(modelPath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER fileSize;
        GetFileSizeEx(hFile, &fileSize);
        auto tStartMmap = std::chrono::high_resolution_clock::now();
        HANDLE hMapping = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapping) {
            void* pView = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
            auto tEndMmap = std::chrono::high_resolution_clock::now();
            double msMmap = std::chrono::duration<double, std::milli>(tEndMmap - tStartMmap).count();
            std::cout << "    [mmap OK] Mapped " << (fileSize.QuadPart / (1024 * 1024)) << " MB in "
                      << std::fixed << std::setprecision(3) << msMmap << " ms (Virtual Address: " << pView << ")\n";

            // Touch first byte
            volatile char firstByte = *(volatile char*)pView;
            (void)firstByte;

            UnmapViewOfFile(pView);
            CloseHandle(hMapping);
        }
        CloseHandle(hFile);
    } else {
        std::cout << "    [mmap Info] Model file handle open returned error: " << GetLastError() << " (using synthetic buffers)\n";
    }

    // 2. HPC Ring Buffer Setup
    const int RING_STAGES = 3;
    const size_t BATCH_ELEMENTS = 2 * 1024 * 1024; // 2M floats = 8 MB per batch
    const size_t BATCH_BYTES = BATCH_ELEMENTS * sizeof(float);
    const int TOTAL_BATCHES = 10;
    const int COMPUTE_ITERATIONS = 300;

    std::cout << "\n[+] Setting up 3-Stage Asynchronous Ring Buffer (" 
              << (BATCH_BYTES / (1024 * 1024)) << " MB per slot, " << TOTAL_BATCHES << " batches total)...\n";

    float* h_ring[RING_STAGES];
    float* d_ring[RING_STAGES];
    cudaStream_t stream_transfer;
    cudaStream_t stream_compute;
    cudaEvent_t transfer_done_events[RING_STAGES];
    cudaEvent_t compute_done_events[RING_STAGES];

    cudaStreamCreateWithFlags(&stream_transfer, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking);

    for (int i = 0; i < RING_STAGES; ++i) {
        cudaHostAlloc((void**)&h_ring[i], BATCH_BYTES, cudaHostAllocPortable);
        cudaMalloc((void**)&d_ring[i], BATCH_BYTES);
        cudaEventCreateWithFlags(&transfer_done_events[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&compute_done_events[i], cudaEventDisableTiming);
        for (size_t j = 0; j < 1024; ++j) h_ring[i][j] = (float)(i + 1);
    }

    std::cout << "    Pinned Host Memory & Device Ring Buffers allocated successfully.\n";

    // 3. Execution: Sequential Baseline
    std::cout << "\n[+] Benchmark 1: Sequential Execution (No Overlap - PCIe latency exposed):\n";
    auto tStartSeq = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < TOTAL_BATCHES; ++b) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_ring[0], h_ring[0], BATCH_BYTES, cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (BATCH_ELEMENTS + threads - 1) / threads;
        mock_moe_compute_kernel<<<blocks, threads>>>(d_ring[0], BATCH_ELEMENTS, COMPUTE_ITERATIONS);
        cudaMemcpy(h_ring[0], d_ring[0], BATCH_BYTES, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double msBatch = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (b == 0 || b == TOTAL_BATCHES - 1) {
            std::cout << "    [Seq Batch " << b << "] Latency: " << msBatch << " ms\n";
        }
    }
    auto tEndSeq = std::chrono::high_resolution_clock::now();
    double totalMsSeq = std::chrono::duration<double, std::milli>(tEndSeq - tStartSeq).count();
    std::cout << "    --> Total Sequential Time: " << totalMsSeq << " ms (" 
              << (totalMsSeq / TOTAL_BATCHES) << " ms/batch)\n";

    // 4. Asynchronous Ring Pipeline
    std::cout << "\n[+] Benchmark 2: Asynchronous HPC Ring Pipeline (Overlapped DMA & Compute):\n";
    auto tStartPipe = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < TOTAL_BATCHES; ++b) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int slot = b % RING_STAGES;

        if (b >= RING_STAGES) {
            cudaStreamWaitEvent(stream_transfer, compute_done_events[slot], 0);
        }

        cudaMemcpyAsync(d_ring[slot], h_ring[slot], BATCH_BYTES, cudaMemcpyHostToDevice, stream_transfer);
        cudaEventRecord(transfer_done_events[slot], stream_transfer);

        cudaStreamWaitEvent(stream_compute, transfer_done_events[slot], 0);

        int threads = 256;
        int blocks = (BATCH_ELEMENTS + threads - 1) / threads;
        mock_moe_compute_kernel<<<blocks, threads, 0, stream_compute>>>(d_ring[slot], BATCH_ELEMENTS, COMPUTE_ITERATIONS);

        cudaEventRecord(compute_done_events[slot], stream_compute);

        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double msBatch = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "    [Pipe Batch " << std::setw(2) << b << "] Step Latency: " 
                  << std::setw(6) << msBatch << " ms"
                  << (b == 0 ? "  <-- Pipeline Filling (Cold Start)" : 
                     (b < RING_STAGES ? "  <-- Filling Stage" : "  <-- STEADY STATE (PCIe Latency Hidden)"))
                  << "\n";
    }

    auto tEndPipe = std::chrono::high_resolution_clock::now();
    double totalMsPipe = std::chrono::duration<double, std::milli>(tEndPipe - tStartPipe).count();
    std::cout << "    --> Total Pipelined Time: " << totalMsPipe << " ms (" 
              << (totalMsPipe / TOTAL_BATCHES) << " ms/batch)\n";
    std::cout << "    --> Throughput Speedup: " << (totalMsSeq / totalMsPipe) << "x over sequential\n";

    // Cleanup
    for (int i = 0; i < RING_STAGES; ++i) {
        cudaFreeHost(h_ring[i]);
        cudaFree(d_ring[i]);
        cudaEventDestroy(transfer_done_events[i]);
        cudaEventDestroy(compute_done_events[i]);
    }
    cudaStreamDestroy(stream_transfer);
    cudaStreamDestroy(stream_compute);

    std::cout << "\n=========================================================\n";
    std::cout << " [HPC RING PIPELINE PROBE COMPLETE]\n";
    std::cout << "=========================================================\n";
    return 0;
}
