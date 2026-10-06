#include <windows.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    std::cout << "=========================================================\n";
    std::cout << " [PROBE] HPC Ring Buffer & Asynchronous Memory Pipeline\n";
    std::cout << "=========================================================\n\n";

    int targetGpu = 1;
    cudaSetDevice(targetGpu);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, targetGpu);
    std::cout << "[+] Bound to: " << prop.name << " (" << prop.multiProcessorCount << " SMs, "
              << prop.asyncEngineCount << " Async DMA Engines)\n";

    // 1. Memory-Mapped File (mmap) Probe
    const wchar_t* modelPath = L"C:\\Users\\alefita\\.litert-lm\\cache\\huggingface\\litert-community\\gemma-4-E2B-it-litert-lm\\gemma-4-E2B-it.litertlm";
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
            UnmapViewOfFile(pView);
            CloseHandle(hMapping);
        }
        CloseHandle(hFile);
    }

    // 2. Ring Buffer
    const int RING_STAGES = 3;
    const size_t BATCH_ELEMENTS = 4 * 1024 * 1024; // 16 MB per slot
    const size_t BATCH_BYTES = BATCH_ELEMENTS * sizeof(float);
    const int TOTAL_BATCHES = 12;

    float* h_ring[RING_STAGES];
    float* d_ring[RING_STAGES];
    cudaStream_t stream_transfer;
    cudaStream_t stream_compute;
    cudaEvent_t transfer_done[RING_STAGES];
    cudaEvent_t compute_done[RING_STAGES];

    cudaStreamCreateWithFlags(&stream_transfer, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking);

    for (int i = 0; i < RING_STAGES; ++i) {
        cudaHostAlloc((void**)&h_ring[i], BATCH_BYTES, cudaHostAllocPortable);
        cudaMalloc((void**)&d_ring[i], BATCH_BYTES);
        cudaEventCreateWithFlags(&transfer_done[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&compute_done[i], cudaEventDisableTiming);
        memset(h_ring[i], 0x3F, BATCH_BYTES);
    }

    // Benchmark 1: Sequential Execution (Exposed Latency)
    std::cout << "\n[+] Benchmark 1: Sequential Execution (PCIe latency fully exposed):\n";
    auto tStartSeq = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < TOTAL_BATCHES; ++b) {
        cudaMemcpy(d_ring[0], h_ring[0], BATCH_BYTES, cudaMemcpyHostToDevice);
        cudaMemset(d_ring[0], 0x55, BATCH_BYTES);
        cudaMemcpy(h_ring[0], d_ring[0], BATCH_BYTES, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    auto tEndSeq = std::chrono::high_resolution_clock::now();
    double totalMsSeq = std::chrono::duration<double, std::milli>(tEndSeq - tStartSeq).count();
    std::cout << "    --> Total Sequential Time: " << totalMsSeq << " ms (" 
              << (totalMsSeq / TOTAL_BATCHES) << " ms/batch)\n";

    // Benchmark 2: True Asynchronous Ring Pipeline
    std::cout << "\n[+] Benchmark 2: True Asynchronous HPC Ring Pipeline (Overlapped in flight):\n";
    auto tStartPipe = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < TOTAL_BATCHES; ++b) {
        int slot = b % RING_STAGES;

        // Slot recycling barrier: if we already used this slot in a previous cycle, wait until its compute finishes
        if (b >= RING_STAGES) {
            cudaStreamWaitEvent(stream_transfer, compute_done[slot], 0);
        }

        // 1. Asynchronous transfer to device
        cudaMemcpyAsync(d_ring[slot], h_ring[slot], BATCH_BYTES, cudaMemcpyHostToDevice, stream_transfer);
        cudaEventRecord(transfer_done[slot], stream_transfer);

        // 2. Compute stream waits for transfer
        cudaStreamWaitEvent(stream_compute, transfer_done[slot], 0);
        cudaMemsetAsync(d_ring[slot], 0x55, BATCH_BYTES, stream_compute);
        cudaEventRecord(compute_done[slot], stream_compute);

        // 3. Return transfer stream waits for compute and pulls back
        cudaStreamWaitEvent(stream_transfer, compute_done[slot], 0);
        cudaMemcpyAsync(h_ring[slot], d_ring[slot], BATCH_BYTES, cudaMemcpyDeviceToHost, stream_transfer);
    }

    // Single pipeline drain synchronization at the end!
    cudaStreamSynchronize(stream_transfer);
    cudaStreamSynchronize(stream_compute);

    auto tEndPipe = std::chrono::high_resolution_clock::now();
    double totalMsPipe = std::chrono::duration<double, std::milli>(tEndPipe - tStartPipe).count();
    std::cout << "    --> Total Pipelined Time: " << totalMsPipe << " ms (" 
              << (totalMsPipe / TOTAL_BATCHES) << " ms/batch)\n";
    std::cout << "    --> Effective Throughput Gain: " << std::fixed << std::setprecision(2) 
              << (totalMsSeq / totalMsPipe) << "x over sequential!\n";

    // Cleanup
    for (int i = 0; i < RING_STAGES; ++i) {
        cudaFreeHost(h_ring[i]);
        cudaFree(d_ring[i]);
        cudaEventDestroy(transfer_done[i]);
        cudaEventDestroy(compute_done[i]);
    }
    cudaStreamDestroy(stream_transfer);
    cudaStreamDestroy(stream_compute);

    std::cout << "\n=========================================================\n";
    std::cout << " [HPC RING PIPELINE PROBE COMPLETE]\n";
    std::cout << "=========================================================\n";
    return 0;
}
