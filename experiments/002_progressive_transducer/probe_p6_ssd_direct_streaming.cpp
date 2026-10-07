#include <windows.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[-] CUDA Error: " << cudaGetErrorString(err) << "\n"; \
        exit(1); \
    } \
} while(0)

int main() {
    std::cout << "=========================================================\n";
    std::cout << " [PROBE P6] Direct SSD -> Host Pinned -> GPU VRAM Streaming\n";
    std::cout << " Testando Win32 Unbuffered Overlapped I/O + GPU DMA Overlap\n";
    std::cout << "=========================================================\n\n";

    const wchar_t* target_model_path = L"Z:\\models\\prism-ml\\Ternary-Bonsai-2-27B-gguf\\Ternary-Bonsai-2-27B-PTQ1_0.gguf";
    std::wcout << L"[+] Alvo de Teste (SSD Drive Z:):\n    " << target_model_path << L"\n\n";

    // 1. Open File with FILE_FLAG_NO_BUFFERING (Direct DMA bypass of OS cache)
    HANDLE hFile = CreateFileW(
        target_model_path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[-] Erro ao abrir arquivo no drive Z: " << GetLastError() << "\n";
        return 1;
    }

    LARGE_INTEGER file_size;
    GetFileSizeEx(hFile, &file_size);
    std::cout << "[+] Arquivo Aberto com Sucesso! Tamanho: " 
              << (file_size.QuadPart / (1024 * 1024)) << " MB ("
              << (file_size.QuadPart / (1024 * 1024 * 1024.0)) << " GB)\n";

    // Initialize GPU 0 (RTX 2060)
    int target_device = 0;
    CHECK_CUDA(cudaSetDevice(target_device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, target_device));
    std::cout << "[+] Dispositivo GPU de Destino: " << prop.name << " (DMA Copy Engines: " 
              << prop.asyncEngineCount << ")\n";

    // Setup Streaming Buffers
    const size_t CHUNK_SIZE = 64 * 1024 * 1024; // 64 MB chunks (setor-alinhado)
    const int NUM_CHUNKS = 8;                   // Total: 512 MB testados
    const int RING_STAGES = 2;                  // Double buffering

    void* h_pinned_ring[RING_STAGES];
    void* d_vram_ring[RING_STAGES];
    cudaStream_t dma_stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&dma_stream, cudaStreamNonBlocking));

    for (int i = 0; i < RING_STAGES; ++i) {
        // Aloca memoria host pinned alinhada para FILE_FLAG_NO_BUFFERING (multiplo de 4096 bytes)
        CHECK_CUDA(cudaHostAlloc(&h_pinned_ring[i], CHUNK_SIZE, cudaHostAllocPortable));
        CHECK_CUDA(cudaMalloc(&d_vram_ring[i], CHUNK_SIZE));
    }

    std::cout << "[+] Buffers Alocados: " << RING_STAGES << " slots de " 
              << (CHUNK_SIZE / (1024*1024)) << " MB em Pinned Host + VRAM.\n\n";

    OVERLAPPED ov[RING_STAGES];
    HANDLE events[RING_STAGES];
    for (int i = 0; i < RING_STAGES; ++i) {
        memset(&ov[i], 0, sizeof(OVERLAPPED));
        events[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
        ov[i].hEvent = events[i];
    }

    std::cout << "[*] Executando Streaming Fim-a-Fim (SSD -> RAM Pinned -> GPU VRAM)...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < NUM_CHUNKS; ++c) {
        int slot = c % RING_STAGES;
        uint64_t file_offset = (uint64_t)c * CHUNK_SIZE;

        ov[slot].Offset = (DWORD)(file_offset & 0xFFFFFFFF);
        ov[slot].OffsetHigh = (DWORD)(file_offset >> 32);
        ResetEvent(events[slot]);

        // 1. Dispara leitura assíncrona direta do SSD NVMe para a memória Pinned (sem cópia do SO)
        DWORD bytes_read = 0;
        BOOL read_ok = ReadFile(hFile, h_pinned_ring[slot], (DWORD)CHUNK_SIZE, &bytes_read, &ov[slot]);
        if (!read_ok && GetLastError() != ERROR_IO_PENDING) {
            std::cerr << "[-] Erro no ReadFile: " << GetLastError() << "\n";
            break;
        }

        // Aguarda conclusao do bloco NVMe
        WaitForSingleObject(events[slot], INFINITE);

        // 2. Dispara DMA assíncrono Pinned Host -> VRAM GPU
        CHECK_CUDA(cudaMemcpyAsync(d_vram_ring[slot], h_pinned_ring[slot], CHUNK_SIZE, cudaMemcpyHostToDevice, dma_stream));
        CHECK_CUDA(cudaStreamSynchronize(dma_stream));

        std::cout << "  [Chunk " << c << "] " << (CHUNK_SIZE / (1024*1024)) 
                  << " MB transferidos (Offset: " << (file_offset / (1024*1024)) << " MB)\n";
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double total_bytes_gb = (double)(NUM_CHUNKS * CHUNK_SIZE) / (1024.0 * 1024.0 * 1024.0);
    double effective_bw_gbs = total_bytes_gb / (total_ms / 1000.0);

    std::cout << "\n=========================================================\n";
    std::cout << " [RESULTADOS DA PROBE P6 (STREAMING DIRETO)]\n";
    std::cout << "=========================================================\n";
    std::cout << "  Dados Totais Lidos do SSD:      " << (NUM_CHUNKS * CHUNK_SIZE / (1024*1024)) << " MB\n";
    std::cout << "  Tempo Total de Streaming:       " << total_ms << " ms\n";
    std::cout << "  Largura de Banda Efetiva SSD->GPU: " << std::fixed << std::setprecision(2) << effective_bw_gbs << " GB/s\n";
    std::cout << "  Vazao em Parametros 1.58b:      " << (effective_bw_gbs * 8.0 / 1.75) << " Bilias de pesos/seg\n";
    std::cout << "  Bypass de Cache do Windows:     100% ATIVO (FILE_FLAG_NO_BUFFERING)\n";
    std::cout << "=========================================================\n";

    // Cleanup
    for (int i = 0; i < RING_STAGES; ++i) {
        cudaFreeHost(h_pinned_ring[i]);
        cudaFree(d_vram_ring[i]);
        CloseHandle(events[i]);
    }
    cudaStreamDestroy(dma_stream);
    CloseHandle(hFile);

    return 0;
}
