#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " -> " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << " -> " \
                  << status << std::endl; \
        exit(1); \
    } \
} while(0)

void run_precision_roofline(cublasHandle_t handle, int device_id, const std::string& prec_name, 
                            cudaDataType_t a_type, cudaDataType_t b_type, cudaDataType_t c_type,
                            cublasComputeType_t compute_type, size_t element_size,
                            int N, int K, const std::vector<int>& k_batches) {
    std::cout << "\n---------------------------------------------------------\n";
    std::cout << " [Precision Mode: " << prec_name << " (Weights: " << K << "x" << N << ")]\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "  k (Draft) | Latency (us) | Marginal Cost | Bandwidth (GB/s) | TFLOPs | Equiv. tok/s\n";
    std::cout << " -----------+--------------+---------------+------------------+--------+--------------\n";

    // Allocate Weight Matrix W [K x N]
    size_t w_bytes = (size_t)K * N * element_size;
    void *d_W = nullptr;
    CHECK_CUDA(cudaMalloc(&d_W, w_bytes));
    CHECK_CUDA(cudaMemset(d_W, 0x3F, w_bytes));

    int max_k = k_batches.back();
    size_t in_bytes = (size_t)max_k * K * element_size;
    size_t out_bytes = (size_t)max_k * N * element_size;

    void *d_X = nullptr;
    void *d_Y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_X, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_Y, out_bytes));
    CHECK_CUDA(cudaMemset(d_X, 0x3C, in_bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float time_k1 = 0.0f;

    for (int k : k_batches) {
        // Warmup
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Perform GEMM: Y = X * W  => [k x N] = [k x K] * [K x N]
        // Note: cuBLAS is column-major. In row-major: C = A * B -> C^T = B^T * A^T
        // B^T is [N x K], A^T is [K x k], C^T is [N x k]
        // ld_b = N, ld_a = K, ld_c = N
        for (int w = 0; w < 10; ++w) {
            CHECK_CUBLAS(cublasGemmEx(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      N, k, K,
                                      &alpha,
                                      d_W, a_type, N,
                                      d_X, b_type, K,
                                      &beta,
                                      d_Y, c_type, N,
                                      compute_type,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        const int ITERS = 100;
        CHECK_CUDA(cudaEventRecord(start));
        for (int it = 0; it < ITERS; ++it) {
            cublasGemmEx(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         N, k, K,
                         &alpha,
                         d_W, a_type, N,
                         d_X, b_type, K,
                         &beta,
                         d_Y, c_type, N,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_us = (total_ms / (float)ITERS) * 1000.0f;

        if (k == 1) time_k1 = avg_us;
        float marginal = ((avg_us - time_k1) / time_k1) * 100.0f;

        // Memory Traffic = W + X + Y
        double mem_traffic_bytes = (double)K * N * element_size + 
                                   (double)k * (K + N) * element_size;
        double bw_gbs = (mem_traffic_bytes / (avg_us * 1e-6)) / 1e9;

        // FLOPs = 2 * k * N * K
        double flops = 2.0 * (double)k * N * K;
        double tflops = (flops / (avg_us * 1e-6)) / 1e12;
        double tok_sec = ((double)k / (avg_us * 1e-6));

        std::cout << "  k = " << std::setw(3) << k << "   | "
                  << std::fixed << std::setprecision(1) << std::setw(7) << avg_us << " us | "
                  << std::showpos << std::setprecision(1) << std::setw(6) << marginal << "% " << std::noshowpos << "| "
                  << std::setprecision(1) << std::setw(8) << bw_gbs << " GB/s | "
                  << std::setprecision(2) << std::setw(6) << tflops << " | "
                  << std::setprecision(0) << std::setw(8) << tok_sec << "\n";
    }

    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << " [PROBE P4] Roofline Empirico de Transducao Especulativa\n";
    std::cout << " Determina k* e o Custo Marginal de Rascunho no Silicio\n";
    std::cout << "=========================================================\n";

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        std::cerr << "[-] Nenhum dispositivo CUDA detectado!\n";
        return 1;
    }

    // Explicitly target Device 0 (RTX 2060)
    int device_id = 0;
    CHECK_CUDA(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "\n[+] Alvo de Teste:\n";
    std::cout << "    Dispositivo:        " << prop.name << "\n";
    std::cout << "    Arquitetura:        sm_" << prop.major << prop.minor 
              << " (" << (prop.major == 7 && prop.minor == 5 ? "Turing" : "Outra") << ")\n";
    std::cout << "    Multiprocessadores: " << prop.multiProcessorCount << " SMs\n";
    std::cout << "    Largura de Banda:   " << (prop.memoryBusWidth) << " bits (~336 GB/s GDDR6)\n";
    std::cout << "    Memoria Global:     " << (prop.totalGlobalMem / (1024*1024)) << " MB\n\n";

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Gemma 4 E2B hidden dimension: 2560 x 2560 projection
    const int N = 2560;
    const int K = 2560;
    const std::vector<int> k_batches = {1, 2, 4, 8, 16, 32, 64};

    // 1. FP16 Tensor Cores
    run_precision_roofline(handle, device_id, "FP16 (Tensor Cores Turing)",
                           CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                           CUBLAS_COMPUTE_16F, sizeof(uint16_t),
                           N, K, k_batches);

    // 2. FP32 Standard CUDA Cores
    run_precision_roofline(handle, device_id, "FP32 (CUDA Cores Standard)",
                           CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                           CUBLAS_COMPUTE_32F, sizeof(float),
                           N, K, k_batches);

    // 3. INT8 Tensor Cores (IMMA)
    run_precision_roofline(handle, device_id, "INT8 (IMMA Tensor Cores / RNS Base)",
                           CUDA_R_8I, CUDA_R_8I, CUDA_R_32I,
                           CUBLAS_COMPUTE_32I, sizeof(int8_t),
                           N, K, k_batches);

    CHECK_CUBLAS(cublasDestroy(handle));

    std::cout << "\n=========================================================\n";
    std::cout << " [CONCLUSAO EMPIRICA DA PROBE P4]\n";
    std::cout << " O ponto onde o custo marginal permanece < 10% define o\n";
    std::cout << " tamanho ideal da arvore especulativa (k*) no silicio.\n";
    std::cout << "=========================================================\n";

    return 0;
}
