#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[-] CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// AABB (Axis-Aligned Bounding Box) in 3D projection space
struct AABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    __host__ __device__ bool contains(float x, float y, float z) const {
        return (x >= min_x && x <= max_x &&
                y >= min_y && y <= max_y &&
                z >= min_z && z <= max_z);
    }

    __host__ __device__ float sq_dist_to_point(float x, float y, float z) const {
        float dx = 0.0f;
        if (x < min_x) dx = min_x - x;
        else if (x > max_x) dx = x - max_x;

        float dy = 0.0f;
        if (y < min_y) dy = min_y - y;
        else if (y > max_y) dy = y - max_y;

        float dz = 0.0f;
        if (z < min_z) dz = min_z - z;
        else if (z > max_z) dz = z - max_z;

        return dx * dx + dy * dy + dz * dz;
    }
};

struct BVHNode {
    AABB box;
    int left_child;   // -1 if leaf
    int right_child;  // -1 if leaf
    int expert_id;    // valid if leaf, -1 otherwise
};

// Projection matrix from d=2880 to 3D
__constant__ float c_proj[3 * 2880];

// Kernel 1: Standard Brute-Force MoE Router (GEMV top-k)
__global__ void brute_force_router_kernel(
    const float* __restrict__ d_tokens,      // [batch_size, hidden_dim]
    const float* __restrict__ d_centroids,   // [num_experts, hidden_dim]
    int* __restrict__ d_topk_experts,        // [batch_size, top_k]
    float* __restrict__ d_topk_scores,       // [batch_size, top_k]
    int hidden_dim, int num_experts, int top_k, int batch_size)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const float* token = d_tokens + b * hidden_dim;

    // Calculo dos scores para cada especialista
    float scores[64]; // max 64 experts in local register
    if (num_experts > 64) return;

    for (int e = 0; e < num_experts; ++e) {
        float dot = 0.0f;
        const float* cent = d_centroids + e * hidden_dim;
        #pragma unroll 4
        for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
            dot += token[d] * cent[d];
        }
        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (threadIdx.x == 0) {
            scores[e] = dot;
        }
    }

    __syncthreads();

    // Encontrar Top-K
    if (threadIdx.x == 0) {
        for (int k = 0; k < top_k; ++k) {
            float max_val = -1e9f;
            int max_idx = 0;
            for (int e = 0; e < num_experts; ++e) {
                if (scores[e] > max_val) {
                    max_val = scores[e];
                    max_idx = e;
                }
            }
            d_topk_experts[b * top_k + k] = max_idx;
            d_topk_scores[b * top_k + k] = max_val;
            scores[max_idx] = -1e9f; // marcar como visitado
        }
    }
}

// Kernel 2: BVH-Accelerated Spatial Router (RT Core simulation)
// Realiza projeção 3D e travessia da árvore AABB hierárquica
__global__ void bvh_accelerated_router_kernel(
    const float* __restrict__ d_tokens,       // [batch_size, hidden_dim]
    const BVHNode* __restrict__ d_bvh_nodes,  // nós da BVH
    const float* __restrict__ d_centroids,    // [num_experts, hidden_dim]
    int* __restrict__ d_topk_experts,         // [batch_size, top_k]
    float* __restrict__ d_topk_scores,        // [batch_size, top_k]
    int hidden_dim, int num_experts, int top_k, int batch_size)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const float* token = d_tokens + b * hidden_dim;

    // 1. Projeção rápida para 3D (Warp-level dot product com 3 vetores base)
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        px += token[d] * c_proj[0 * hidden_dim + d];
        py += token[d] * c_proj[1 * hidden_dim + d];
        pz += token[d] * c_proj[2 * hidden_dim + d];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        px += __shfl_down_sync(0xffffffff, px, offset);
        py += __shfl_down_sync(0xffffffff, py, offset);
        pz += __shfl_down_sync(0xffffffff, pz, offset);
    }

    __shared__ float s_px, s_py, s_pz;
    if (threadIdx.x == 0) {
        s_px = px; s_py = py; s_pz = pz;
    }
    __syncthreads();

    // 2. Travessia BVH de pilha curta para podar especialistas (1 thread por warp ou cooperativa)
    __shared__ int candidate_experts[16];
    __shared__ int candidate_count;

    if (threadIdx.x == 0) {
        candidate_count = 0;
        int stack[32];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; // Raiz da BVH

        while (stack_ptr > 0 && candidate_count < 12) {
            int node_idx = stack[--stack_ptr];
            const BVHNode& node = d_bvh_nodes[node_idx];

            if (node.expert_id >= 0) {
                // Folha encontrada!
                candidate_experts[candidate_count++] = node.expert_id;
            } else {
                // Teste de intersecção/proximidade da AABB (simulando RT Core Box Test)
                float d_left = d_bvh_nodes[node.left_child].box.sq_dist_to_point(s_px, s_py, s_pz);
                float d_right = d_bvh_nodes[node.right_child].box.sq_dist_to_point(s_px, s_py, s_pz);

                // Push no stack ordenado por proximidade
                if (d_left < d_right) {
                    stack[stack_ptr++] = node.right_child;
                    stack[stack_ptr++] = node.left_child;
                } else {
                    stack[stack_ptr++] = node.left_child;
                    stack[stack_ptr++] = node.right_child;
                }
            }
        }
    }
    __syncthreads();

    // 3. Avaliar dot-product exato APENAS para os candidatos podados pela BVH (Refinamento)
    int n_cands = candidate_count;
    float scores[16];
    for (int c = 0; c < n_cands; ++c) {
        int e = candidate_experts[c];
        float dot = 0.0f;
        const float* cent = d_centroids + e * hidden_dim;
        for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
            dot += token[d] * cent[d];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (threadIdx.x == 0) {
            scores[c] = dot;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < top_k; ++k) {
            float max_val = -1e9f;
            int best_cand = 0;
            for (int c = 0; c < n_cands; ++c) {
                if (scores[c] > max_val) {
                    max_val = scores[c];
                    best_cand = c;
                }
            }
            d_topk_experts[b * top_k + k] = candidate_experts[best_cand];
            d_topk_scores[b * top_k + k] = max_val;
            scores[best_cand] = -1e9f;
        }
    }
}

// ---------------------------------------------------------------------------
// Hardware Inspection: DirectX 12 Raytracing Tier Check
// ---------------------------------------------------------------------------
void inspect_d3d12_raytracing() {
    printf("[+] ========================================================================\n");
    printf("[+]  SONDAGEM DE HARDWARE D3D12 RAYTRACING / RT CORES (DirectX 12 API)      \n");
    printf("[+] ========================================================================\n");

    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        printf("[-] Falha ao instanciar IDXGIFactory6\n");
        return;
    }

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        wprintf(L"    Adaptador DXGI %u: %s\n", i, desc.Description);
        printf("    VRAM Dedicada: %.2f GB\n", desc.DedicatedVideoMemory / (1024.0 * 1024.0 * 1024.0));

        ComPtr<ID3D12Device> device;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
            if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5)))) {
                const char* tier_str = "Não Suportado";
                if (opts5.RaytracingTier == D3D12_RAYTRACING_TIER_1_0) tier_str = "Tier 1.0 (DirectX Raytracing Básico / DXR)";
                else if (opts5.RaytracingTier == D3D12_RAYTRACING_TIER_1_1) tier_str = "Tier 1.1 (Hardware Nativo Completo / RT Cores + Inline Ray Tracing)";
                printf("    Suporte D3D12 Raytracing: %s\n", tier_str);
            } else {
                printf("    Suporte D3D12 Raytracing: Consulta não suportada pelo driver.\n");
            }
        } else {
            printf("    [-] Não foi possível criar ID3D12Device para este adaptador.\n");
        }
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// Construction of BVH Tree for MoE Centroids
// ---------------------------------------------------------------------------
void build_bvh_tree(const std::vector<float>& proj_centroids, int num_experts, std::vector<BVHNode>& bvh_nodes) {
    // Nós folha
    bvh_nodes.clear();
    bvh_nodes.resize(2 * num_experts - 1);

    int next_free = num_experts;

    // Inicializar folhas (índices 0 .. num_experts-1)
    for (int e = 0; e < num_experts; ++e) {
        float x = proj_centroids[e * 3 + 0];
        float y = proj_centroids[e * 3 + 1];
        float z = proj_centroids[e * 3 + 2];
        float pad = 0.05f;
        bvh_nodes[e].box = { x - pad, y - pad, z - pad, x + pad, y + pad, z + pad };
        bvh_nodes[e].left_child = -1;
        bvh_nodes[e].right_child = -1;
        bvh_nodes[e].expert_id = e;
    }

    // Agrupamento hierárquico simples (Bottom-up agglomerative clustering)
    std::vector<int> active_nodes(num_experts);
    for (int i = 0; i < num_experts; ++i) active_nodes[i] = i;

    while (active_nodes.size() > 1) {
        float min_dist = 1e9f;
        int best_i = 0, best_j = 1;
        for (size_t i = 0; i < active_nodes.size(); ++i) {
            for (size_t j = i + 1; j < active_nodes.size(); ++j) {
                int ni = active_nodes[i];
                int nj = active_nodes[j];
                float dx = (bvh_nodes[ni].box.min_x + bvh_nodes[ni].box.max_x) * 0.5f - (bvh_nodes[nj].box.min_x + bvh_nodes[nj].box.max_x) * 0.5f;
                float dy = (bvh_nodes[ni].box.min_y + bvh_nodes[ni].box.max_y) * 0.5f - (bvh_nodes[nj].box.min_y + bvh_nodes[nj].box.max_y) * 0.5f;
                float dz = (bvh_nodes[ni].box.min_z + bvh_nodes[ni].box.max_z) * 0.5f - (bvh_nodes[nj].box.min_z + bvh_nodes[nj].box.max_z) * 0.5f;
                float d = dx * dx + dy * dy + dz * dz;
                if (d < min_dist) {
                    min_dist = d;
                    best_i = (int)i;
                    best_j = (int)j;
                }
            }
        }

        int parent_idx = next_free++;
        int left = active_nodes[best_i];
        int right = active_nodes[best_j];

        bvh_nodes[parent_idx].left_child = left;
        bvh_nodes[parent_idx].right_child = right;
        bvh_nodes[parent_idx].expert_id = -1;
        bvh_nodes[parent_idx].box = {
            std::min(bvh_nodes[left].box.min_x, bvh_nodes[right].box.min_x),
            std::min(bvh_nodes[left].box.min_y, bvh_nodes[right].box.min_y),
            std::min(bvh_nodes[left].box.min_z, bvh_nodes[right].box.min_z),
            std::max(bvh_nodes[left].box.max_x, bvh_nodes[right].box.max_x),
            std::max(bvh_nodes[left].box.max_y, bvh_nodes[right].box.max_y),
            std::max(bvh_nodes[left].box.max_z, bvh_nodes[right].box.max_z)
        };

        // Atualizar ativos
        active_nodes.erase(active_nodes.begin() + std::max(best_i, best_j));
        active_nodes.erase(active_nodes.begin() + std::min(best_i, best_j));
        active_nodes.push_back(parent_idx);
    }

    // A raiz é o último nó criado (trocar com o nó 0 para conveniência de travessia)
    int root_idx = next_free - 1;
    BVHNode root_node = bvh_nodes[root_idx];
    BVHNode orig_0 = bvh_nodes[0];
    bvh_nodes[0] = root_node;
    bvh_nodes[root_idx] = orig_0;
}

int main() {
    inspect_d3d12_raytracing();

    printf("[+] ========================================================================\n");
    printf("[+]  EXPERIMENTO: PODA GEOMÉTRICA DE MoE VIA ÁRVORES BVH vs BRUTE-FORCE     \n");
    printf("[+]  Topologia: GPT-OSS-20B (32 Especialistas, d = 2880, Top-4 Roteamento)  \n");
    printf("[+] ========================================================================\n\n");

    const int HIDDEN_DIM = 2880;
    const int NUM_EXPERTS = 32;
    const int TOP_K = 4;
    const int BATCH_SIZE = 128; // Teste simultâneo de prefill (128 tokens) e decode (1 token)

    // Configurar GPU 0 (RTX 2060 com SM 7.5 Turing)
    CHECK_CUDA(cudaSetDevice(0));

    // Gerar centroides sintéticos consistentes
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)HIDDEN_DIM));

    std::vector<float> h_centroids(NUM_EXPERTS * HIDDEN_DIM);
    for (auto& v : h_centroids) v = dist(rng);

    // Gerar matriz de projeção 3D ortogonal aleatória
    std::vector<float> h_proj(3 * HIDDEN_DIM);
    for (auto& v : h_proj) v = dist(rng);
    CHECK_CUDA(cudaMemcpyToSymbol(c_proj, h_proj.data(), 3 * HIDDEN_DIM * sizeof(float)));

    // Calcular coordenadas projetadas dos centroides em 3D
    std::vector<float> proj_centroids(NUM_EXPERTS * 3);
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        for (int d = 0; d < HIDDEN_DIM; ++d) {
            px += h_centroids[e * HIDDEN_DIM + d] * h_proj[0 * HIDDEN_DIM + d];
            py += h_centroids[e * HIDDEN_DIM + d] * h_proj[1 * HIDDEN_DIM + d];
            pz += h_centroids[e * HIDDEN_DIM + d] * h_proj[2 * HIDDEN_DIM + d];
        }
        proj_centroids[e * 3 + 0] = px;
        proj_centroids[e * 3 + 1] = py;
        proj_centroids[e * 3 + 2] = pz;
    }

    // Construir árvore BVH
    std::vector<BVHNode> h_bvh_nodes;
    build_bvh_tree(proj_centroids, NUM_EXPERTS, h_bvh_nodes);

    // Alocar recursos na GPU
    float* d_centroids = nullptr;
    BVHNode* d_bvh_nodes = nullptr;
    float* d_tokens = nullptr;
    int* d_topk_bf = nullptr;
    float* d_scores_bf = nullptr;
    int* d_topk_bvh = nullptr;
    float* d_scores_bvh = nullptr;

    CHECK_CUDA(cudaMalloc(&d_centroids, NUM_EXPERTS * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bvh_nodes, h_bvh_nodes.size() * sizeof(BVHNode)));
    CHECK_CUDA(cudaMalloc(&d_tokens, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_bf, BATCH_SIZE * TOP_K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_scores_bf, BATCH_SIZE * TOP_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_bvh, BATCH_SIZE * TOP_K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_scores_bvh, BATCH_SIZE * TOP_K * sizeof(float)));

    // Upload dos pesos e nós
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), NUM_EXPERTS * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bvh_nodes, h_bvh_nodes.data(), h_bvh_nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

    // Gerar tokens de teste
    std::vector<float> h_tokens(BATCH_SIZE * HIDDEN_DIM);
    for (auto& v : h_tokens) v = dist(rng);
    CHECK_CUDA(cudaMemcpy(d_tokens, h_tokens.data(), BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    const int WARMUP = 10;
    const int ITERS = 200;

    // -----------------------------------------------------------------------
    // TESTE A: BRUTE FORCE GEMV ROUTER
    // -----------------------------------------------------------------------
    for (int i = 0; i < WARMUP; ++i) {
        brute_force_router_kernel<<<BATCH_SIZE, 256>>>(d_tokens, d_centroids, d_topk_bf, d_scores_bf, HIDDEN_DIM, NUM_EXPERTS, TOP_K, BATCH_SIZE);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(ev_start));
    for (int i = 0; i < ITERS; ++i) {
        brute_force_router_kernel<<<BATCH_SIZE, 256>>>(d_tokens, d_centroids, d_topk_bf, d_scores_bf, HIDDEN_DIM, NUM_EXPERTS, TOP_K, BATCH_SIZE);
    }
    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float ms_bf = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_bf, ev_start, ev_stop));
    ms_bf /= ITERS;

    // -----------------------------------------------------------------------
    // TESTE B: BVH-ACCELERATED SPATIAL ROUTER
    // -----------------------------------------------------------------------
    for (int i = 0; i < WARMUP; ++i) {
        bvh_accelerated_router_kernel<<<BATCH_SIZE, 256>>>(d_tokens, d_bvh_nodes, d_centroids, d_topk_bvh, d_scores_bvh, HIDDEN_DIM, NUM_EXPERTS, TOP_K, BATCH_SIZE);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(ev_start));
    for (int i = 0; i < ITERS; ++i) {
        bvh_accelerated_router_kernel<<<BATCH_SIZE, 256>>>(d_tokens, d_bvh_nodes, d_centroids, d_topk_bvh, d_scores_bvh, HIDDEN_DIM, NUM_EXPERTS, TOP_K, BATCH_SIZE);
    }
    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float ms_bvh = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_bvh, ev_start, ev_stop));
    ms_bvh /= ITERS;

    // Verificar fidelidade top-k
    std::vector<int> res_bf(BATCH_SIZE * TOP_K);
    std::vector<int> res_bvh(BATCH_SIZE * TOP_K);
    CHECK_CUDA(cudaMemcpy(res_bf.data(), d_topk_bf, BATCH_SIZE * TOP_K * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(res_bvh.data(), d_topk_bvh, BATCH_SIZE * TOP_K * sizeof(int), cudaMemcpyDeviceToHost));

    int matched_top1 = 0;
    int matched_any_k = 0;
    for (int b = 0; b < BATCH_SIZE; ++b) {
        int best_bf = res_bf[b * TOP_K + 0];
        int best_bvh = res_bvh[b * TOP_K + 0];
        if (best_bf == best_bvh) matched_top1++;

        for (int k = 0; k < TOP_K; ++k) {
            int cand = res_bvh[b * TOP_K + k];
            for (int k2 = 0; k2 < TOP_K; ++k2) {
                if (cand == res_bf[b * TOP_K + k2]) {
                    matched_any_k++;
                    break;
                }
            }
        }
    }

    float top1_acc = (float)matched_top1 / BATCH_SIZE * 100.0f;
    float topk_acc = (float)matched_any_k / (BATCH_SIZE * TOP_K) * 100.0f;
    float speedup = ms_bf / ms_bvh;

    printf("========================================================================\n");
    printf(" RESULTADOS COMPARATIVOS: BRUTE FORCE vs BVH SPATIAL ROUTER             \n");
    printf("========================================================================\n");
    printf("  Batch Size (Tokens):            %d tokens\n", BATCH_SIZE);
    printf("  Roteamento Brute-Force (GEMV):  %.4f ms (Vazão: %.2f k-tokens/s)\n", ms_bf, (BATCH_SIZE / ms_bf));
    printf("  Roteamento BVH Espacial:        %.4f ms (Vazão: %.2f k-tokens/s)\n", ms_bvh, (BATCH_SIZE / ms_bvh));
    printf("  Fator de Aceleração (Speedup):  %.2fx de Velocidade no Roteador\n", speedup);
    printf("  Fidelidade Top-1 Exata:         %.2f%%\n", top1_acc);
    printf("  Retenção Top-4 Conjunta:        %.2f%%\n", topk_acc);
    printf("  Poda de Especialistas:          De 32 avaliações para 12 avaliações (62.5%% de corte!)\n");
    printf("========================================================================\n\n");

    // Limpeza
    cudaFree(d_centroids);
    cudaFree(d_bvh_nodes);
    cudaFree(d_tokens);
    cudaFree(d_topk_bf);
    cudaFree(d_scores_bf);
    cudaFree(d_topk_bvh);
    cudaFree(d_scores_bvh);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
