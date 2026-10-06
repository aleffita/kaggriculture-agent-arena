#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <queue>
#include <algorithm>
#include <random>
#include <numeric>

// Metric Ball Bounding Volume in High-Dimensional Embedding Space
struct MetricBallNode {
    std::vector<float> centroid;
    float radius = 0.0f; // Max L2 distance from centroid to any vector in this subtree
    int start_idx = 0;
    int end_idx = 0;
    bool is_leaf = false;
    int left_child = -1;
    int right_child = -1;

    // Exact Cauchy-Schwarz Upper Bound: max_{v in Ball(c, R)} (q . v) = (q . c) + ||q||_2 * R
    inline float max_dot_product(const float* query, float norm_query, int dim) const {
        float dot_c = 0.0f;
        for (int i = 0; i < dim; ++i) {
            dot_c += query[i] * centroid[i];
        }
        return dot_c + norm_query * radius;
    }
};

class VocabMetricBVH {
public:
    int dim;
    int num_items;
    std::vector<float> embeddings; // contiguous [num_items x dim]
    std::vector<MetricBallNode> nodes;
    std::vector<int> item_indices;

    VocabMetricBVH(int n, int d) : num_items(n), dim(d) {
        embeddings.resize(n * d);
        item_indices.resize(n);
        std::iota(item_indices.begin(), item_indices.end(), 0);
    }

    void build_tree() {
        nodes.reserve(num_items * 2);
        build_recursive(0, num_items);
    }

    int build_recursive(int start, int end) {
        int node_idx = (int)nodes.size();
        nodes.emplace_back();
        MetricBallNode& node = nodes[node_idx];
        node.start_idx = start;
        node.end_idx = end;
        node.centroid.assign(dim, 0.0f);

        int count = end - start;
        if (count == 0) return -1;

        // 1. Compute Centroid
        for (int i = start; i < end; ++i) {
            int item = item_indices[i];
            const float* vec = &embeddings[item * dim];
            for (int d = 0; d < dim; ++d) {
                node.centroid[d] += vec[d];
            }
        }
        float inv_count = 1.0f / (float)count;
        for (int d = 0; d < dim; ++d) {
            node.centroid[d] *= inv_count;
        }

        // 2. Compute L2 Radius
        float max_r2 = 0.0f;
        int farthest_item = item_indices[start];
        for (int i = start; i < end; ++i) {
            int item = item_indices[i];
            const float* vec = &embeddings[item * dim];
            float r2 = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = vec[d] - node.centroid[d];
                r2 += diff * diff;
            }
            if (r2 > max_r2) {
                max_r2 = r2;
                farthest_item = item;
            }
        }
        node.radius = std::sqrt(max_r2);

        // Leaf threshold: 32 items
        if (count <= 32) {
            node.is_leaf = true;
            return node_idx;
        }

        node.is_leaf = false;

        // 3. 2-Means / Farthest-point Bisection Projection
        const float* pole_a = &embeddings[farthest_item * dim];
        float max_dist_pole = -1.0f;
        int pole_b_item = item_indices[start];
        for (int i = start; i < end; ++i) {
            int item = item_indices[i];
            const float* vec = &embeddings[item * dim];
            float dist = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = vec[d] - pole_a[d];
                dist += diff * diff;
            }
            if (dist > max_dist_pole) {
                max_dist_pole = dist;
                pole_b_item = item;
            }
        }
        const float* pole_b = &embeddings[pole_b_item * dim];

        // Split vector v_split = pole_a - pole_b
        std::vector<float> v_split(dim);
        for (int d = 0; d < dim; ++d) {
            v_split[d] = pole_a[d] - pole_b[d];
        }

        // Partition by median projection onto split vector
        int mid = (start + end) / 2;
        std::nth_element(item_indices.begin() + start,
                         item_indices.begin() + mid,
                         item_indices.begin() + end,
                         [&](int a, int b) {
                             float proj_a = 0.0f, proj_b = 0.0f;
                             const float* va = &embeddings[a * dim];
                             const float* vb = &embeddings[b * dim];
                             for (int d = 0; d < dim; ++d) {
                                 proj_a += va[d] * v_split[d];
                                 proj_b += vb[d] * v_split[d];
                             }
                             return proj_a < proj_b;
                         });

        int left = build_recursive(start, mid);
        int right = build_recursive(mid, end);
        nodes[node_idx].left_child = left;
        nodes[node_idx].right_child = right;
        return node_idx;
    }

    // Exact Top-K Search with Cauchy-Schwarz Ball Pruning
    void query_top_k(const float* query, int k, int& dot_products_evaluated, int& nodes_pruned, float& top1_score) {
        dot_products_evaluated = 0;
        nodes_pruned = 0;

        float norm_q2 = 0.0f;
        for (int d = 0; d < dim; ++d) norm_q2 += query[d] * query[d];
        float norm_query = std::sqrt(norm_q2);

        // Min-heap for k-th score threshold
        std::priority_queue<float, std::vector<float>, std::greater<float>> top_k_heap;

        // Priority queue for Branch-and-Bound: highest upper bound first
        auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        };
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> q(cmp);

        float root_bound = nodes[0].max_dot_product(query, norm_query, dim);
        q.push({root_bound, 0});

        while (!q.empty()) {
            auto [bound, node_idx] = q.top();
            q.pop();

            // Certified Cauchy-Schwarz Pruning Criterion:
            if ((int)top_k_heap.size() >= k && bound <= top_k_heap.top()) {
                nodes_pruned++;
                continue; // 100% Mathematically Proven: No vector in this ball can enter top-k!
            }

            const MetricBallNode& node = nodes[node_idx];
            if (node.is_leaf) {
                for (int i = node.start_idx; i < node.end_idx; ++i) {
                    int item = item_indices[i];
                    const float* vec = &embeddings[item * dim];
                    float dot = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        dot += query[d] * vec[d];
                    }
                    dot_products_evaluated++;

                    if ((int)top_k_heap.size() < k) {
                        top_k_heap.push(dot);
                    } else if (dot > top_k_heap.top()) {
                        top_k_heap.pop();
                        top_k_heap.push(dot);
                    }
                }
            } else {
                if (node.left_child != -1) {
                    float b_left = nodes[node.left_child].max_dot_product(query, norm_query, dim);
                    q.push({b_left, node.left_child});
                }
                if (node.right_child != -1) {
                    float b_right = nodes[node.right_child].max_dot_product(query, norm_query, dim);
                    q.push({b_right, node.right_child});
                }
            }
        }

        top1_score = top_k_heap.top();
    }
};

int main() {
    std::cout << "=========================================================\n";
    std::cout << " [PROBE P3] Certified Ball-Tree Vocabulary Pruning (MIPS)\n";
    std::cout << "=========================================================\n\n";

    const int VOCAB_SIZE = 65536;
    const int DIM = 256;
    const int TOP_K = 10;
    const int NUM_QUERIES = 20;
    const int NUM_CLUSTERS = 128; // Semantic topic clusters (NLP vocabulary manifold)

    std::cout << "[+] Synthesizing vocabulary with realistic semantic cluster manifold:\n"
              << "    Tokens: " << VOCAB_SIZE << ", Dimension: " << DIM << ", Clusters: " << NUM_CLUSTERS << "...\n";

    VocabMetricBVH bvh(VOCAB_SIZE, DIM);
    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    // 1. Generate cluster centroids on unit sphere
    std::vector<std::vector<float>> cluster_centers(NUM_CLUSTERS, std::vector<float>(DIM));
    for (int c = 0; c < NUM_CLUSTERS; ++c) {
        float norm = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            cluster_centers[c][d] = norm_dist(rng);
            norm += cluster_centers[c][d] * cluster_centers[c][d];
        }
        norm = std::sqrt(norm);
        for (int d = 0; d < DIM; ++d) cluster_centers[c][d] /= norm;
    }

    // 2. High-dimensional manifold: intra-cluster variance scaled by 1/sqrt(DIM)
    float sigma_intra = 0.15f / std::sqrt((float)DIM);
    float sigma_query = 0.05f / std::sqrt((float)DIM);

    int items_per_cluster = VOCAB_SIZE / NUM_CLUSTERS;
    for (int c = 0; c < NUM_CLUSTERS; ++c) {
        for (int j = 0; j < items_per_cluster; ++j) {
            int i = c * items_per_cluster + j;
            float norm = 0.0f;
            for (int d = 0; d < DIM; ++d) {
                float val = cluster_centers[c][d] + sigma_intra * norm_dist(rng);
                bvh.embeddings[i * DIM + d] = val;
                norm += val * val;
            }
            norm = std::sqrt(norm);
            for (int d = 0; d < DIM; ++d) {
                bvh.embeddings[i * DIM + d] /= norm;
            }
        }
    }

    auto t0_build = std::chrono::high_resolution_clock::now();
    bvh.build_tree();
    auto t1_build = std::chrono::high_resolution_clock::now();
    double ms_build = std::chrono::duration<double, std::milli>(t1_build - t0_build).count();
    std::cout << "[+] Ball-Tree built in " << ms_build << " ms (Total Nodes: " << bvh.nodes.size() << ")\n\n";

    // Run queries: query vectors aligned with semantic directions
    std::vector<float> query(DIM);
    double total_eval_time_ms = 0.0;
    int total_dots = 0;
    int total_pruned = 0;

    std::cout << "[*] Executing " << NUM_QUERIES << " Top-" << TOP_K << " MIPS queries via Certified Ball-Tree...\n";

    for (int q = 0; q < NUM_QUERIES; ++q) {
        int target_cluster = q % NUM_CLUSTERS;
        float norm = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            query[d] = cluster_centers[target_cluster][d] + sigma_query * norm_dist(rng);
            norm += query[d] * query[d];
        }
        norm = std::sqrt(norm);
        for (int d = 0; d < DIM; ++d) query[d] /= norm;

        int dots = 0;
        int pruned = 0;
        float top_score = 0.0f;

        auto t0_q = std::chrono::high_resolution_clock::now();
        bvh.query_top_k(query.data(), TOP_K, dots, pruned, top_score);
        auto t1_q = std::chrono::high_resolution_clock::now();

        total_eval_time_ms += std::chrono::duration<double, std::milli>(t1_q - t0_q).count();
        total_dots += dots;
        total_pruned += pruned;
    }

    double avg_ms = total_eval_time_ms / NUM_QUERIES;
    double avg_dots = (double)total_dots / NUM_QUERIES;
    double prune_ratio = (1.0 - (avg_dots / VOCAB_SIZE)) * 100.0;

    std::cout << "\n=========================================================\n";
    std::cout << " [RESULTADOS DA PROBE P3 (BALL-TREE CAUCHY-SCHWARZ)]\n";
    std::cout << "=========================================================\n";
    std::cout << "  Tamanho Total do Vocabulario:     " << VOCAB_SIZE << " tokens\n";
    std::cout << "  Dimensao Latente:                 " << DIM << "\n";
    std::cout << "  Numero de Topicos Semanticos:     " << NUM_CLUSTERS << "\n";
    std::cout << "  Media de Dot Products Calculados: " << (int)avg_dots << " / " << VOCAB_SIZE << "\n";
    std::cout << "  Taxa de Poda Certificada:         " << std::fixed << std::setprecision(2) << prune_ratio << "%\n";
    std::cout << "  Reducao Efetiva de FLOPs no Head: " << std::setprecision(2) << (VOCAB_SIZE / avg_dots) << "x\n";
    std::cout << "  Latencia Media por Query (CPU):   " << std::setprecision(3) << avg_ms << " ms\n";
    std::cout << "  Garantia Matematica:              100% EXATA (bounds Cauchy-Schwarz ||q|| * R)\n";
    std::cout << "=========================================================\n";

    return 0;
}
