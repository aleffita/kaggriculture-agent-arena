#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <algorithm>

// Suffix Automaton / Trie based Dynamic Transducer for Zero-Compute Drafting
class SuffixFstDrafter {
private:
    struct State {
        int len = 0;
        int link = -1;
        std::unordered_map<int, int> next;
    };

    std::vector<State> st;
    int sz = 1;
    int last = 0;
    std::vector<int> history;

public:
    SuffixFstDrafter() {
        st.resize(2);
        st[0].len = 0;
        st[0].link = -1;
        sz = 1;
        last = 0;
    }

    void extend(int token) {
        history.push_back(token);
        int cur = sz++;
        if (sz >= (int)st.size()) {
            st.resize(sz * 2);
        }
        st[cur].len = st[last].len + 1;
        int p = last;
        while (p != -1 && !st[p].next.count(token)) {
            st[p].next[token] = cur;
            p = st[p].link;
        }
        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[token];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int clone = sz++;
                if (sz >= (int)st.size()) {
                    st.resize(sz * 2);
                }
                st[clone].len = st[p].len + 1;
                st[clone].next = st[q].next;
                st[clone].link = st[q].link;
                while (p != -1 && st[p].next[token] == q) {
                    st[p].next[token] = clone;
                    p = st[p].link;
                }
                st[q].link = st[clone].link = clone;
            }
        }
        last = cur;
    }

    // Proposes draft tokens by walking the automaton from the current context
    std::vector<int> propose_draft(int max_draft_len, int match_window = 4) {
        if (history.empty()) return {};

        // Find the deepest state matching the recent suffix
        int cur = 0;
        int start_idx = std::max(0, (int)history.size() - match_window);
        for (int i = start_idx; i < (int)history.size(); ++i) {
            int tok = history[i];
            if (st[cur].next.count(tok)) {
                cur = st[cur].next[tok];
            } else {
                break;
            }
        }

        // Greedy path following from this state
        std::vector<int> draft;
        for (int step = 0; step < max_draft_len; ++step) {
            if (st[cur].next.empty()) break;
            // Pick the first available branch or transition
            auto it = st[cur].next.begin();
            draft.push_back(it->first);
            cur = it->second;
        }
        return draft;
    }

    size_t state_count() const { return sz; }
    size_t token_count() const { return history.size(); }
};

int main() {
    std::cout << "=========================================================\n";
    std::cout << " [PROBE P2] CPU Zero-Compute Suffix FST Transducer Draft\n";
    std::cout << "=========================================================\n\n";

    SuffixFstDrafter drafter;

    // Simulate ingesting a prompt of 512 tokens with recurring syntactic structures (like code/JSON)
    std::vector<int> synthetic_context;
    for (int i = 0; i < 512; ++i) {
        // Repeated cyclic patterns: e.g. "def foo(): return bar"
        int tok = (i % 23) + 100;
        synthetic_context.push_back(tok);
    }

    auto t0_ingest = std::chrono::high_resolution_clock::now();
    for (int tok : synthetic_context) {
        drafter.extend(tok);
    }
    auto t1_ingest = std::chrono::high_resolution_clock::now();
    double us_ingest = std::chrono::duration<double, std::micro>(t1_ingest - t0_ingest).count();

    std::cout << "[+] Ingested " << synthetic_context.size() << " prompt tokens into Suffix FST:\n"
              << "    Total Automaton States: " << drafter.state_count() << "\n"
              << "    Total Ingestion Time:   " << std::fixed << std::setprecision(2) << us_ingest << " us (" 
              << (us_ingest / synthetic_context.size()) << " us/token)\n\n";

    // Measure Draft Generation Latency over 10,000 queries
    const int NUM_QUERIES = 10000;
    const int MAX_DRAFT_LEN = 4;

    auto t0_draft = std::chrono::high_resolution_clock::now();
    size_t total_drafted_tokens = 0;
    for (int q = 0; q < NUM_QUERIES; ++q) {
        auto draft = drafter.propose_draft(MAX_DRAFT_LEN);
        total_drafted_tokens += draft.size();
    }
    auto t1_draft = std::chrono::high_resolution_clock::now();
    double us_draft = std::chrono::duration<double, std::micro>(t1_draft - t0_draft).count();

    std::cout << "[+] Benchmark: Draft Proposal Latency (" << NUM_QUERIES << " iterations):\n"
              << "    Average Proposal Latency: " << (us_draft / NUM_QUERIES) << " microseconds (us)\n"
              << "    Throughput:               " << (total_drafted_tokens / (us_draft / 1e6)) << " draft tokens/sec\n"
              << "    Avg Draft Chain Length:   " << ((double)total_drafted_tokens / NUM_QUERIES) << " tokens\n\n";

    // Acceptance Simulation
    // Suppose verification acceptance rate alpha = 60%
    double alpha = 0.60;
    double expected_tokens_per_step = 1.0;
    double p = 1.0;
    for (int k = 1; k <= MAX_DRAFT_LEN; ++k) {
        p *= alpha;
        expected_tokens_per_step += p;
    }

    std::cout << "[+] Mathematical Yield under Speculative Verifier:\n"
              << "    Assuming acceptance alpha = " << (alpha * 100.0) << "%:\n"
              << "    Expected Accepted Tokens/Step: " << std::setprecision(3) << expected_tokens_per_step << "x\n"
              << "    Draft Compute Cost on GPU:     0.00 FLOPS (100% on CPU FST in " 
              << (us_draft / NUM_QUERIES) << " us)\n";

    std::cout << "\n=========================================================\n";
    std::cout << " [PROBE P2 COMPLETED SUCCESSFULLY]\n";
    std::cout << "=========================================================\n";
    return 0;
}
