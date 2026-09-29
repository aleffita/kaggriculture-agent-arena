# Experiment 006: KV Cache Management & TurboQuant Research

- **Target Device**: NVIDIA GeForce GTX 1050 Ti (4GB GDDR5)
- **Model**: `gemma-4-E2B-it.litertlm`
- **Runtime**: WebGPU via Google Dawn (Direct3D 12)
- **Date**: 2026-09-29

---

## 1. The KV Cache Memory Wall

In transformer inference, Key-Value pairs for all preceding tokens must be cached across layers to compute self-attention without recomputing past states:

$$\text{KV Memory (Bytes)} = 2 \times L \times H_{\text{kv}} \times d_{\text{head}} \times \text{bytes\_per\_elem} \times S$$

For Gemma-4-E2B ($L=18, H_{\text{kv}}=1, d=256$, FP16 2 bytes):
- $18 \text{ KB per token}$.
- At 4096 tokens $\approx 73.7 \text{ MB}$.
- For larger models (e.g. Gemma 4 26B with $L=46, H_{\text{kv}}=8, d=128$), KV cache grows to **~188 KB per token** ($\approx 770 \text{ MB}$ at 4K context, and $>6 \text{ GB}$ at 32K context), creating an immediate memory wall for a 4GB card like the GTX 1050 Ti.

---

## 2. TurboQuant Mathematical Foundation (Google Research 2026)

TurboQuant introduces an online, data-oblivious vector quantization scheme designed specifically for KV cache compression:

1. **PolarQuant**:
   - Multiplies high-dimensional Key/Value vectors by a random orthogonal matrix $R \in \mathbb{R}^{d \times d}$.
   - By the Johnson-Lindenstrauss lemma and isotropic distribution, this flattens dimensional variance and kurtosis, making simple uniform scalar quantizers asymptotically optimal in mean-squared error (MSE).
2. **QJL (Quantized Johnson-Lindenstrauss)**:
   - Evaluates residual errors $e = x - \hat{x}$.
   - Projects $e$ through a 1-bit sketching matrix $S \in \{-1, +1\}^{m \times d}$, preserving inner products $\langle q, k \rangle$ with zero bias:
     $$\mathbb{E}[\langle \hat{q}, \hat{k} \rangle] = \langle q, k \rangle$$
3. **Compression Ratio**:
   - Reduces KV cache storage from 16-bit to **4-bit (4.0x compression)** or **3-bit (5.33x compression)** with less than 0.1 degradation in perplexity.

---

## 3. Applicability to GTX 1050 Ti

On Pascal GP107, memory capacity (4096 MB) is the primary constraint preventing large-context execution:
- With TurboQuant, a 16K or 32K context window fits comfortably within the 1.2 GB available headroom on the GTX 1050 Ti.
- While LiteRT-LM does not yet natively incorporate TurboQuant in its WebGPU shader kernels (it currently uses `use_ringbuffers_local_attention` to drop older context), a custom WebGPU HLSL shader or custom CUDA kernel implementing PolarQuant + QJL is an exceptional research direction for this repository.

---

## 4. How to Run the Probe

```powershell
uv run python experiments/006_kv_cache_and_turboquant/run_probe.py
```
