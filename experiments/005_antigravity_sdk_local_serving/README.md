# Experiment 005: Google Antigravity SDK & Local Serving Architecture

- **Target Device**: NVIDIA GeForce GTX 1050 Ti (4GB GDDR5)
- **Model**: `gemma-4-E2B-it.litertlm`
- **SDK**: `google-antigravity` (v0.1.20)
- **Serving Protocol**: OpenAI-Compatible `/v1/chat/completions` (JSON & SSE streaming)
- **Date**: 2026-09-29

---

## 1. Google Antigravity SDK Local Model Integration

Announced September 23, 2026, the **Google Antigravity SDK** introduces native support for offline, on-device agent execution:
- **`LiteRTAgentConfig`**: Runs models via the embedded LiteRT-LM runtime directly on the user's hardware.
- **`LocalOpenAIAgentConfig`**: Routes agent requests to any local OpenAI-compatible inference server (Ollama, vLLM, LM Studio, or custom LiteRT servers).
- **Hybrid Workflows**: Allows mixing local on-device models for privacy-sensitive or high-frequency tasks with cloud models (e.g. Gemini 2.5 Pro) for heavy multi-step planning.

---

## 2. Multi-User Concurrency & Thread Safety

The C++ LiteRT execution engine is non-reentrant during an ongoing generation turn. To safely expose the model as an HTTP server:
1. `LiteRTOpenAIServer` uses an internal synchronization lock (`with server.engine_lock:`).
2. When multiple clients connect concurrently, their requests are serialized in the HTTP worker pool.
3. Once a generation turn finishes, the lock is yielded, allowing the next queued client to proceed without corrupting the model's KV cache or internal state buffers.

---

## 3. KV Cache Compaction Thresholds

To prevent long multi-turn conversations from overflowing the 4GB VRAM limit on the GTX 1050 Ti, the SDK computes:
$$\text{Threshold} = \max(1024, \text{Max KV Tokens} - \text{Max Output Tokens} - 8192)$$

When total history approaches this ceiling, the SDK triggers conversation compaction (summarization / pruning), ensuring stability over extended agent runs.

---

## 4. How to Run the Server & Client

### Step 1: Launch Local Server on Port 9379 (targeting GTX 1050 Ti)
```powershell
uv run python experiments/005_antigravity_sdk_local_serving/local_server.py
```

### Step 2: Run Multi-User Test Client
```powershell
uv run python experiments/005_antigravity_sdk_local_serving/test_client.py
```
