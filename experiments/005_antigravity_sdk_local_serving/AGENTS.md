# AGENTS.md: Google Antigravity SDK Local Serving & Multi-User Architecture

## 1. Study Scope
Integration of **Google Antigravity SDK** (`google-antigravity`) with local LiteRT-LM models on the **NVIDIA GeForce GTX 1050 Ti**, covering:
1. `LiteRTAgentConfig` (managed on-device inference).
2. `LocalOpenAIAgentConfig` (external local inference router).
3. OpenAI-compatible local HTTP completions server (`/v1/chat/completions` with SSE streaming).
4. Multi-user request queuing, thread safety, and KV-cache compaction.

## 2. Architectural Mechanics
- **Local Server (`litert_server.py`)**:
  - Implements an HTTP server translating OpenAI schema payloads (`chat.completion` and `chat.completion.chunk`) to LiteRT `Message`, `Contents`, and `ToolCall` representations.
- **Concurrency & Multi-User Serialization**:
  - The underlying C++ LiteRT engine is non-reentrant for active token generation.
  - Multi-user concurrency is handled via a synchronization lock (`server.engine_lock`). Inbound concurrent requests are queued sequentially.
- **KV Cache Compaction**:
  - `derive_litert_compaction_config()` sets an automated ceiling:
    $$\text{Token Threshold} = \max(1024, \text{Max KV Tokens} - \text{Max Output Tokens} - \text{Buffer Tokens})$$
- **GTX 1050 Ti Routing**:
  - Setting `$env:LITERT_GPU_INDEX="1"` or `select_gpu("1050ti")` guarantees the local server runs all tensor computations on the GTX 1050 Ti.

## 3. Execution Directives
- **Target Device**: ALWAYS enforce GTX 1050 Ti (`$env:LITERT_GPU_INDEX="1"`).
- **Commands**:
  ```powershell
  # Start local server
  uv run python experiments/005_antigravity_sdk_local_serving/local_server.py

  # Test client with concurrent requests
  uv run python experiments/005_antigravity_sdk_local_serving/test_client.py
  ```
