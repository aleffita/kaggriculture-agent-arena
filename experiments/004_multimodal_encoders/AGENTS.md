# AGENTS.md: Multimodal Encoders (Vision & Audio)

## 1. Study Scope
Investigation of multimodal execution architectures in LiteRT-LM (Vision encoders, ASR speech recognition, and TTS Kokoro synthesis) on the **NVIDIA GeForce GTX 1050 Ti**.

## 2. Multimodal Architecture in LiteRT-LM
- **Vision Subsystem**:
  - Employs SigLIP / PaliGemma vision encoders (`vision_litert_compiled_model_executor.cc`).
  - Image inputs are divided into patches (e.g. 256 patches), transformed into visual token embeddings, and concatenated into the input sequence.
- **Audio Subsystem**:
  - Implements ASR audio runners (`omni/asr/asr_engine.cc`) and Kokoro TTS (`omni/tts/kokoro/`).
- **Heterogeneous Backend Routing**:
  - LiteRT-LM supports granular per-modality backends:
    - `backend="gpu"` (Language generation on GTX 1050 Ti)
    - `vision_backend="gpu"` or `"cpu"`
    - `audio_backend="cpu"` (Audio processing offloaded to host RAM to save GPU VRAM)

## 3. 4GB VRAM Budget Guardrails
- Loading vision weights simultaneously with LLM weights can exceed the 4096 MB VRAM ceiling.
- Best Practice on GTX 1050 Ti: Route heavy auxiliary encoders (Audio/Vision) to `CPU` while keeping autoregressive decode on `GPU` (1050 Ti).

## 4. Execution Directives
- **Probe Command**:
  ```powershell
  uv run python experiments/004_multimodal_encoders/run_probe.py
  ```
