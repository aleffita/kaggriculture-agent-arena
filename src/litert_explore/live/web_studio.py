"""
Unified CED Minimal Web UI (llama.cpp-style Dark Mode) and OpenAI-Compliant HTTP Server.
Equipped with dual-mode Chat/Responses SSE streaming, model and reasoning effort selectors,
image attachment support, collapsible reasoning thoughts, and NVMe KV-cache purge.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Optional
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context

from .audio_engine import AudioStreamEngine
from .session import LiveSession
from .openai_api import create_openai_blueprint

# Desabilitar logs verbosos do Flask werkzeug
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

HTML_STUDIO_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified CED Runtime - Minimal Web UI (Dual-GPU RTX 2060 + GTX 1050 Ti)</title>
    <style>
        :root {
            --bg-canvas: #09090b;
            --bg-surface: #141416;
            --bg-card: #1c1c1f;
            --bg-input: #18181b;
            --border: #27272a;
            --border-focus: #3f3f46;
            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
            --text-dim: #71717a;
            --accent-primary: #10b981;
            --accent-primary-dim: rgba(16, 185, 129, 0.12);
            --accent-reasoning: #38bdf8;
            --accent-reasoning-dim: rgba(56, 189, 248, 0.10);
            --danger: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-canvas);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        /* Top Navigation Header */
        header {
            height: 52px;
            padding: 0 16px;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-shrink: 0;
            z-index: 10;
        }
        .brand-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-title {
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--text-primary);
        }
        .brand-badge {
            font-size: 0.70rem;
            font-family: monospace;
            padding: 2px 6px;
            border-radius: 4px;
            background: var(--accent-primary-dim);
            color: var(--accent-primary);
            border: 1px solid rgba(16, 185, 129, 0.3);
            text-transform: uppercase;
        }
        .controls-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .ctrl-select, .ctrl-btn {
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 0.82rem;
            outline: none;
            cursor: pointer;
            transition: border-color 0.15s ease;
        }
        .ctrl-select:focus, .ctrl-btn:hover {
            border-color: var(--border-focus);
        }
        .ctrl-btn.btn-danger {
            color: var(--danger);
        }
        .ctrl-btn.btn-danger:hover {
            border-color: var(--danger);
            background: rgba(239, 68, 68, 0.1);
        }
        /* Main Chat Container */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            max-width: 920px;
            width: 100%;
            margin: 0 auto;
            position: relative;
        }
        #chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message-row {
            display: flex;
            flex-direction: column;
            gap: 6px;
            width: 100%;
        }
        .message-row.user {
            align-items: flex-end;
        }
        .message-row.assistant {
            align-items: flex-start;
        }
        .sender-tag {
            font-size: 0.72rem;
            color: var(--text-dim);
            font-family: monospace;
            padding: 0 4px;
        }
        .bubble {
            max-width: 90%;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.92rem;
            line-height: 1.55;
            word-break: break-word;
            white-space: pre-wrap;
        }
        .bubble.user {
            background: #27272a;
            color: var(--text-primary);
            border: 1px solid #3f3f46;
        }
        .bubble.assistant {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border);
            width: 100%;
            max-width: 100%;
        }
        /* Reasoning / Thought Details Box */
        details.reasoning-box {
            background: var(--accent-reasoning-dim);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 6px;
            margin-bottom: 12px;
            overflow: hidden;
        }
        details.reasoning-box summary {
            padding: 8px 12px;
            font-size: 0.78rem;
            color: var(--accent-reasoning);
            cursor: pointer;
            font-weight: 600;
            user-select: none;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        details.reasoning-box summary:hover {
            background: rgba(56, 189, 248, 0.08);
        }
        .reasoning-content {
            padding: 10px 12px;
            font-size: 0.80rem;
            color: #bae6fd;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            line-height: 1.45;
            border-top: 1px solid rgba(56, 189, 248, 0.15);
            max-height: 280px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        /* Code and Pre formatting */
        pre {
            background: #111113;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 12px;
            margin: 8px 0;
            overflow-x: auto;
            font-family: ui-monospace, monospace;
            font-size: 0.84rem;
        }
        code {
            font-family: ui-monospace, monospace;
            font-size: 0.86rem;
            background: rgba(255, 255, 255, 0.08);
            padding: 2px 4px;
            border-radius: 4px;
        }
        /* Attachment Preview */
        .attachment-bar {
            display: flex;
            gap: 8px;
            padding: 4px 16px;
            overflow-x: auto;
        }
        .attachment-chip {
            display: flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        .attachment-chip img {
            width: 24px;
            height: 24px;
            object-fit: cover;
            border-radius: 3px;
        }
        .attachment-chip .btn-remove {
            cursor: pointer;
            color: var(--danger);
            font-weight: bold;
        }
        /* Bottom Input Bar */
        footer {
            padding: 12px 16px 16px 16px;
            background: var(--bg-surface);
            border-top: 1px solid var(--border);
            flex-shrink: 0;
        }
        .input-box {
            display: flex;
            gap: 8px;
            align-items: flex-end;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
            transition: border-color 0.15s ease;
        }
        .input-box:focus-within {
            border-color: var(--border-focus);
        }
        textarea#prompt-input {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 0.92rem;
            font-family: inherit;
            resize: none;
            outline: none;
            max-height: 160px;
            min-height: 24px;
            line-height: 1.4;
        }
        .input-actions {
            display: flex;
            gap: 6px;
            align-items: center;
        }
        .icon-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 6px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: color 0.15s ease, background 0.15s ease;
        }
        .icon-btn:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.06);
        }
        .send-btn {
            background: var(--accent-primary);
            color: #000;
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.15s ease;
        }
        .send-btn:hover {
            opacity: 0.9;
        }
        .send-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
        /* Metrics Footer */
        .metrics-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.72rem;
            color: var(--text-dim);
            font-family: monospace;
            padding-top: 6px;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand-group">
            <span class="brand-title">Unified CED</span>
            <span class="brand-badge">Dual-GPU Ring</span>
            <span style="font-size: 0.76rem; color: var(--text-dim);">RTX 2060 + GTX 1050 Ti | NVMe Direct I/O</span>
        </div>
        <div class="controls-group">
            <select id="model-select" class="ctrl-select" title="Modelo Neural">
                <option value="gpt-oss-20b" selected>gpt-oss-20b (MoE Top-4)</option>
                <option value="gemma-4-E2B-it">gemma-4-E2B-it (LiteRT D3D12)</option>
                <option value="bonsai-27b">bonsai-27b (Ternário 1.58b)</option>
                <option value="ornith-35b">ornith-35b (Unsloth 2-bit)</option>
            </select>
            <select id="effort-select" class="ctrl-select" title="Nível de Raciocínio (Reasoning Effort)">
                <option value="low">low (8k)</option>
                <option value="medium" selected>medium (16k)</option>
                <option value="high">high (64k)</option>
                <option value="dynamic">dynamic (unbounded)</option>
                <option value="ultra">ultra (64k + Virtual Experts)</option>
                <option value="ultra-dynamic">ultra-dynamic (Virtual Experts)</option>
            </select>
            <select id="endpoint-select" class="ctrl-select" title="Modo de Protocolo">
                <option value="completions" selected>Completions (/v1/chat/completions)</option>
                <option value="responses">Responses (/v1/responses)</option>
            </select>
            <button id="reset-btn" class="ctrl-btn btn-danger" title="Limpar NVMe KV-Cache e Histórico">Reset</button>
        </div>
    </header>

    <main>
        <div id="chat-messages">
            <div class="message-row assistant">
                <span class="sender-tag">GPT-OSS 20B (Unified CED)</span>
                <div class="bubble assistant">
                    Olá! O runtime unificado heterogêneo está pronto. Você pode conversar diretamente com o GPT-OSS 20B ou selecionar outro modelo no menu superior.
                </div>
            </div>
        </div>
        <div id="attachments-bar" class="attachment-bar" style="display: none;"></div>
    </main>

    <footer>
        <div class="input-box">
            <textarea id="prompt-input" rows="1" placeholder="Envie uma mensagem ou consulte o modelo... (Enter para enviar, Shift+Enter para nova linha)"></textarea>
            <div class="input-actions">
                <input type="file" id="image-file-input" accept="image/*" style="display: none;">
                <button id="attach-btn" class="icon-btn" title="Anexar Imagem">📷</button>
                <button id="send-btn" class="send-btn">Enviar</button>
            </div>
        </div>
        <div class="metrics-strip">
            <span id="speed-metric">Vazão: -- tok/s | TTFT: -- ms</span>
            <span id="status-metric">Status: Conectado</span>
        </div>
    </footer>

    <script>
        const messagesContainer = document.getElementById('chat-messages');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        const resetBtn = document.getElementById('reset-btn');
        const modelSelect = document.getElementById('model-select');
        const effortSelect = document.getElementById('effort-select');
        const endpointSelect = document.getElementById('endpoint-select');
        const speedMetric = document.getElementById('speed-metric');
        const statusMetric = document.getElementById('status-metric');
        const attachBtn = document.getElementById('attach-btn');
        const imageFileInput = document.getElementById('image-file-input');
        const attachmentsBar = document.getElementById('attachments-bar');

        let conversationHistory = [];
        let attachedImages = [];
        let isGenerating = false;
        let activeAbortController = null;

        // Auto-resize do textarea
        promptInput.addEventListener('input', () => {
            promptInput.style.height = 'auto';
            promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + 'px';
        });

        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitMessage();
            }
        });

        sendBtn.addEventListener('click', submitMessage);

        // Upload de Imagem
        attachBtn.addEventListener('click', () => imageFileInput.click());
        imageFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (evt) => {
                const b64 = evt.target.result;
                attachedImages.push({ name: file.name, dataUrl: b64 });
                renderAttachments();
            };
            reader.readAsDataURL(file);
            imageFileInput.value = '';
        });

        function renderAttachments() {
            if (attachedImages.length === 0) {
                attachmentsBar.style.display = 'none';
                attachmentsBar.innerHTML = '';
                return;
            }
            attachmentsBar.style.display = 'flex';
            attachmentsBar.innerHTML = '';
            attachedImages.forEach((img, idx) => {
                const chip = document.createElement('div');
                chip.className = 'attachment-chip';
                chip.innerHTML = `
                    <img src="${img.dataUrl}" alt="${img.name}">
                    <span>${img.name}</span>
                    <span class="btn-remove" onclick="removeAttachment(${idx})">&times;</span>
                `;
                attachmentsBar.appendChild(chip);
            });
        }

        window.removeAttachment = function(idx) {
            attachedImages.splice(idx, 1);
            renderAttachments();
        };

        // Reset de Memória
        resetBtn.addEventListener('click', async () => {
            if (!confirm('Deseja purgar o histórico de contexto e buffers de disco em Z:\\models?')) return;
            try {
                const res = await fetch('/v1/runtime/reset', { method: 'POST' });
                if (res.ok) {
                    conversationHistory = [];
                    messagesContainer.innerHTML = `
                        <div class="message-row assistant">
                            <span class="sender-tag">Unified CED</span>
                            <div class="bubble assistant">Memória e cache NVMe purgados com sucesso. Pronto para nova sessão.</div>
                        </div>
                    `;
                    statusMetric.textContent = 'Status: Memória Resetada';
                }
            } catch (err) {
                alert('Erro ao resetar runtime: ' + err.message);
            }
        });

        async function submitMessage() {
            if (isGenerating) {
                if (activeAbortController) {
                    activeAbortController.abort();
                    activeAbortController = null;
                }
                isGenerating = false;
                sendBtn.textContent = 'Enviar';
                statusMetric.textContent = 'Status: Interrompido';
                return;
            }

            const text = promptInput.value.trim();
            if (!text && attachedImages.length === 0) return;

            promptInput.value = '';
            promptInput.style.height = 'auto';

            // Monta mensagem do usuário
            let userMsgContent;
            if (attachedImages.length > 0) {
                userMsgContent = [{ type: 'text', text: text }];
                attachedImages.forEach(img => {
                    userMsgContent.push({
                        type: 'image_url',
                        image_url: { url: img.dataUrl }
                    });
                });
            } else {
                userMsgContent = text;
            }

            // Append na UI
            appendUserMessage(text, attachedImages);
            conversationHistory.push({ role: 'user', content: userMsgContent });
            attachedImages = [];
            renderAttachments();

            // Prepara resposta do assistente
            const assistantRow = createAssistantMessageRow();
            const reasoningBox = assistantRow.querySelector('.reasoning-box');
            const reasoningContent = assistantRow.querySelector('.reasoning-content');
            const textBubble = assistantRow.querySelector('.bubble-text');

            isGenerating = true;
            sendBtn.textContent = 'Parar';
            statusMetric.textContent = 'Status: Gerando...';

            activeAbortController = new AbortController();
            const selectedModel = modelSelect.value;
            const selectedEffort = effortSelect.value;
            const selectedEndpoint = endpointSelect.value;

            const tStart = performance.now();
            let tokenCount = 0;
            let firstTokenTime = null;

            try {
                if (selectedEndpoint === 'completions') {
                    // Endpoint /v1/chat/completions
                    const payload = {
                        model: selectedModel,
                        messages: conversationHistory,
                        stream: true,
                        reasoning_effort: selectedEffort
                    };

                    const response = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer alefita'
                        },
                        body: JSON.stringify(payload),
                        signal: activeAbortController.signal
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    let fullText = '';
                    let fullReasoning = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop(); // Mantém linha incompleta no buffer

                        for (const line of lines) {
                            const trimmed = line.trim();
                            if (!trimmed || !trimmed.startsWith('data: ')) continue;
                            const dataStr = trimmed.slice(6);
                            if (dataStr === '[DONE]') break;

                            try {
                                const parsed = JSON.parse(dataStr);
                                const delta = parsed.choices?.[0]?.delta;
                                if (!delta) continue;

                                if (delta.reasoning_content) {
                                    if (!firstTokenTime) firstTokenTime = performance.now();
                                    fullReasoning += delta.reasoning_content;
                                    reasoningBox.style.display = 'block';
                                    reasoningContent.textContent = fullReasoning;
                                    tokenCount++;
                                }

                                if (delta.content) {
                                    if (!firstTokenTime) firstTokenTime = performance.now();
                                    fullText += delta.content;
                                    textBubble.textContent = fullText;
                                    tokenCount++;
                                }
                            } catch (e) {
                                // Ignora JSON malformado temporário
                            }
                        }
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }

                    conversationHistory.push({ role: 'assistant', content: fullText });

                } else {
                    // Endpoint /v1/responses
                    const payload = {
                        model: selectedModel,
                        input: text,
                        stream: true,
                        reasoning: { effort: selectedEffort }
                    };

                    const response = await fetch('/v1/responses', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer alefita'
                        },
                        body: JSON.stringify(payload),
                        signal: activeAbortController.signal
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    let fullText = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();

                        for (const line of lines) {
                            const trimmed = line.trim();
                            if (!trimmed || !trimmed.startsWith('data: ')) continue;
                            const dataStr = trimmed.slice(6);
                            if (dataStr === '[DONE]') break;

                            try {
                                const parsed = JSON.parse(dataStr);
                                if (parsed.type === 'response.output_text.delta' && parsed.delta) {
                                    if (!firstTokenTime) firstTokenTime = performance.now();
                                    fullText += parsed.delta;
                                    textBubble.textContent = fullText;
                                    tokenCount++;
                                } else if (parsed.type === 'response.output_item.done' && parsed.item?.content?.[0]?.text) {
                                    fullText = parsed.item.content[0].text;
                                    textBubble.textContent = fullText;
                                }
                            } catch (e) {}
                        }
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }

                    conversationHistory.push({ role: 'assistant', content: fullText });
                }

                const elapsedSec = (performance.now() - tStart) / 1000;
                const ttftMs = firstTokenTime ? Math.round(firstTokenTime - tStart) : 0;
                const tokPerSec = elapsedSec > 0 ? (tokenCount / elapsedSec).toFixed(1) : '0';
                speedMetric.textContent = `Vazão: ${tokPerSec} tok/s | TTFT: ${ttftMs} ms | Tokens: ${tokenCount}`;
                statusMetric.textContent = 'Status: Concluído';

            } catch (err) {
                if (err.name === 'AbortError') {
                    statusMetric.textContent = 'Status: Interrompido pelo usuário';
                } else {
                    textBubble.textContent += `\n[Erro: ${err.message}]`;
                    statusMetric.textContent = 'Status: Erro na requisição';
                }
            } finally {
                isGenerating = false;
                sendBtn.textContent = 'Enviar';
                activeAbortController = null;
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }

        function appendUserMessage(text, images) {
            const row = document.createElement('div');
            row.className = 'message-row user';

            let imgHtml = '';
            if (images && images.length > 0) {
                imgHtml = '<div style="display:flex;gap:6px;margin-bottom:6px;">' +
                    images.map(img => `<img src="${img.dataUrl}" style="max-width:140px;max-height:140px;border-radius:4px;border:1px solid #3f3f46;">`).join('') +
                    '</div>';
            }

            row.innerHTML = `
                <span class="sender-tag">Usuário</span>
                <div class="bubble user">${imgHtml}${escapeHtml(text)}</div>
            `;
            messagesContainer.appendChild(row);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function createAssistantMessageRow() {
            const row = document.createElement('div');
            row.className = 'message-row assistant';
            const modelName = modelSelect.options[modelSelect.selectedIndex].text;

            row.innerHTML = `
                <span class="sender-tag">${escapeHtml(modelName)}</span>
                <div class="bubble assistant">
                    <details class="reasoning-box" style="display: none;">
                        <summary><span>🧠 Raciocínio & Virtual Experts</span></summary>
                        <div class="reasoning-content"></div>
                    </details>
                    <div class="bubble-text">...</div>
                </div>
            `;
            messagesContainer.appendChild(row);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            return row;
        }

        function escapeHtml(str) {
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }
    </script>
</body>
</html>
"""

def create_studio_app(session: Optional[LiveSession] = None, headless: bool = False) -> Flask:
    """Cria e configura a aplicação Flask para o Live Studio e API OpenAI."""
    app = Flask(__name__)
    live_session = session or LiveSession(model="gpt-oss", dual_session=False)

    @app.before_request
    def handle_global_options():
        if request.method == "OPTIONS":
            resp = Response(status=204)
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Headers"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            return resp

    @app.after_request
    def add_global_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Expose-Headers"] = "*"
        return response

    # Registra a API compatível com OpenAI e extensões do protocolo CORDIS
    app.register_blueprint(create_openai_blueprint(live_session))

    @app.route("/")
    def index():
        if headless:
            return jsonify({
                "status": "online",
                "mode": "headless_api",
                "runtime": "Unified Heterogeneous CED Engine",
                "drafter": "auto (eagle3/mtp/dspark)",
                "endpoints": {
                    "models": "/v1/models",
                    "chat_completions": "/v1/chat/completions",
                    "responses": "/v1/responses",
                    "audio_transcriptions": "/v1/audio/transcriptions",
                    "audio_speech": "/v1/audio/speech",
                    "runtime_reset": "/v1/runtime/reset"
                }
            })
        return render_template_string(HTML_STUDIO_TEMPLATE)

    @app.route("/api/subagents", methods=["GET"])
    def get_subagents():
        return jsonify(live_session.list_subagents())

    @app.route("/api/subagents/spawn", methods=["POST"])
    def spawn_subagent():
        data = request.get_json() or {}
        role = data.get("role", "Especialista Analítico")
        goal = data.get("goal", "Auxiliar no raciocínio profundo")
        sub = live_session.spawn_subagent(role=role, goal=goal)
        return jsonify(sub.to_dict())

    @app.route("/api/subagents/execute", methods=["POST"])
    def execute_subagent():
        data = request.get_json() or {}
        sub_id = data.get("subagent_id", "")
        task = data.get("task", "")
        out = live_session.execute_subagent(sub_id, task)
        return jsonify({"subagent_id": sub_id, "output": out})

    @app.route("/api/switch_model", methods=["POST"])
    def switch_model():
        data = request.get_json() or {}
        new_model = data.get("model", "gpt-oss")
        msg = live_session.switch_model(new_model)
        return jsonify({"status": msg, "model": live_session.model})

    @app.route("/api/chat/interrupt", methods=["POST"])
    def interrupt_turn():
        live_session.interrupt()
        return jsonify({"status": "interrupted", "success": True})

    @app.route("/api/chat/stream", methods=["POST", "GET"])
    def chat_stream():
        if request.method == "POST":
            data = request.get_json() or {}
            user_msg = data.get("message", "")
        else:
            user_msg = request.args.get("message", "")

        def generate_sse():
            for stream_type, chunk, meta in live_session.stream_turn(user_msg):
                packet = {
                    "type": stream_type,
                    "chunk": chunk,
                    "metadata": meta
                }
                yield f"data: {json.dumps(packet, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return Response(
            stream_with_context(generate_sse()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive"
            }
        )

    @app.route("/api/chat", methods=["POST"])
    def chat_turn():
        data = request.get_json() or {}
        user_msg = data.get("message", "")

        thought_text = ""
        critic_text = ""
        spoken_text = ""
        last_meta = {}

        for stream_type, chunk, meta in live_session.stream_turn(user_msg):
            last_meta = meta
            if stream_type == "thought":
                thought_text += chunk
            elif stream_type == "critic":
                critic_text += chunk
            elif stream_type == "text":
                spoken_text += chunk
            elif stream_type == "interrupted":
                spoken_text += f"\n[Interrompido: {chunk}]"
                break

        return jsonify({
            "thought": thought_text.strip(),
            "critic": critic_text.strip(),
            "text": spoken_text.strip(),
            "metadata": last_meta
        })

    return app

def main():
    import argparse
    import socket
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Unified CED Minimal Live Web UI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Endereço de rede para escuta (padrão: 127.0.0.1, use 0.0.0.0 para expor na LAN)")
    parser.add_argument("--port", type=int, default=8765, help="Porta do servidor Web (padrão: 8765)")
    parser.add_argument("--model", type=str, default="gpt-oss-20b", help="Modelo inicial (padrão: gpt-oss-20b)")
    parser.add_argument("--headless", action="store_true", default=False, help="Executa em modo API headless puro sem interface Web")
    args = parser.parse_args()

    session = LiveSession(model=args.model, dual_session=False)
    app = create_studio_app(session, headless=args.headless)

    mode_label = "Headless API (OpenAI Wire Protocol)" if args.headless else "Live Web UI (llama.cpp-style Dark Mode)"
    print(f"\n[+] [Unified CED Server Iniciado - {mode_label}]:")
    print(f"   * Local: http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        try:
            hostname = socket.gethostname()
            lan_ips = [ip for ip in socket.gethostbyname_ex(hostname)[2] if not ip.startswith("127.")]
            for ip in lan_ips:
                print(f"   * LAN:   http://{ip}:{args.port}")
        except Exception:
            print(f"   * LAN:   http://0.0.0.0:{args.port}")
    elif args.host != "127.0.0.1":
        print(f"   * Host:  http://{args.host}:{args.port}")
    print()

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
