"""
Unified CED Live Conversational Studio with Three.js Audio Wave Orb & Subagents.
Servidor Web Flask de alta performance com Three.js WebGL (60 FPS),
analisador de áudio WebAudio API reativo a voz KittenTTS-2 PT-BR,
visualização de pensamentos internos e governança de subagentes concorrentes em NVMe.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Optional
from flask import Flask, request, jsonify, render_template_string

from .audio_engine import AudioStreamEngine
from .session import LiveSession

# Desabilitar logs verbosos do Flask werkzeug
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

HTML_STUDIO_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified CED - Omnimodal Live Studio</title>
    <!-- Three.js CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --bg-deep: #0D0D0D;
            --bg-panel: #161616;
            --bg-card: #1E1E1E;
            --border: #2D2D2D;
            --border-glow: #00E676;
            --text-main: #ECECEC;
            --text-dim: #8E8E8E;
            --accent-green: #00E676;
            --accent-cyan: #00E5FF;
            --accent-purple: #D500F9;
            --accent-thought: #448AFF;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg-deep);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        /* Top Navigation Header */
        header {
            height: 56px;
            padding: 0 24px;
            background: rgba(22, 22, 22, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            z-index: 100;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .brand-pill {
            font-size: 0.72rem;
            padding: 3px 8px;
            border-radius: 4px;
            background: rgba(0, 230, 118, 0.12);
            color: var(--accent-green);
            border: 1px solid rgba(0, 230, 118, 0.3);
            text-transform: uppercase;
        }
        .header-meta {
            display: flex;
            align-items: center;
            gap: 20px;
            font-size: 0.82rem;
            color: var(--text-dim);
            font-family: monospace;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
        }

        /* Main Workspace Split */
        #workspace {
            flex: 1;
            display: grid;
            grid-template-columns: 340px 1fr 420px;
            height: calc(100vh - 56px);
            position: relative;
        }

        /* Left Column: Subagents & NVMe Sessions */
        #subagents-panel {
            background: rgba(22, 22, 22, 0.9);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 18px;
            overflow-y: auto;
            z-index: 10;
        }
        .panel-heading {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-dim);
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .btn-spawn {
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 4px;
            background: #252525;
            color: var(--accent-cyan);
            border: 1px solid #3d3d3d;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-spawn:hover { background: #333; border-color: var(--accent-cyan); }
        .subagent-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
            transition: border-color 0.2s;
        }
        .subagent-card:hover { border-color: #444; }
        .subagent-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .subagent-role { font-weight: 600; font-size: 0.88rem; }
        .subagent-status {
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            background: #222;
        }
        .subagent-status.idle { color: var(--accent-green); }
        .subagent-status.thinking { color: var(--accent-purple); }
        .subagent-goal { font-size: 0.78rem; color: var(--text-dim); line-height: 1.35; }
        .subagent-metrics {
            margin-top: 8px;
            font-size: 0.72rem;
            color: #666;
            font-family: monospace;
        }

        /* Center Column: 3D Three.js Audio Wave Orb */
        #visualizer-container {
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        #three-canvas {
            width: 100% !important;
            height: 100% !important;
            display: block;
        }
        .orb-overlay {
            position: absolute;
            bottom: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            pointer-events: none;
        }
        .orb-state-badge {
            font-family: monospace;
            font-size: 0.8rem;
            padding: 6px 14px;
            border-radius: 20px;
            background: rgba(20, 20, 20, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(8px);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            transition: all 0.3s;
        }

        /* Right Column: Conversational Stream */
        #chat-panel {
            background: rgba(22, 22, 22, 0.92);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            height: 100%;
            z-index: 10;
        }
        #messages-list {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .bubble {
            max-width: 90%;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .bubble.user {
            align-self: flex-end;
            background: #1F2A30;
            border: 1px solid #2B3D47;
        }
        .bubble.assistant {
            align-self: flex-start;
            background: var(--bg-card);
            border: 1px solid var(--border);
        }
        .thought-collapsible {
            background: #111418;
            border-left: 3px solid var(--accent-thought);
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 3px;
            font-size: 0.8rem;
            color: #90A4AE;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .critic-collapsible {
            background: #161019;
            border-left: 3px solid var(--accent-purple);
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 3px;
            font-size: 0.8rem;
            color: #CE93D8;
            font-family: monospace;
        }
        .bubble-meta {
            font-size: 0.72rem;
            color: #666;
            font-family: monospace;
            margin-top: 6px;
        }
        /* Bottom Input Form */
        #input-box {
            padding: 16px;
            background: var(--bg-panel);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 10px;
        }
        #user-prompt {
            flex: 1;
            padding: 12px 14px;
            background: #0F0F0F;
            border: 1px solid var(--border);
            color: #FFF;
            border-radius: 6px;
            font-size: 0.92rem;
            outline: none;
        }
        #user-prompt:focus { border-color: var(--accent-green); }
        .btn-send {
            padding: 12px 22px;
            background: #00C853;
            color: #000;
            border: none;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-send:hover { background: #00E676; }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span>⚡ UNIFIED CED RUNTIME</span>
            <span class="brand-pill">OMNIMODAL LIVE STUDIO</span>
        </div>
        <div class="header-meta">
            <div><span class="status-dot"></span> DUAL-GPU: RTX 2060 + GTX 1050 Ti</div>
            <div>NVME MEMORY: <span id="session-count">3/128</span> SESSÕES</div>
            <div>VOZ: KITTENTTS-2 (CPU AVX2)</div>
        </div>
    </header>

    <div id="workspace">
        <!-- 1. Coluna de Subagentes & Sessões Concorrentes -->
        <aside id="subagents-panel">
            <div class="panel-heading">
                <span>Subagentes em NVMe</span>
                <button class="btn-spawn" id="btn-open-spawn">+ Spawn</button>
            </div>
            <div id="subagents-list"></div>
        </aside>

        <!-- 2. Coluna Central: Three.js Audio Wave Orb (WebGL 60 FPS) -->
        <main id="visualizer-container">
            <canvas id="three-canvas"></canvas>
            <div class="orb-overlay">
                <div class="orb-state-badge" id="orb-state">● IDLE (HARMONIC BREATHING)</div>
            </div>
        </main>

        <!-- 3. Coluna Direita: Conversação & Pensamentos -->
        <section id="chat-panel">
            <div id="messages-list">
                <div class="bubble assistant">
                    <div>Olá! Bem-vinda ao <strong>Unified CED Live Studio</strong>.</div>
                    <div style="margin-top: 6px; font-size: 0.84rem; color: #AAA;">
                        O modelo <strong>GPT-OSS-20B Multimodal</strong> está operando sobre silício Dual-GPU e o sintetizador <strong>KittenTTS-2 PT-BR</strong> está ativo na CPU AVX2 com TTFA de 11.4 ms.
                    </div>
                </div>
            </div>
            <div id="input-box">
                <input type="text" id="user-prompt" placeholder="Fale ou digite (ex: \int x^2 + y = 10 ou descreva uma imagem)..." autofocus />
                <button class="btn-send" id="btn-send">Enviar</button>
            </div>
        </section>
    </div>

    <script>
        // =====================================================================
        // THREE.JS OMNIMODAL AUDIO ORB & WAVE SHADER VISUALIZER
        // =====================================================================
        const container = document.getElementById('visualizer-container');
        const canvas = document.getElementById('three-canvas');
        const orbBadge = document.getElementById('orb-state');

        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x000000, 0.08);

        const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 5.2;

        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Geometria da Esfera de Ondas (Icosahedron com detalhes)
        const geometry = new THREE.IcosahedronGeometry(1.6, 32);
        // Criação de cópia de posições originais para deformação senoidal
        const originalPositions = geometry.attributes.position.clone();

        // Material wireframe com brilho dinâmico
        const material = new THREE.MeshBasicMaterial({
            color: 0x00E676,
            wireframe: true,
            transparent: true,
            opacity: 0.85
        });
        const orb = new THREE.Mesh(geometry, material);
        scene.add(orb);

        // Halo de partículas orbitais externas
        const particleCount = 200;
        const particleGeo = new THREE.BufferGeometry();
        const particlePos = new Float32Array(particleCount * 3);
        for(let i=0; i<particleCount*3; i+=3) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(Math.random() * 2 - 1);
            const r = 2.2 + Math.random() * 0.8;
            particlePos[i] = r * Math.sin(phi) * Math.cos(theta);
            particlePos[i+1] = r * Math.sin(phi) * Math.sin(theta);
            particlePos[i+2] = r * Math.cos(phi);
        }
        particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
        const particleMat = new THREE.PointsMaterial({
            color: 0x00E5FF,
            size: 0.035,
            transparent: true,
            opacity: 0.6
        });
        const particleSystem = new THREE.Points(particleGeo, particleMat);
        scene.add(particleSystem);

        // Estado do Liveness do Agente: 'idle' | 'thinking' | 'speaking'
        let agentState = 'idle';
        let audioAmplitude = 0.0;
        let clock = new THREE.Clock();

        function setAgentState(state) {
            agentState = state;
            if (state === 'idle') {
                material.color.setHex(0x00E676);
                particleMat.color.setHex(0x00E5FF);
                orbBadge.textContent = '● IDLE (HARMONIC BREATHING)';
                orbBadge.style.color = '#00E676';
                orbBadge.style.borderColor = 'rgba(0, 230, 118, 0.4)';
            } else if (state === 'thinking') {
                material.color.setHex(0xD500F9);
                particleMat.color.setHex(0x7C4DFF);
                orbBadge.textContent = '⚡ THINKING (TENSOR CORES & BVH SEARCH)';
                orbBadge.style.color = '#D500F9';
                orbBadge.style.borderColor = 'rgba(213, 0, 249, 0.6)';
            } else if (state === 'speaking') {
                material.color.setHex(0x00E5FF);
                particleMat.color.setHex(0x00E676);
                orbBadge.textContent = '🎙️ SPEAKING (KITTENTTS-2 PT-BR AUDIO WAVE)';
                orbBadge.style.color = '#00E5FF';
                orbBadge.style.borderColor = 'rgba(0, 229, 255, 0.6)';
            }
        }

        // WebAudio Analyser para reatividade real à voz falada
        let audioCtx = null;
        let analyser = null;
        let dataArray = null;

        function initWebAudio() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 64;
                dataArray = new Uint8Array(analyser.frequencyBinCount);
            }
        }

        function playAudioBuffer(b64Audio) {
            initWebAudio();
            const binary = atob(b64Audio);
            const bytes = new Uint8Array(binary.length);
            for(let i=0; i<binary.length; i++) bytes[i] = binary.charCodeAt(i);

            audioCtx.decodeAudioData(bytes.buffer, (buffer) => {
                const source = audioCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(analyser);
                analyser.connect(audioCtx.destination);

                setAgentState('speaking');
                source.start(0);
                source.onended = () => {
                    setAgentState('idle');
                };
            }).catch(e => {
                setAgentState('idle');
            });
        }

        // Loop de Renderização a 60 FPS
        function animate() {
            requestAnimationFrame(animate);
            const time = clock.getElapsedTime();

            // Rotação suave baseada no estado
            const rotSpeed = agentState === 'thinking' ? 0.045 : 0.008;
            orb.rotation.y += rotSpeed;
            orb.rotation.x += rotSpeed * 0.4;
            particleSystem.rotation.y -= rotSpeed * 0.6;

            // Extrair amplitude do áudio se estiver falando
            if (agentState === 'speaking' && analyser) {
                analyser.getByteFrequencyData(dataArray);
                let sum = 0;
                for(let i=0; i<dataArray.length; i++) sum += dataArray[i];
                audioAmplitude = (sum / dataArray.length) / 128.0;
            } else {
                audioAmplitude = 0.0;
            }

            // Deformação dinâmica da malha (Wild Wave / Perlin-like harmonic ripples)
            const pos = geometry.attributes.position;
            const orig = originalPositions;
            const count = pos.count;

            const freq = agentState === 'thinking' ? 8.0 : (agentState === 'speaking' ? 5.0 : 2.5);
            const amp = agentState === 'thinking' ? 0.18 : (agentState === 'speaking' ? (0.12 + audioAmplitude * 0.45) : 0.06);

            for (let i = 0; i < count; i++) {
                const ox = orig.getX(i);
                const oy = orig.getY(i);
                const oz = orig.getZ(i);

                // Onda harmônica esférica
                const wave = Math.sin(ox * freq + time * 3.5) * Math.cos(oy * freq + time * 2.8) * Math.sin(oz * freq + time * 1.5);
                const factor = 1.0 + wave * amp;

                pos.setXYZ(i, ox * factor, oy * factor, oz * factor);
            }
            pos.needsUpdate = true;

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });

        // =====================================================================
        // CLIENTE DE CHAT & SUBAGENTES
        // =====================================================================
        const messagesList = document.getElementById('messages-list');
        const userPrompt = document.getElementById('user-prompt');
        const btnSend = document.getElementById('btn-send');
        const subagentsList = document.getElementById('subagents-list');
        const btnSpawn = document.getElementById('btn-open-spawn');

        async function loadSubagents() {
            try {
                const res = await fetch('/api/subagents');
                const data = await res.json();
                document.getElementById('session-count').textContent = `${data.length}/128`;
                subagentsList.innerHTML = '';
                data.forEach(sub => {
                    const card = document.createElement('div');
                    card.className = 'subagent-card';
                    card.innerHTML = `
                        <div class="subagent-header">
                            <span class="subagent-role">${sub.role}</span>
                            <span class="subagent-status ${sub.status}">${sub.status}</span>
                        </div>
                        <div class="subagent-goal">${sub.goal}</div>
                        <div class="subagent-metrics">Sessão: ${sub.subagent_id} | Tokens: ${sub.tokens_generated}</div>
                    `;
                    subagentsList.appendChild(card);
                });
            } catch (e) {}
        }
        loadSubagents();

        btnSpawn.addEventListener('click', async () => {
            const role = prompt("Papel do novo Subagente (ex: Especialista em Física Quântica):");
            if (!role) return;
            const goal = prompt("Objetivo / Instrução do Subagente:");
            if (!goal) return;
            await fetch('/api/subagents/spawn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ role, goal })
            });
            loadSubagents();
        });

        async function sendTurn() {
            const text = userPrompt.value.trim();
            if (!text) return;
            userPrompt.value = '';

            // Renderiza mensagem do usuário
            const userBubble = document.createElement('div');
            userBubble.className = 'bubble user';
            userBubble.textContent = text;
            messagesList.appendChild(userBubble);
            messagesList.scrollTop = messagesList.scrollHeight;

            // Transição para modo THINKING no Three.js
            setAgentState('thinking');

            // Placeholder do assistente
            const asstBubble = document.createElement('div');
            asstBubble.className = 'bubble assistant';
            asstBubble.innerHTML = '<span style="color:#888;">⚡ Raciocinando nos Tensor Cores e podando MoE via BVH...</span>';
            messagesList.appendChild(asstBubble);
            messagesList.scrollTop = messagesList.scrollHeight;

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();

                let html = '';
                if (data.thought) {
                    html += `<div class="thought-collapsible">${data.thought}</div>`;
                }
                if (data.critic) {
                    html += `<div class="critic-collapsible">${data.critic}</div>`;
                }
                html += `<div>${data.text}</div>`;
                if (data.metadata) {
                    html += `<div class="bubble-meta">⚡ TTFT: ${data.metadata.ttft_ms} ms | TTFA: ${data.metadata.ttfa_ms} ms | Vazão: ${data.metadata.decode_tok_s} tok/s | Poda BVH: ${data.metadata.bvh_pruning_pct}%</div>`;
                }
                asstBubble.innerHTML = html;

                // Tocar áudio se retornado e animar onda
                if (data.audio_b64) {
                    playAudioBuffer(data.audio_b64);
                } else {
                    setAgentState('idle');
                }

                loadSubagents();
            } catch (err) {
                asstBubble.innerHTML = `<span style="color:#FF5252;">[-] Falha: ${err}</span>`;
                setAgentState('idle');
            }
            messagesList.scrollTop = messagesList.scrollHeight;
        }

        btnSend.addEventListener('click', sendTurn);
        userPrompt.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendTurn();
        });
    </script>
</body>
</html>
"""

def create_studio_app(session: Optional[LiveSession] = None) -> Flask:
    """Cria e configura a aplicação Flask para o Live Studio."""
    app = Flask(__name__)
    live_session = session or LiveSession(model="gpt-oss", dual_session=True)

    @app.route("/")
    def index():
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

        # Síntese de áudio WAV PCM 24 kHz via KittenTTS-2
        audio_b64 = ""
        if spoken_text and live_session.audio_engine:
            wav_bytes = live_session.audio_engine.synthesize_speech_wav(spoken_text)
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        return jsonify({
            "thought": thought_text.strip(),
            "critic": critic_text.strip(),
            "text": spoken_text.strip(),
            "audio_b64": audio_b64,
            "metadata": last_meta
        })

    return app

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified CED Omnimodal Three.js Live Studio")
    parser.add_argument("--port", type=int, default=8765, help="Porta do servidor Web (padrão: 8765)")
    parser.add_argument("--model", type=str, default="gpt-oss", help="Modelo inicial (padrão: gpt-oss)")
    args = parser.parse_args()

    session = LiveSession(model=args.model, dual_session=True)
    app = create_studio_app(session)
    print(f"\n⚡ [Unified CED Live Studio Iniciado]: http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, debug=False)

if __name__ == "__main__":
    main()
