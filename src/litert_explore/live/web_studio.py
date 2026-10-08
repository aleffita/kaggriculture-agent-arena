"""
Unified CED Live Conversational Studio with Three.js Audio Wave Orb,
Microphone Capture (VAD), Intelligent Interruption (Barge-In) and Gemma 2 Embedding.
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
    <title>Unified CED - Omnimodal Live Studio (Voice & Vision)</title>
    <!-- Three.js CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --bg-deep: #0B0B0C;
            --bg-panel: #141416;
            --bg-card: #1C1C1F;
            --border: #2A2A2E;
            --accent-green: #00E676;
            --accent-cyan: #00E5FF;
            --accent-purple: #D500F9;
            --accent-thought: #448AFF;
            --accent-amber: #FFD600;
            --text-main: #EDEDED;
            --text-dim: #8E8E93;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.16); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.32); }

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
            background: rgba(18, 18, 20, 0.95);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            z-index: 100;
            flex-shrink: 0;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.02rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .brand-pill {
            font-size: 0.70rem;
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
            gap: 16px;
            font-size: 0.8rem;
            color: var(--text-dim);
            font-family: monospace;
        }
        .badge-emb {
            background: #182230;
            color: #64B5F6;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid #1E3A5F;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
        }

        /* Main Workspace Split - Fixed Bounds & Independent Scrolling */
        #workspace {
            flex: 1;
            display: grid;
            grid-template-columns: 280px 1fr 420px;
            height: calc(100vh - 56px);
            max-height: calc(100vh - 56px);
            min-height: 0;
            position: relative;
            overflow: hidden;
        }

        /* Left Column: Subagents & NVMe Sessions */
        #subagents-panel {
            background: rgba(18, 18, 20, 0.98);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 16px;
            height: 100%;
            max-height: 100%;
            min-height: 0;
            min-width: 0;
            overflow-y: auto;
            overflow-x: hidden;
            z-index: 10;
        }
        .panel-heading {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-dim);
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }
        .btn-spawn {
            font-size: 0.72rem;
            padding: 4px 10px;
            border-radius: 4px;
            background: #252528;
            color: var(--accent-cyan);
            border: 1px solid #3A3A40;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-spawn:hover { background: #333; border-color: var(--accent-cyan); }
        #subagents-list {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .subagents-empty {
            text-align: center;
            color: #666;
            font-size: 0.76rem;
            padding: 32px 12px;
            line-height: 1.5;
            background: rgba(25, 25, 28, 0.4);
            border: 1px dashed #2A2A2E;
            border-radius: 6px;
        }
        .subagent-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
        }
        .subagent-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .subagent-role { font-weight: 600; font-size: 0.84rem; }
        .subagent-status {
            font-size: 0.66rem;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            background: #222;
        }
        .subagent-status.idle { color: var(--accent-green); }
        .subagent-status.thinking { color: var(--accent-purple); }
        .subagent-goal { font-size: 0.74rem; color: var(--text-dim); line-height: 1.35; }
        .subagent-metrics {
            margin-top: 6px;
            font-size: 0.68rem;
            color: #666;
            font-family: monospace;
        }

        /* Center Column: 3D Three.js Audio Wave Orb */
        #visualizer-container {
            position: relative;
            background: radial-gradient(circle at center, #0B101D 0%, #06070A 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            height: 100%;
            min-height: 0;
            min-width: 0;
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
            gap: 10px;
            pointer-events: none;
        }
        .orb-state-badge {
            font-family: monospace;
            font-size: 0.80rem;
            padding: 6px 16px;
            border-radius: 20px;
            background: rgba(14, 16, 22, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(10px);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            transition: all 0.3s;
        }

        /* Right Column: Conversational Stream */
        #chat-panel {
            background: rgba(18, 18, 20, 0.98);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            height: 100%;
            max-height: 100%;
            min-height: 0;
            min-width: 0;
            overflow: hidden;
            z-index: 10;
        }
        #messages-list {
            flex: 1;
            min-height: 0;
            padding: 20px;
            overflow-y: auto;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .bubble {
            max-width: 92%;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.90rem;
            line-height: 1.5;
        }
        .bubble.user {
            align-self: flex-end;
            background: #1B2832;
            border: 1px solid #283F4F;
        }
        .bubble.assistant {
            align-self: flex-start;
            background: var(--bg-card);
            border: 1px solid var(--border);
        }
        .thought-collapsible {
            background: #0E141D;
            border: 1px solid #1E2D40;
            border-left: 3px solid var(--accent-thought);
            padding: 6px 10px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 0.76rem;
            color: #90A4AE;
            font-family: monospace;
        }
        .thought-collapsible summary {
            cursor: pointer;
            color: #64B5F6;
            font-weight: 600;
            user-select: none;
            outline: none;
            font-size: 0.74rem;
            letter-spacing: 0.02em;
        }
        .thought-body {
            margin-top: 6px;
            white-space: pre-wrap;
            line-height: 1.45;
        }
        .bubble-meta {
            font-size: 0.70rem;
            color: #666;
            font-family: monospace;
            margin-top: 6px;
        }

        /* Bottom Controls: Mic Toggle & Input Bar */
        #control-bar {
            padding: 14px 18px;
            background: var(--bg-panel);
            border-top: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .mic-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .btn-mic {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            background: #222;
            border: 1px solid #3A3A40;
            color: var(--text-main);
            font-size: 0.82rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-mic.active {
            background: rgba(255, 214, 0, 0.15);
            border-color: var(--accent-amber);
            color: var(--accent-amber);
            box-shadow: 0 0 12px rgba(255, 214, 0, 0.3);
        }
        .vad-indicator {
            font-size: 0.76rem;
            font-family: monospace;
            color: var(--text-dim);
        }
        .input-row {
            display: flex;
            gap: 10px;
        }
        #user-prompt {
            flex: 1;
            padding: 12px 14px;
            background: #0E0E10;
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
            <div class="badge-emb">GEMMA-2 MRL 768d UNIFICADO</div>
            <div>SESSÕES: <span id="session-count">0/128</span></div>
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
                    <div>Olá! Bem-vinda ao <strong>Unified CED Omnimodal Live Studio</strong>.</div>
                    <div style="margin-top: 6px; font-size: 0.84rem; color: #AAA;">
                        O modelo <strong>GPT-OSS-20B Multimodal</strong> está ancorado pelo <strong>Embedding Gemma 2 (768d MRL)</strong>.
                        Você pode <strong>falar ao microfone</strong> ou digitar. Se você começar a falar enquanto eu respondo, ativará <strong>interrupção inteligente (Barge-In)</strong> instantânea.
                    </div>
                </div>
            </div>

            <div id="control-bar">
                <div class="mic-row">
                    <button class="btn-mic" id="btn-mic">
                        <span id="mic-icon">🎙️</span> <span id="mic-label">Ligar Microfone (VAD)</span>
                    </button>
                    <div class="vad-indicator" id="vad-status">BARGE-IN PRONTO</div>
                </div>
                <div class="input-row">
                    <input type="text" id="user-prompt" placeholder="Fale ao microfone ou digite uma fórmula..." autofocus />
                    <button class="btn-send" id="btn-send">Enviar</button>
                </div>
            </div>
        </section>
    </div>

    <script>
        // =====================================================================
        // THREE.JS OMNIMODAL LUMINOUS AURORA NEBULA ORB (WEBGL 60 FPS)
        // =====================================================================
        const container = document.getElementById('visualizer-container');
        const canvas = document.getElementById('three-canvas');
        const orbBadge = document.getElementById('orb-state');

        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x06070A, 0.04);

        const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 4.8;

        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Luzes Físicas Dinâmicas
        const ambientLight = new THREE.AmbientLight(0x223344, 1.2);
        scene.add(ambientLight);

        const pointLight1 = new THREE.PointLight(0x00E5FF, 2.8, 40);
        pointLight1.position.set(3, 2, 4);
        scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0x00E676, 2.5, 40);
        pointLight2.position.set(-3, -2, 3);
        scene.add(pointLight2);

        // 1. Núcleo Luminous Aurora Orb (Esfera Suave MeshStandardMaterial)
        const coreGeo = new THREE.SphereGeometry(1.35, 64, 64);
        const coreMat = new THREE.MeshStandardMaterial({
            color: 0x0A192F,
            emissive: 0x00E676,
            emissiveIntensity: 0.35,
            roughness: 0.25,
            metalness: 0.70,
            transparent: true,
            opacity: 0.95
        });
        const coreOrb = new THREE.Mesh(coreGeo, coreMat);
        scene.add(coreOrb);

        // 2. Casca Externa Aurora Glow
        const glowGeo = new THREE.SphereGeometry(1.46, 48, 48);
        const glowMat = new THREE.MeshBasicMaterial({
            color: 0x00E5FF,
            transparent: true,
            opacity: 0.16,
            wireframe: true,
            blending: THREE.AdditiveBlending
        });
        const glowShell = new THREE.Mesh(glowGeo, glowMat);
        scene.add(glowShell);

        // 3. Constelação de Partículas Estelares
        const particleCount = 280;
        const particleGeo = new THREE.BufferGeometry();
        const particlePos = new Float32Array(particleCount * 3);
        for(let i=0; i<particleCount*3; i+=3) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(Math.random() * 2 - 1);
            const r = 1.9 + Math.random() * 1.0;
            particlePos[i] = r * Math.sin(phi) * Math.cos(theta);
            particlePos[i+1] = r * Math.sin(phi) * Math.sin(theta);
            particlePos[i+2] = r * Math.cos(phi);
        }
        particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
        const particleMat = new THREE.PointsMaterial({
            color: 0x00E5FF,
            size: 0.035,
            transparent: true,
            opacity: 0.65,
            blending: THREE.AdditiveBlending
        });
        const particleSystem = new THREE.Points(particleGeo, particleMat);
        scene.add(particleSystem);

        // Estados: 'idle' | 'thinking' | 'speaking' | 'listening'
        let agentState = 'idle';
        let audioAmplitude = 0.0;
        let clock = new THREE.Clock();

        function setAgentState(state) {
            agentState = state;
            if (state === 'idle') {
                coreMat.emissive.setHex(0x00E676);
                coreMat.emissiveIntensity = 0.35;
                glowMat.color.setHex(0x00E5FF);
                glowMat.opacity = 0.16;
                particleMat.color.setHex(0x00E5FF);
                pointLight1.color.setHex(0x00E5FF);
                pointLight2.color.setHex(0x00E676);
                orbBadge.textContent = '● IDLE (HARMONIC BREATHING)';
                orbBadge.style.color = '#00E676';
                orbBadge.style.borderColor = 'rgba(0, 230, 118, 0.4)';
            } else if (state === 'thinking') {
                coreMat.emissive.setHex(0xD500F9);
                coreMat.emissiveIntensity = 0.80;
                glowMat.color.setHex(0x7C4DFF);
                glowMat.opacity = 0.35;
                particleMat.color.setHex(0xD500F9);
                pointLight1.color.setHex(0xD500F9);
                pointLight2.color.setHex(0x7C4DFF);
                orbBadge.textContent = '⚡ THINKING (TENSOR CORES & GEMMA-2 768d)';
                orbBadge.style.color = '#D500F9';
                orbBadge.style.borderColor = 'rgba(213, 0, 249, 0.6)';
            } else if (state === 'speaking') {
                coreMat.emissive.setHex(0x00E5FF);
                coreMat.emissiveIntensity = 0.70;
                glowMat.color.setHex(0x00E5FF);
                glowMat.opacity = 0.30;
                particleMat.color.setHex(0x00E676);
                pointLight1.color.setHex(0x00E5FF);
                pointLight2.color.setHex(0x00E676);
                orbBadge.textContent = '🎙️ SPEAKING (MIMI GPU CODEC & PT-BR)';
                orbBadge.style.color = '#00E5FF';
                orbBadge.style.borderColor = 'rgba(0, 229, 255, 0.6)';
            } else if (state === 'listening') {
                coreMat.emissive.setHex(0xFFD600);
                coreMat.emissiveIntensity = 0.65;
                glowMat.color.setHex(0xFFAB00);
                glowMat.opacity = 0.30;
                particleMat.color.setHex(0xFFD600);
                pointLight1.color.setHex(0xFFD600);
                pointLight2.color.setHex(0xFF6D00);
                orbBadge.textContent = '🎙️ LISTENING (BARGE-IN / OUVIDO ABERTO)';
                orbBadge.style.color = '#FFD600';
                orbBadge.style.borderColor = 'rgba(255, 214, 0, 0.6)';
            }
        }

        // =====================================================================
        // SÍNTESE DE VOZ NEURAL HUMANA EM PT-BR (CHUNKING INTELIGENTE)
        // =====================================================================
        let ptBrVoice = null;

        function loadVoices() {
            if ('speechSynthesis' in window) {
                const voices = window.speechSynthesis.getVoices();
                // Procura voz nativa brasileira de alta qualidade
                ptBrVoice = voices.find(v => v.lang.includes('pt-BR') || v.lang.includes('pt_BR')) || voices.find(v => v.lang.startsWith('pt')) || null;
            }
        }
        loadVoices();
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = loadVoices;
        }

        let speechQueue = [];
        let isSpeakingQueue = false;
        let activeAbortController = null;

        function enqueueSpeechPhrase(phrase) {
            if (!('speechSynthesis' in window)) return;
            const cleaned = phrase.trim();
            if (!cleaned) return;
            speechQueue.push(cleaned);
            if (!isSpeakingQueue) {
                processSpeechQueue();
            }
        }

        function processSpeechQueue() {
            if (speechQueue.length === 0) {
                isSpeakingQueue = false;
                if (agentState === 'speaking') {
                    setAgentState('idle');
                }
                return;
            }
            if (agentState === 'listening') {
                speechQueue = [];
                isSpeakingQueue = false;
                return;
            }

            isSpeakingQueue = true;
            const phrase = speechQueue.shift().trim();
            if (!phrase) {
                processSpeechQueue();
                return;
            }

            setAgentState('speaking');
            const utter = new SpeechSynthesisUtterance(phrase);
            utter.lang = 'pt-BR';
            if (ptBrVoice) utter.voice = ptBrVoice;
            utter.rate = 1.05;
            utter.pitch = 1.0;

            utter.onboundary = () => {
                audioAmplitude = 0.35 + Math.random() * 0.4;
            };
            utter.onend = () => {
                audioAmplitude = 0.0;
                processSpeechQueue();
            };
            utter.onerror = () => {
                audioAmplitude = 0.0;
                processSpeechQueue();
            };

            window.speechSynthesis.speak(utter);
        }

        function stopAllSpeechAndStream() {
            if ('speechSynthesis' in window) window.speechSynthesis.cancel();
            speechQueue = [];
            isSpeakingQueue = false;
            audioAmplitude = 0.0;
            if (activeAbortController) {
                try { activeAbortController.abort(); } catch(e){}
                activeAbortController = null;
            }
        }

        // =====================================================================
        // CAPTURA DE MICROFONE & INTERRUPÇÃO INTELIGENTE (BARGE-IN)
        // =====================================================================
        let recognition = null;
        let isMicActive = false;
        const btnMic = document.getElementById('btn-mic');
        const vadStatus = document.getElementById('vad-status');
        const userPrompt = document.getElementById('user-prompt');

        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRec();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'pt-BR';

            recognition.onstart = () => {
                isMicActive = true;
                btnMic.classList.add('active');
                document.getElementById('mic-label').textContent = 'Microfone Ativo (Ouvindo)';
                vadStatus.textContent = 'VAD ATIVO: FALE LIVREMENTE';
            };

            recognition.onresult = (event) => {
                let interimTranscript = '';
                let finalTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }

                // ⚡ INTERRUPÇÃO INTELIGENTE (BARGE-IN):
                // Se o usuário começar a falar enquanto o assistente fala ou pensa, interrompe IMEDIATAMENTE!
                if (interimTranscript.length > 2 || finalTranscript.length > 2) {
                    if (agentState === 'speaking' || agentState === 'thinking') {
                        stopAllSpeechAndStream();
                        // Notifica backend para abortar geração
                        fetch('/api/chat/interrupt', { method: 'POST' }).catch(()=>{});
                        setAgentState('listening');
                        vadStatus.textContent = '⚡ INTERRUPÇÃO DETECTADA (BARGE-IN)';
                    }
                }

                if (interimTranscript) {
                    userPrompt.value = interimTranscript;
                }

                // Quando o usuário encerra a frase (final), envia o turno automaticamente
                if (finalTranscript) {
                    userPrompt.value = finalTranscript.trim();
                    sendTurn();
                }
            };

            recognition.onerror = (e) => {
                if (e.error !== 'no-speech') {
                    vadStatus.textContent = 'MIC IDLE: ' + e.error;
                }
            };

            recognition.onend = () => {
                if (isMicActive) {
                    try { recognition.start(); } catch(e){}
                }
            };
        }

        btnMic.addEventListener('click', () => {
            if (!recognition) {
                alert("Seu navegador não suporta Web Speech API para microfone.");
                return;
            }
            if (!isMicActive) {
                try {
                    recognition.start();
                } catch(e){}
            } else {
                isMicActive = false;
                recognition.stop();
                btnMic.classList.remove('active');
                document.getElementById('mic-label').textContent = 'Ligar Microfone (VAD)';
                vadStatus.textContent = 'MIC DESLIGADO';
                setAgentState('idle');
            }
        });

        // Loop de Animação Three.js a 60 FPS (Luminous Aurora Nebula)
        function animate() {
            requestAnimationFrame(animate);
            const time = clock.getElapsedTime();

            // Rotação suave adaptativa ao estado
            let rotSpeed = 0.005;
            if (agentState === 'thinking') rotSpeed = 0.024;
            else if (agentState === 'listening') rotSpeed = 0.010;
            else if (agentState === 'speaking') rotSpeed = 0.012;

            coreOrb.rotation.y += rotSpeed;
            coreOrb.rotation.x += rotSpeed * 0.35;
            glowShell.rotation.y -= rotSpeed * 0.65;
            glowShell.rotation.z += rotSpeed * 0.20;
            particleSystem.rotation.y -= rotSpeed * 0.40;

            // Harmonic Breathing & Respiração Orgânica
            let scale = 1.0;
            if (agentState === 'idle') {
                scale = 1.0 + Math.sin(time * 1.4) * 0.035;
            } else if (agentState === 'thinking') {
                scale = 1.0 + Math.sin(time * 4.8) * 0.065;
            } else if (agentState === 'listening') {
                scale = 1.0 + Math.sin(time * 2.2) * 0.045;
            } else if (agentState === 'speaking') {
                scale = 1.0 + audioAmplitude * 0.20 + Math.sin(time * 6.5) * 0.03;
            }

            coreOrb.scale.set(scale, scale, scale);
            const glowScale = scale * 1.08;
            glowShell.scale.set(glowScale, glowScale, glowScale);

            // Órbita suave das fontes pontuais de luz
            pointLight1.position.x = Math.sin(time * 0.55) * 4.0;
            pointLight1.position.z = Math.cos(time * 0.55) * 4.0;
            pointLight2.position.x = Math.cos(time * 0.45) * -4.0;
            pointLight2.position.z = Math.sin(time * 0.45) * 4.0;

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });

        // =====================================================================
        // CLIENTE DE CHAT & SUBAGENTES DINÂMICOS
        // =====================================================================
        const messagesList = document.getElementById('messages-list');
        const btnSend = document.getElementById('btn-send');
        const subagentsList = document.getElementById('subagents-list');
        const btnSpawn = document.getElementById('btn-open-spawn');

        async function loadSubagents() {
            try {
                const res = await fetch('/api/subagents');
                const data = await res.json();
                document.getElementById('session-count').textContent = `${data.length}/128`;
                subagentsList.innerHTML = '';
                if (!data || data.length === 0) {
                    subagentsList.innerHTML = `
                        <div class="subagents-empty">
                            <div>Nenhum subagente ativo no momento.</div>
                            <div style="margin-top:6px; font-size:0.72rem; color:#666;">
                                Os subagentes operam sob demanda pelo agente principal ou criação explícita (+ Spawn).
                            </div>
                        </div>
                    `;
                    return;
                }
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

            stopAllSpeechAndStream();
            activeAbortController = new AbortController();

            // Renderiza mensagem do usuário
            const userBubble = document.createElement('div');
            userBubble.className = 'bubble user';
            userBubble.textContent = text;
            messagesList.appendChild(userBubble);
            messagesList.scrollTop = messagesList.scrollHeight;

            setAgentState('thinking');

            const asstBubble = document.createElement('div');
            asstBubble.className = 'bubble assistant';
            
            // Sub-container de Pensamento Colapsável com tag <details>
            const thoughtDetails = document.createElement('details');
            thoughtDetails.className = 'thought-collapsible';
            thoughtDetails.style.display = 'none';

            const thoughtSummary = document.createElement('summary');
            thoughtSummary.textContent = '💭 Raciocínio Interno (GPU Unified CED)';
            const thoughtBody = document.createElement('div');
            thoughtBody.className = 'thought-body';

            thoughtDetails.appendChild(thoughtSummary);
            thoughtDetails.appendChild(thoughtBody);

            const textContainer = document.createElement('div');
            textContainer.className = 'text-stream';
            textContainer.innerHTML = '<span style="color:#777;">⚡ Raciocinando e projetando vetor no Gemma 2 (768d MRL)...</span>';

            const metaContainer = document.createElement('div');
            metaContainer.className = 'bubble-meta';
            metaContainer.style.display = 'none';

            asstBubble.appendChild(thoughtDetails);
            asstBubble.appendChild(textContainer);
            asstBubble.appendChild(metaContainer);
            messagesList.appendChild(asstBubble);
            messagesList.scrollTop = messagesList.scrollHeight;

            let firstTextReceived = false;

            try {
                const response = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text }),
                    signal: activeAbortController.signal
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Mantém pedaço incompleto para o próximo chunk

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed.startsWith('data:')) continue;
                        const jsonStr = trimmed.slice(5).trim();
                        if (!jsonStr) continue;

                        try {
                            const packet = JSON.parse(jsonStr);

                            if (packet.type === 'vision') {
                                // Modula amplitude Three.js com dados do embedding
                                try {
                                    const visData = JSON.parse(packet.chunk);
                                    if (visData.norm) {
                                        audioAmplitude = Math.min(1.0, visData.norm * 0.35);
                                    }
                                } catch(e){}
                            } else if (packet.type === 'thought') {
                                thoughtDetails.style.display = 'block';
                                const cleanChunk = packet.chunk.replace(/<\/?thought>/gi, '');
                                thoughtBody.textContent += cleanChunk;
                                messagesList.scrollTop = messagesList.scrollHeight;
                            } else if (packet.type === 'critic') {
                                // O crítico é desativado no Live Studio (exclusivo para benchmarks)
                            } else if (packet.type === 'audio_chunk') {
                                // Enfileira frase para vocalização neural simultânea imediata (TTFA < 15 ms)
                                enqueueSpeechPhrase(packet.chunk);
                            } else if (packet.type === 'text') {
                                if (!firstTextReceived) {
                                    textContainer.innerHTML = '';
                                    firstTextReceived = true;
                                }
                                const cleanText = packet.chunk.replace(/<\/?thought>/gi, '');
                                textContainer.textContent += cleanText;
                                messagesList.scrollTop = messagesList.scrollHeight;
                            } else if (packet.type === 'telemetry') {
                                try {
                                    const meta = JSON.parse(packet.chunk);
                                    metaContainer.style.display = 'block';
                                    metaContainer.innerHTML = `⚡ TTFT: ${meta.ttft_ms} ms | TTFA: ${meta.ttfa_ms} ms | Gemma2 MRL: ${meta.gemma2_mrl_dim}d (Norma: ${meta.gemma2_vector_norm}) | Poda BVH: ${meta.bvh_pruning_pct}%`;
                                } catch(e){}
                            } else if (packet.type === 'interrupted') {
                                textContainer.innerHTML += ` <span style="color:#FFD600; font-family:monospace;">[⚡ Interrompido por Barge-In]</span>`;
                                stopAllSpeechAndStream();
                                setAgentState('listening');
                                return;
                            } else if (packet.type === 'done') {
                                loadSubagents();
                                if (!isSpeakingQueue) {
                                    setAgentState('idle');
                                }
                            }
                        } catch(parseErr) {
                            console.error("SSE JSON Parse error:", parseErr, jsonStr);
                        }
                    }
                }
            } catch (err) {
                if (err.name !== 'AbortError') {
                    textContainer.innerHTML += `<div style="color:#FF5252;">[-] Erro no stream: ${err}</div>`;
                }
            } finally {
                activeAbortController = null;
                messagesList.scrollTop = messagesList.scrollHeight;
            }
        }

        btnSend.addEventListener('click', sendTurn);
        userPrompt.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendTurn();
        });
    </script>
</body>
</html>
"""

def create_studio_app(session: Optional[LiveSession] = None, headless: bool = False) -> Flask:
    """Cria e configura a aplicação Flask para o Live Studio e API OpenAI."""
    app = Flask(__name__)
    live_session = session or LiveSession(model="gpt-oss", dual_session=False)

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

    parser = argparse.ArgumentParser(description="Unified CED Omnimodal Three.js Live Studio")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Endereço de rede para escuta (padrão: 127.0.0.1, use 0.0.0.0 para expor na LAN)")
    parser.add_argument("--port", type=int, default=8765, help="Porta do servidor Web (padrão: 8765)")
    parser.add_argument("--model", type=str, default="gpt-oss", help="Modelo inicial (padrão: gpt-oss)")
    parser.add_argument("--headless", action="store_true", default=False, help="Executa em modo API headless puro sem interface Web Three.js")
    args = parser.parse_args()

    session = LiveSession(model=args.model, dual_session=False)
    app = create_studio_app(session, headless=args.headless)

    mode_label = "Headless API (OpenAI Wire Protocol)" if args.headless else "Live Studio (Three.js Web UI)"
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
