"""Kaggriculture Replay Visualizer, Live Arena, Match History & Submission Sandbox Web Server."""

from __future__ import annotations

import concurrent.futures
import datetime
import importlib.util
import io
import json
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_file, Response

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "kaggriculture" / "agents"))

from kaggle_environments import make
from kaggriculture.db.schema import (
    get_connection,
    list_replays,
    record_replay,
    record_stage_match,
    initialize_schema,
)
from kaggriculture.arena.arena_engine import load_agent_callable

app = Flask(
    __name__,
    template_folder=str(pathlib.Path(__file__).parent / "templates"),
    static_folder=str(pathlib.Path(__file__).parent / "static"),
)

REPLAYS_DIR = repo_root / "kaggriculture" / "data" / "replays"
SUBMISSIONS_DIR = repo_root / "kaggriculture" / "data" / "submissions"
REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

VITE_PLAYER_PATH = (
    repo_root
    / ".venv"
    / "Lib"
    / "site-packages"
    / "kaggle_environments"
    / "envs"
    / "kaggriculture"
    / "visualizer"
    / "default"
    / "dist"
    / "index.html"
)

SUBMISSION_REGISTRY: Dict[str, Dict[str, Any]] = {}
ACTIVE_DUELS: Dict[str, Dict[str, Any]] = {}
DUELS_LOCK = threading.Lock()


def _get_player_template() -> str:
    """Reads the Vite-built official Kaggriculture player HTML."""
    if not VITE_PLAYER_PATH.exists():
        raise FileNotFoundError(f"Vite player not found at {VITE_PLAYER_PATH}")
    return VITE_PLAYER_PATH.read_text(encoding="utf-8")


def _load_submission_callable(sub_id: str):
    """Loads an extracted submission callable from its staged directory."""
    if sub_id not in SUBMISSION_REGISTRY:
        sub_dir = SUBMISSIONS_DIR / sub_id
        if sub_dir.exists():
            entrypoint = sub_dir / "main.py"
            if not entrypoint.exists():
                py_files = list(sub_dir.glob("**/*.py"))
                if py_files:
                    entrypoint = py_files[0]
            if entrypoint and entrypoint.exists():
                SUBMISSION_REGISTRY[sub_id] = {
                    "sub_id": sub_id,
                    "filename": sub_id,
                    "dir": str(sub_dir),
                    "entrypoint": str(entrypoint),
                    "files": [f.name for f in sub_dir.iterdir()],
                    "uploaded_at": datetime.datetime.now().isoformat(),
                }
        if sub_id not in SUBMISSION_REGISTRY:
            raise ValueError(f"Submissão {sub_id} não encontrada.")

    sub_info = SUBMISSION_REGISTRY[sub_id]
    entrypoint_path = pathlib.Path(sub_info["entrypoint"])
    entry_dir = entrypoint_path.parent

    if str(entry_dir) not in sys.path:
        sys.path.insert(0, str(entry_dir))

    spec = importlib.util.spec_from_file_location(f"sub_{sub_id}", str(entrypoint_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Falha ao carregar spec para {entrypoint_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for candidate in ["agent", "my_agent", "submission", "act"]:
        if hasattr(mod, candidate) and callable(getattr(mod, candidate)):
            return getattr(mod, candidate)

    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if callable(attr) and hasattr(attr, "__code__") and attr.__code__.co_argcount in (1, 2):
            return attr

    raise ValueError(f"Nenhuma função de agente encontrada em {entrypoint_path.name}")


@app.route("/")
def index():
    """Serves the main Dashboard with full multi-view layout."""
    return render_template("index.html", v=int(time.time()))


@app.route("/player/<replay_id>")
def serve_player(replay_id: str):
    """Renders the official Kaggriculture visualizer with the replay injected."""
    replay_file = REPLAYS_DIR / f"{replay_id}.json"

    # Resolve player names and match metadata
    p0_label = "Player 1"
    p1_label = "Player 2"
    steps = 72
    match_row = None
    is_live_duel = False

    with DUELS_LOCK:
        if replay_id in ACTIVE_DUELS:
            is_live_duel = True
            d_info = ACTIVE_DUELS[replay_id]
            p0_label = d_info.get("agent_p0", "Player 1")
            p1_label = d_info.get("agent_p1", "Player 2")
            steps = d_info.get("steps", 72)

    if p0_label == "Player 1":
        con = get_connection(read_only=True)
        match_row = con.execute(
            "SELECT agent_p0, agent_p1, steps FROM matches WHERE match_id = ?",
            [replay_id],
        ).fetchone()
        con.close()
        if match_row:
            p0_label, p1_label, steps = match_row

    clean_p0 = p0_label.replace("llm_", "").replace("_v1", "").replace("_", " ").title() if "llm_" in p0_label else p0_label
    clean_p1 = p1_label.replace("llm_", "").replace("_v1", "").replace("_", " ").title() if "llm_" in p1_label else p1_label

    # If file doesn't exist or was truncated by an earlier run, generate with full steps (for database matches only)
    needs_generation = not replay_file.exists() and not is_live_duel
    if not needs_generation and match_row and not is_live_duel:
        try:
            with open(replay_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached.get("steps", [])) < steps:
                needs_generation = True
        except Exception:
            needs_generation = True

    if needs_generation:
        if match_row:
            try:
                fn_p0 = load_agent_callable(p0_label)
                fn_p1 = load_agent_callable(p1_label)
                env = make("kaggriculture", configuration={"episodeSteps": steps, "actTimeout": 999999, "runTimeout": 999999})
                env.run([fn_p0, fn_p1])
                rep_data = env.toJSON()
                rep_data.setdefault("info", {})
                rep_data["info"]["TeamNames"] = [clean_p0, clean_p1]
                rep_data["info"]["Agents"] = [{"index": 0, "name": clean_p0}, {"index": 1, "name": clean_p1}]
                with open(replay_file, "w", encoding="utf-8") as f:
                    json.dump(rep_data, f)
            except Exception as e:
                return f"<h3>Erro simulando replay sob demanda: {e}</h3>", 500
        else:
            return f"<h3>Replay '{replay_id}' não encontrado em disco nem no banco.</h3>", 404

    if not replay_file.exists() and is_live_duel:
        for _ in range(20):
            time.sleep(0.1)
            if replay_file.exists():
                break

    try:
        with open(replay_file, "r", encoding="utf-8") as f:
            replay_data = json.load(f)
    except Exception as e:
        return f"<h3>Erro lendo replay: {e}</h3>", 500

    # Inject official player names into replay metadata and info object
    replay_data.setdefault("info", {})
    replay_data["info"]["TeamNames"] = [clean_p0, clean_p1]
    replay_data["info"]["Agents"] = [{"index": 0, "name": clean_p0}, {"index": 1, "name": clean_p1}]

    base_html = _get_player_template()
    window_kaggle = {
        "debug": False,
        "playing": True,
        "step": 0,
        "controls": True,
        "environment": replay_data,
        "agents": [
            {"index": 0, "name": clean_p0},
            {"index": 1, "name": clean_p1},
        ],
        "logs": [],
    }

    snippet = f"<script>window.kaggle = {json.dumps(window_kaggle)};</script>"
    is_mini = request.args.get("mini", "false").lower() == "true"

    if is_mini:
        mini_style = """
        <style>
            body { margin: 0; padding: 0; overflow: hidden; background: #0b0f19; }
            #root { transform-origin: top left; }
        </style>
        """
        snippet += mini_style
    else:
        fit_style = """
        <style>
            html, body {
                width: 100vw !important;
                height: 100vh !important;
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
                background-color: #0b0f19 !important;
                display: flex !important;
                flex-direction: column !important;
            }
            #app, .player {
                width: 100% !important;
                flex: 1 1 0 !important;
                height: auto !important;
                max-height: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
                display: flex !important;
                flex-direction: column !important;
            }
            html.in-iframe #standalone-hud-bar, body.in-iframe #standalone-hud-bar {
                display: none !important;
            }
            .viewer {
                flex: 1 1 auto !important;
                width: 100% !important;
                height: 100% !important;
                overflow: hidden !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                position: relative !important;
                background-color: #0b0f19 !important;
            }
            .kaggriculture-container {
                width: 100% !important;
                height: 100% !important;
                overflow: hidden !important;
                padding: 0 !important;
                margin: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                background-color: #aacf8a !important;
            }
            .kaggriculture-main {
                width: 1380px !important;
                min-width: 1380px !important;
                max-width: 1380px !important;
                height: auto !important;
                max-height: none !important;
                display: grid !important;
                grid-template-columns: minmax(280px, 1.4fr) minmax(260px, 1fr) minmax(280px, 1.4fr) !important;
                grid-template-rows: auto 1fr auto auto !important;
                grid-template-areas: "bush bush bush" "p1-field town p2-field" "p1-combo market p2-combo" "bush-bot bush-bot bush-bot" !important;
                row-gap: 20px !important;
                column-gap: 24px !important;
                transform-origin: center center !important;
                will-change: transform !important;
                margin: 0 auto !important;
            }
            .player-box {
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
            .town-wrap {
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
            .mobile-title-bar {
                display: none !important;
            }
            .controls {
                flex-shrink: 0 !important;
                height: 48px !important;
                padding: 4px 16px !important;
                background-color: #0f172a !important;
                border-top: 1px solid #1e293b !important;
                box-sizing: border-box !important;
                z-index: 50 !important;
            }
            /* Standalone HUD bar */
            .standalone-hud-bar {
                display: grid;
                grid-template-columns: minmax(280px, 1.1fr) minmax(360px, 1.4fr) minmax(280px, 1.1fr);
                gap: 12px;
                padding: 8px 16px;
                background: #090e17;
                border-bottom: 1px solid #1e293b;
                align-items: center;
                flex-shrink: 0;
                z-index: 100;
                box-sizing: border-box;
            }
            .hud-player-card {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 6px;
                padding: 6px 12px;
                display: flex;
                flex-direction: column;
                gap: 4px;
            }
            .hud-player-card.p0 { border-left: 3px solid #3b82f6; }
            .hud-player-card.p1 { border-right: 3px solid #f59e0b; }
            .hud-player-top {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .hud-avatar { font-size: 18px; }
            .hud-name-box { display: flex; flex-direction: column; flex: 1; overflow: hidden; }
            .hud-player-name { font-size: 12px; font-weight: 700; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
            .hud-player-role { font-size: 10px; color: #94a3b8; }
            .hud-coin-badge { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.35); color: #f59e0b; font-family: monospace; font-weight: 700; font-size: 12px; padding: 2px 8px; border-radius: 4px; }
            .hud-inventory-box { display: flex; flex-wrap: wrap; gap: 4px; min-height: 20px; }
            .hud-inv-pill { background: #1e293b; border: 1px solid #334155; color: #cbd5e1; font-size: 10px; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
            .hud-center-card { background: #111827; border: 1px solid #1f2937; border-radius: 6px; padding: 6px 12px; display: flex; flex-direction: column; align-items: center; gap: 5px; }
            .hud-turn-clock { display: flex; align-items: center; gap: 8px; }
            .hud-clock-pill { font-size: 11px; font-weight: 600; background: #1e293b; padding: 2px 8px; border-radius: 4px; color: #f8fafc; }
            .hud-clock-pill.step { border: 1px solid rgba(139, 92, 246, 0.4); color: #a78bfa; }
            .hud-market-prices { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
            .mkt-pill { background: #1e293b; border: 1px solid #334155; font-size: 10px; padding: 2px 6px; border-radius: 4px; color: #cbd5e1; font-family: monospace; }
            .mkt-pill strong { color: #10b981; }
        </style>
        <script>
        (function() {
            if (window.parent !== window) {
                document.documentElement.classList.add('in-iframe');
                if (document.body) {
                    document.body.classList.add('in-iframe');
                } else {
                    document.addEventListener('DOMContentLoaded', () => document.body && document.body.classList.add('in-iframe'));
                }
            }

            function autoFitGame() {
                const container = document.querySelector('.kaggriculture-container');
                const main = document.querySelector('.kaggriculture-main');
                if (!container || !main) return;

                main.style.transform = 'none';

                const contW = container.clientWidth;
                const contH = container.clientHeight;
                const mainW = 1380;
                const mainH = main.scrollHeight || main.offsetHeight || 780;

                if (mainW > 0 && mainH > 0 && contW > 0 && contH > 0) {
                    const scaleX = (contW * 0.97) / mainW;
                    const scaleY = (contH * 0.97) / mainH;
                    const scale = Math.min(scaleX, scaleY);
                    main.style.transform = 'scale(' + scale.toFixed(4) + ')';
                    main.style.transformOrigin = 'center center';
                }
            }

            window.addEventListener('resize', autoFitGame);
            window.addEventListener('load', autoFitGame);

            const observer = new MutationObserver(() => {
                if (document.querySelector('.kaggriculture-main')) {
                    autoFitGame();
                }
            });
            observer.observe(document.documentElement, { childList: true, subtree: true });

            [20, 60, 150, 400, 800, 1500].forEach(t => setTimeout(autoFitGame, t));

            function extractInv(obs) {
                const priv = (obs && obs.private) ? obs.private : {};
                const seeds = priv.seeds || {};
                const shed = priv.shed || {};
                const inventories = Array.isArray(priv.inventories) ? priv.inventories : [];
                const carried = {};
                for (const inv of inventories) {
                    if (inv && typeof inv === 'object') {
                        for (const [k, v] of Object.entries(inv)) {
                            if (v > 0) carried[k] = (carried[k] || 0) + v;
                        }
                    }
                }
                return { seeds, shed, carried };
            }

            function renderInvPills(containerId, invData) {
                const el = document.getElementById(containerId);
                if (!el) return;
                const items = [];
                const icons = { WHEAT: '🌾', CARROT: '🥕', TOMATO: '🍅', STRAWBERRY: '🍓', MELON: '🍉', EGG: '🥚', MILK: '🥛', WOOL: '🐑' };
                for (const [k, v] of Object.entries(invData.carried || {})) {
                    if (v > 0) items.push((icons[k] || '🎒') + ' ' + k.toLowerCase() + ' x' + v);
                }
                for (const [k, v] of Object.entries(invData.seeds || {})) {
                    if (v > 0) items.push((icons[k] || '🌱') + ' sem. ' + k.toLowerCase() + ' x' + v);
                }
                for (const [k, v] of Object.entries(invData.shed || {})) {
                    if (v > 0) items.push((icons[k] || '📦') + ' ' + k.toLowerCase() + ': ' + v);
                }
                el.innerHTML = items.length === 0 ? '<span class="hud-inv-pill" style="opacity:0.6;">Bolsa vazia</span>' : items.map(it => '<span class="hud-inv-pill">' + it + '</span>').join('');
            }

            function updateStandaloneHud(step, totalSteps, day, totalDays, turn, p0_bank, p1_bank, inv0, inv1, prices) {
                const stepVal = document.getElementById('sa-hud-step-val');
                if (stepVal) stepVal.textContent = step;
                const totVal = document.getElementById('sa-hud-total-steps');
                if (totVal) totVal.textContent = totalSteps;
                const dayVal = document.getElementById('sa-hud-day-val');
                if (dayVal) dayVal.textContent = day + ' / ' + totalDays;
                const turnVal = document.getElementById('sa-hud-turn-val');
                if (turnVal) turnVal.textContent = turn + ' / 24';
                const p0Coins = document.getElementById('sa-hud-p0-coins');
                if (p0Coins) p0Coins.textContent = Math.round(p0_bank).toLocaleString();
                const p1Coins = document.getElementById('sa-hud-p1-coins');
                if (p1Coins) p1Coins.textContent = Math.round(p1_bank).toLocaleString();

                renderInvPills('sa-hud-p0-inv', inv0);
                renderInvPills('sa-hud-p1-inv', inv1);

                const mktMap = { 'sa-mkt-wheat': prices.WHEAT, 'sa-mkt-carrot': prices.CARROT, 'sa-mkt-tomato': prices.TOMATO, 'sa-mkt-strawberry': prices.STRAWBERRY, 'sa-mkt-melon': prices.MELON, 'sa-mkt-egg': prices.EGG, 'sa-mkt-milk': prices.MILK, 'sa-mkt-wool': prices.WOOL };
                for (const [id, val] of Object.entries(mktMap)) {
                    const el = document.getElementById(id);
                    if (el) el.textContent = val !== undefined ? val : '--';
                }
            }

            // Real-time telemetry dispatcher to parent window AND standalone HUD
            let lastStep = -1;
            function notifyTelemetry() {
                try {
                    if (!window.kaggle || !window.kaggle.environment || !window.kaggle.environment.steps) return;
                    
                    // Read from step counter element which updates on every animation frame!
                    const counterEl = document.querySelector('[data-testid="step-counter"]') || document.querySelector('.step-counter');
                    let currentStep = 0;
                    if (counterEl && counterEl.textContent && counterEl.textContent.includes('/')) {
                        const parts = counterEl.textContent.trim().split('/');
                        const parsed = parseInt(parts[0]);
                        if (!isNaN(parsed) && parsed > 0) {
                            currentStep = parsed - 1;
                        }
                    } else {
                        const slider = document.querySelector('.controls input[type=range]') || document.querySelector('input[type=range]');
                        if (slider && !isNaN(parseInt(slider.value))) {
                            currentStep = Math.max(0, parseInt(slider.value) - 1);
                        } else if (typeof window.kaggle.step === 'number') {
                            currentStep = window.kaggle.step;
                        }
                    }

                    if (currentStep === lastStep) return;
                    lastStep = currentStep;

                    const steps = window.kaggle.environment.steps;
                    if (!steps || !steps[currentStep]) return;

                    const p0Step = steps[currentStep][0];
                    const p1Step = steps[currentStep][1] || steps[currentStep][0];
                    const obs0 = (p0Step && p0Step.observation) ? p0Step.observation : {};
                    const obs1 = (p1Step && p1Step.observation) ? p1Step.observation : {};

                    const totalSteps = steps.length - 1;
                    const dayNumber = Math.floor(currentStep / 24) + 1;
                    const totalDays = Math.max(1, Math.ceil(totalSteps / 24));
                    const turnNumber = (currentStep % 24) + 1;

                    let p0_bank = 0;
                    let p1_bank = 0;
                    if (obs0.farms && obs0.farms[0] && obs0.farms[0].money !== undefined) {
                        p0_bank = obs0.farms[0].money;
                    } else if (p0Step.reward !== null && p0Step.reward !== undefined) {
                        p0_bank = p0Step.reward;
                    }

                    if (obs0.farms && obs0.farms[1] && obs0.farms[1].money !== undefined) {
                        p1_bank = obs0.farms[1].money;
                    } else if (obs1.farms && obs1.farms[1] && obs1.farms[1].money !== undefined) {
                        p1_bank = obs1.farms[1].money;
                    } else if (p1Step.reward !== null && p1Step.reward !== undefined) {
                        p1_bank = p1Step.reward;
                    }

                    const inv0 = extractInv(obs0);
                    const inv1 = extractInv(obs1);
                    const market_prices = (obs0.market && obs0.market.prices) ? obs0.market.prices : {};

                    // Update standalone HUD elements if present
                    updateStandaloneHud(currentStep, totalSteps, dayNumber, totalDays, turnNumber, p0_bank, p1_bank, inv0, inv1, market_prices);

                    // Dispatch to parent modal only if inside an iframe to prevent recursive loop with Vite player
                    if (window.parent !== window) {
                        window.parent.postMessage({
                            type: 'KAGGRICULTURE_LIVE_STEP',
                            currentStep: currentStep,
                            totalSteps: totalSteps,
                            day: dayNumber,
                            totalDays: totalDays,
                            turn: turnNumber,
                            p0_bank: p0_bank,
                            p1_bank: p1_bank,
                            p0_inv: inv0,
                            p1_inv: inv1,
                            prices: market_prices
                        }, '*');
                    }
                } catch(e) {}
            }

            setInterval(notifyTelemetry, 50);

            // Live replay step streaming with native Vite visualizer message dispatch
            (function() {
                const repId = """ + json.dumps(replay_id) + """;
                let curLen = (window.kaggle && window.kaggle.environment && window.kaggle.environment.steps) 
                             ? window.kaggle.environment.steps.length : 0;

                let streamTimer = setInterval(async () => {
                    try {
                        const res = await fetch('/api/replay_stream/' + repId);
                        if (!res.ok) return;
                        const d = await res.json();
                        if (d.steps && d.steps.length > 0) {
                            if (!window.kaggle) window.kaggle = {};
                            if (!window.kaggle.environment) window.kaggle.environment = {};
                            if (d.steps.length > curLen) {
                                curLen = d.steps.length;
                                window.kaggle.environment.steps = d.steps;
                                window.kaggle.step = d.steps.length - 1;

                                // Broadcast to Vite visualizer's internal message listener
                                window.postMessage({
                                    environment: { steps: d.steps },
                                    step: d.steps.length - 1
                                }, '*');

                                const slider = document.querySelector('.controls input[type=range]');
                                if (slider) {
                                    slider.max = d.steps.length - 1;
                                    slider.value = d.steps.length - 1;
                                }
                                notifyTelemetry();
                            }
                        }
                        if (d.is_done) {
                            clearInterval(streamTimer);
                        }
                    } catch(e) {}
                }, 350);
            })();
        })();
        </script>
        """
        snippet += fit_style

    total_days = max(1, (steps + 23) // 24)
    hud_markup = f"""
    <div class="standalone-hud-bar" id="standalone-hud-bar">
        <div class="hud-player-card p0">
            <div class="hud-player-top">
                <span class="hud-avatar">🌾</span>
                <div class="hud-name-box">
                    <span class="hud-player-name">{clean_p0}</span>
                    <span class="hud-player-role">Fazenda Esquerda (P0)</span>
                </div>
                <span class="hud-coin-badge">💰 <span id="sa-hud-p0-coins">3.000</span></span>
            </div>
            <div class="hud-inventory-box" id="sa-hud-p0-inv">
                <span class="hud-inv-pill">Bolsa vazia</span>
            </div>
        </div>

        <div class="hud-center-card">
            <div class="hud-turn-clock">
                <span class="hud-clock-pill">📅 Dia <strong id="sa-hud-day-val">1 / {total_days}</strong></span>
                <span class="hud-clock-pill">⏱️ Turno <strong id="sa-hud-turn-val">1 / 24</strong></span>
                <span class="hud-clock-pill step">Passo <strong id="sa-hud-step-val">0</strong> / <span id="sa-hud-total-steps">{steps}</span></span>
            </div>
            <div class="hud-market-prices">
                <span class="mkt-pill">🌾 Trigo: <strong id="sa-mkt-wheat">--</strong></span>
                <span class="mkt-pill">🥕 Cenoura: <strong id="sa-mkt-carrot">--</strong></span>
                <span class="mkt-pill">🍅 Tomate: <strong id="sa-mkt-tomato">--</strong></span>
                <span class="mkt-pill">🍓 Morango: <strong id="sa-mkt-strawberry">--</strong></span>
                <span class="mkt-pill">🍉 Melancia: <strong id="sa-mkt-melon">--</strong></span>
                <span class="mkt-pill">🥚 Ovo: <strong id="sa-mkt-egg">--</strong></span>
                <span class="mkt-pill">🥛 Leite: <strong id="sa-mkt-milk">--</strong></span>
                <span class="mkt-pill">🐑 Lã: <strong id="sa-mkt-wool">--</strong></span>
            </div>
        </div>

        <div class="hud-player-card p1">
            <div class="hud-player-top">
                <span class="hud-coin-badge">💰 <span id="sa-hud-p1-coins">3.000</span></span>
                <div class="hud-name-box" style="text-align: right;">
                    <span class="hud-player-name">{clean_p1}</span>
                    <span class="hud-player-role">Fazenda Direita (P1)</span>
                </div>
                <span class="hud-avatar">⚡</span>
            </div>
            <div class="hud-inventory-box" id="sa-hud-p1-inv" style="justify-content: flex-end;">
                <span class="hud-inv-pill">Bolsa vazia</span>
            </div>
        </div>
    </div>
    """

    if "</head>" in base_html:
        final_html = base_html.replace("</head>", snippet + "</head>", 1)
    else:
        final_html = snippet + base_html

    if not is_mini:
        if "<body>" in final_html:
            final_html = final_html.replace("<body>", "<body>" + hud_markup, 1)
        elif "<body" in final_html:
            import re
            final_html = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + hud_markup, final_html, count=1)

    return Response(final_html, mimetype="text/html")


@app.route("/api/live", methods=["GET"])
def api_live():
    """Returns currently active and recently completed live duels."""
    with DUELS_LOCK:
        now = time.time()
        recent = [
            d for d in ACTIVE_DUELS.values()
            if d.get("status") == "running" or (now - d.get("end_time", now) < 900)
        ]
        has_running = any(d.get("status") == "running" for d in recent)
        return jsonify({
            "active": has_running or len(recent) > 0,
            "has_running": has_running,
            "count": len(recent),
            "duels": sorted(recent, key=lambda x: x.get("start_time", 0), reverse=True),
        })


@app.route("/api/matches", methods=["GET"])
def api_matches():
    """Returns all 120+ matches from DuckDB with search, stage, and agent filters."""
    limit = int(request.args.get("limit", 150))
    offset = int(request.args.get("offset", 0))
    stage_filter = request.args.get("stage", "all")
    agent_filter = request.args.get("agent", "").strip().lower()
    league_filter = request.args.get("league", "").strip().lower()

    try:
        con = get_connection(read_only=True)
        query = """
            SELECT match_id, epoch, stage, steps, bracket, agent_p0, agent_p1,
                   p0_bank, p1_bank, winner, margin, duration_s, timestamp
            FROM matches
            ORDER BY epoch DESC, timestamp DESC
        """
        rows = con.execute(query).fetchall()

        # Check existing replay files
        existing_replays = {f.stem for f in REPLAYS_DIR.glob("*.json")}
        con.close()

        matches = []
        for r in rows:
            mid, ep, stg, steps, brk, p0, p1, b0, b1, win, marg, dur, ts = r
            if isinstance(ts, (datetime.datetime, datetime.date)):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)

            matches.append({
                "match_id": mid,
                "replay_id": mid,
                "has_replay_cached": mid in existing_replays,
                "epoch": ep,
                "stage": stg,
                "steps": steps,
                "bracket": brk,
                "agent_p0": p0,
                "agent_p1": p1,
                "p0_bank": b0,
                "p1_bank": b1,
                "winner": win,
                "margin": marg,
                "duration_s": dur,
                "timestamp": ts_str,
            })

        # Apply filters
        filtered = matches
        if stage_filter != "all":
            filtered = [m for m in filtered if str(m["steps"]) == stage_filter or m["stage"].lower() == stage_filter.lower()]
        if league_filter:
            if "48" in league_filter or "hybrid" in league_filter:
                h48_agents = {"llm_labor_magnate_v1", "llm_sprint_rusher_v1", "llm_land_baron_v1", "llm_melon_monopolist_v1"}
                filtered = [m for m in filtered if m["agent_p0"] in h48_agents or m["agent_p1"] in h48_agents]
            elif "24" in league_filter or "macro" in league_filter or "daily" in league_filter:
                h24_agents = {"llm_market_arbitrageur_v1", "llm_cautious_farmer_v1"}
                filtered = [m for m in filtered if m["agent_p0"] in h24_agents or m["agent_p1"] in h24_agents]
        if agent_filter:
            filtered = [
                m for m in filtered
                if agent_filter in m["agent_p0"].lower()
                or agent_filter in m["agent_p1"].lower()
                or agent_filter in m["winner"].lower()
            ]

        total = len(filtered)
        paged = filtered[offset : offset + limit]

        return jsonify({
            "total": total,
            "matches": paged,
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        return jsonify({"error": str(e), "matches": []}), 500


@app.route("/api/replays", methods=["GET"])
def api_replays():
    """Returns replays indexed in DuckDB for the Replay Gallery."""
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    replays = list_replays(limit=limit, offset=offset)
    return jsonify({"replays": replays, "count": len(replays)})


@app.route("/api/replays/<replay_id>", methods=["GET"])
def api_get_replay(replay_id: str):
    """Downloads or reads raw replay JSON."""
    replay_file = REPLAYS_DIR / f"{replay_id}.json"
    con = get_connection(read_only=True)
    match_row = con.execute("SELECT agent_p0, agent_p1, steps FROM matches WHERE match_id = ?", [replay_id]).fetchone()
    con.close()

    needs_gen = not replay_file.exists()
    if not needs_gen and match_row:
        try:
            with open(replay_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached.get("steps", [])) < match_row[2]:
                needs_gen = True
        except Exception:
            needs_gen = True

    if needs_gen:
        if match_row:
            p0, p1, steps = match_row
            fn_p0 = load_agent_callable(p0)
            fn_p1 = load_agent_callable(p1)
            env = make("kaggriculture", configuration={"episodeSteps": steps, "actTimeout": 999999, "runTimeout": 999999})
            env.run([fn_p0, fn_p1])
            rep_data = env.toJSON()
            clean_p0 = p0.replace("llm_", "").replace("_v1", "").replace("_", " ").title() if "llm_" in p0 else p0
            clean_p1 = p1.replace("llm_", "").replace("_v1", "").replace("_", " ").title() if "llm_" in p1 else p1
            rep_data.setdefault("info", {})
            rep_data["info"]["TeamNames"] = [clean_p0, clean_p1]
            rep_data["info"]["Agents"] = [{"index": 0, "name": clean_p0}, {"index": 1, "name": clean_p1}]
            with open(replay_file, "w", encoding="utf-8") as f:
                json.dump(rep_data, f)
        else:
            return jsonify({"error": "Replay não encontrado."}), 404

    return send_file(replay_file, mimetype="application/json")


@app.route("/api/replay_stream/<replay_id>", methods=["GET"])
def api_replay_stream(replay_id: str):
    """Returns the latest steps array for a live-running or completed duel."""
    replay_file = REPLAYS_DIR / f"{replay_id}.json"
    if not replay_file.exists():
        return jsonify({"error": "Replay ainda não iniciado.", "steps": []}), 404

    with DUELS_LOCK:
        duel = ACTIVE_DUELS.get(replay_id, {})
        status = duel.get("status", "unknown")
        is_running = status in ("running", "starting")
        is_done = status in ("completed", "error")
        curr_step = duel.get("current_step", 0)
        total_steps = duel.get("total_steps", 72)
        duel_error = duel.get("error")

    try:
        with open(replay_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps = data.get("steps", [])
        return jsonify({
            "status": status,
            "is_running": is_running,
            "is_done": is_done,
            "error": duel_error,
            "current_step": curr_step,
            "total_steps": total_steps,
            "steps": steps,
            "count": len(steps),
        })
    except (json.JSONDecodeError, PermissionError):
        # File currently being flushed by worker; return soft retry without breaking client
        return jsonify({
            "status": status,
            "is_running": is_running,
            "is_done": is_done,
            "error": duel_error,
            "current_step": curr_step,
            "total_steps": total_steps,
            "steps": [],
            "count": 0,
        })
    except Exception as e:
        return jsonify({"error": str(e), "steps": []}), 500


@app.route("/api/leaderboard", methods=["GET"])
def api_leaderboard():
    """Returns official agent leaderboard from DuckDB with matches breakdown."""
    try:
        con = get_connection(read_only=True)
        rows = con.execute(
            """
            SELECT agent_id, name, version, league, elo, matches_played,
                   wins, losses, draws, total_coins
            FROM agents
            ORDER BY elo DESC
            """
        ).fetchall()
        con.close()
        cols = [
            "agent_id", "name", "version", "league", "elo", "matches_played",
            "wins", "losses", "draws", "total_coins"
        ]
        leaderboard = [dict(zip(cols, r)) for r in rows]
        return jsonify({"leaderboard": leaderboard})
    except Exception as e:
        return jsonify({"error": str(e), "leaderboard": []}), 500


@app.route("/api/agents", methods=["GET"])
def api_agents():
    """Returns list of selectable opponents."""
    opponents = [
        {
            "id": "llm_labor_magnate_v1",
            "name": "Labor Magnate (Top 1 Suíço Season 3)",
            "architecture": "H48-Hybrid",
            "tag": "Campeão do Torneio (72 pts)",
            "avatar": "🌾",
            "is_llm": True,
        },
        {
            "id": "llm_sprint_rusher_v1",
            "name": "Sprint Rusher (Top Elo 694.9)",
            "architecture": "H48-Hybrid",
            "tag": "97.5% Invicto / Trigo Rápido",
            "avatar": "⚡",
            "is_llm": True,
        },
        {
            "id": "llm_land_baron_v1",
            "name": "Land Baron (Top Capital)",
            "architecture": "H48-Hybrid",
            "tag": "104.100 Moedas / Conservador",
            "avatar": "🏰",
            "is_llm": True,
        },
        {
            "id": "llm_market_arbitrageur_v1",
            "name": "Market Arbitrageur",
            "architecture": "H24 Daily Macro",
            "tag": "Volatilidade / Arbitragem",
            "avatar": "📈",
            "is_llm": True,
        },
        {
            "id": "llm_cautious_farmer_v1",
            "name": "Cautious Farmer",
            "architecture": "H24 Daily Macro",
            "tag": "Foco em Cenoura",
            "avatar": "🥕",
            "is_llm": True,
        },
        {
            "id": "llm_melon_monopolist_v1",
            "name": "Melon Monopolist",
            "architecture": "H48-Hybrid",
            "tag": "Alto Risco / Melão",
            "avatar": "🍉",
            "is_llm": True,
        },
        {
            "id": "starter",
            "name": "Kaggle Starter Baseline",
            "architecture": "Heurística Oficial",
            "tag": "Rápido (<1s)",
            "avatar": "🤖",
            "is_llm": False,
        },
        {
            "id": "random",
            "name": "Random Actions Baseline",
            "architecture": "Estocástico",
            "tag": "Ações Aleatórias",
            "avatar": "🎲",
            "is_llm": False,
        },
    ]
    return jsonify({"agents": opponents})


@app.route("/api/upload_replay", methods=["POST"])
def api_upload_replay():
    """Uploads and indexes an official Kaggle replay JSON."""
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".json"):
        return jsonify({"error": "O arquivo deve ser um JSON de replay."}), 400

    try:
        content = json.load(file)
        if "steps" not in content or "configuration" not in content:
            return jsonify({"error": "JSON inválido: ausência de steps ou configuration."}), 400

        replay_id = f"custom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        out_path = REPLAYS_DIR / f"{replay_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(content, f)

        steps_count = len(content.get("steps", []))
        last_step = content["steps"][-1]
        p0_bank = float(last_step[0].get("reward", 0.0) or 0.0)
        p1_bank = float(last_step[1].get("reward", 0.0) or 0.0)

        p0_name = "Player 0"
        p1_name = "Player 1"
        winner = p0_name if p0_bank > p1_bank else (p1_name if p1_bank > p0_bank else "Draw")
        margin = abs(p0_bank - p1_bank)

        record_replay(
            replay_id=replay_id,
            agent_p0=p0_name,
            agent_p1=p1_name,
            steps=steps_count,
            p0_bank=p0_bank,
            p1_bank=p1_bank,
            winner=winner,
            margin=margin,
            replay_path=str(out_path.relative_to(repo_root)),
            title=file.filename,
            source="upload",
        )

        return jsonify({
            "status": "success",
            "replay_id": replay_id,
            "title": file.filename,
            "steps": steps_count,
            "p0_bank": p0_bank,
            "p1_bank": p1_bank,
            "winner": winner,
        })
    except Exception as e:
        return jsonify({"error": f"Erro processando replay: {str(e)}"}), 500


@app.route("/api/upload_submission", methods=["POST"])
def api_upload_submission():
    """Uploads and stages a Kaggle submission (.tar.gz, .zip, .py)."""
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    fname = file.filename or "submission"
    ext = ""
    for known in [".tar.gz", ".tgz", ".zip", ".py"]:
        if fname.lower().endswith(known):
            ext = known
            break

    if not ext:
        return jsonify({"error": "Formato não suportado. Envie .tar.gz, .zip ou .py."}), 400

    sub_id = f"sub_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    sub_dir = SUBMISSIONS_DIR / sub_id
    sub_dir.mkdir(parents=True, exist_ok=True)

    archive_path = sub_dir / fname
    file.save(str(archive_path))

    extracted_files = []
    entrypoint = None

    try:
        if ext in [".tar.gz", ".tgz"]:
            with tarfile.open(str(archive_path), "r:*") as tar:
                tar.extractall(str(sub_dir))
                extracted_files = [f.name for f in tar.getmembers() if not f.isdir()]
        elif ext == ".zip":
            with zipfile.ZipFile(str(archive_path), "r") as z:
                z.extractall(str(sub_dir))
                extracted_files = z.namelist()
        elif ext == ".py":
            extracted_files = [fname]

        if (sub_dir / "main.py").exists():
            entrypoint = sub_dir / "main.py"
        else:
            py_files = list(sub_dir.glob("**/*.py"))
            if py_files:
                entrypoint = py_files[0]

        if not entrypoint:
            return jsonify({"error": "Nenhum script Python (.py) encontrado no arquivo."}), 400

        SUBMISSION_REGISTRY[sub_id] = {
            "sub_id": sub_id,
            "filename": fname,
            "dir": str(sub_dir),
            "entrypoint": str(entrypoint),
            "files": extracted_files,
            "uploaded_at": datetime.datetime.now().isoformat(),
        }

        return jsonify({
            "status": "success",
            "submission_id": sub_id,
            "filename": fname,
            "entrypoint": entrypoint.name,
            "files_count": len(extracted_files),
        })
    except Exception as e:
        return jsonify({"error": f"Erro descompactando submissão: {str(e)}"}), 500


def _invoke_agent_safe(agent_fn, observation, configuration):
    """Safely calls an agent function, adapting to 1-arg or 2-arg signature."""
    try:
        import inspect
        sig = inspect.signature(agent_fn)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(params) == 1:
            return agent_fn(observation)
        return agent_fn(observation, configuration)
    except Exception:
        try:
            return agent_fn(observation, configuration)
        except TypeError:
            return agent_fn(observation)


def _execute_duel_worker(duel_id: str, p0_name: str, p1_name: str, steps: int, seed: Optional[int]):
    """Background worker that runs a single duel step-by-step, streaming replay JSON in real time."""
    try:
        if p0_name.startswith("sub_"):
            callable_p0 = _load_submission_callable(p0_name)
            p0_label = f"Submissão ({SUBMISSION_REGISTRY[p0_name]['filename']})"
        else:
            callable_p0 = load_agent_callable(p0_name)
            p0_label = p0_name

        if p1_name.startswith("sub_"):
            callable_p1 = _load_submission_callable(p1_name)
            p1_label = f"Submissão ({SUBMISSION_REGISTRY[p1_name]['filename']})"
        else:
            callable_p1 = load_agent_callable(p1_name)
            p1_label = p1_name

        config = {
            "episodeSteps": steps,
            "actTimeout": 999999,
            "runTimeout": 999999,
        }
        if seed is not None:
            config["randomSeed"] = seed

        clean_p0 = p0_label.replace("llm_", "").replace("_v1", "").replace("_", " ").title() if "llm_" in p0_label else p0_label
        clean_p1 = p1_label.replace("llm_", "").replace("_v1", "").replace("_", " ").title() if "llm_" in p1_label else p1_label

        def _decorate_json(rep):
            rep.setdefault("info", {})
            rep["info"]["TeamNames"] = [clean_p0, clean_p1]
            rep["info"]["Agents"] = [{"index": 0, "name": clean_p0}, {"index": 1, "name": clean_p1}]
            return rep

        env = make("kaggriculture", configuration=config, debug=False)
        env.reset()
        t0 = time.time()

        replay_path = REPLAYS_DIR / f"{duel_id}.json"

        # Save step 0 immediately so duel_id.json is ready and accessible in 0ms!
        initial_json = _decorate_json(env.toJSON())
        with open(replay_path, "w", encoding="utf-8") as f:
            json.dump(initial_json, f)

        with DUELS_LOCK:
            ACTIVE_DUELS[duel_id].update({
                "status": "running",
                "agent_p0": p0_label,
                "agent_p1": p1_label,
                "replay_id": duel_id,
                "current_step": 0,
                "total_steps": steps,
                "p0_bank": 0.0,
                "p1_bank": 0.0,
                "progress": f"0/{steps}",
            })

        step_idx = 0
        while not env.done:
            obs0 = env.state[0].observation
            obs1 = env.state[1].observation

            act0 = None
            act1 = None
            try:
                act0 = _invoke_agent_safe(callable_p0, obs0, env.configuration)
            except Exception:
                pass

            try:
                act1 = _invoke_agent_safe(callable_p1, obs1, env.configuration)
            except Exception:
                pass

            env.step([act0, act1])
            step_idx += 1

            s0_reward = float(env.state[0].reward if env.state[0].reward is not None else 0.0)
            s1_reward = float(env.state[1].reward if env.state[1].reward is not None else 0.0)

            # Persist replay every 2 steps or on completion
            if step_idx % 2 == 0 or env.done:
                partial_json = _decorate_json(env.toJSON())
                with open(replay_path, "w", encoding="utf-8") as f:
                    json.dump(partial_json, f)

            with DUELS_LOCK:
                ACTIVE_DUELS[duel_id]["current_step"] = step_idx
                ACTIVE_DUELS[duel_id]["p0_bank"] = s0_reward
                ACTIVE_DUELS[duel_id]["p1_bank"] = s1_reward
                ACTIVE_DUELS[duel_id]["progress"] = f"{step_idx}/{steps}"

        duration_s = time.time() - t0
        final_step = env.steps[-1]
        p0_bank = float(final_step[0].reward if final_step[0].reward is not None else 0.0)
        p1_bank = float(final_step[1].reward if final_step[1].reward is not None else 0.0)

        if p0_bank > p1_bank:
            winner = p0_label
            margin = p0_bank - p1_bank
        elif p1_bank > p0_bank:
            winner = p1_label
            margin = p1_bank - p0_bank
        else:
            winner = "Draw"
            margin = 0.0

        # Save final complete replay
        final_json = _decorate_json(env.toJSON())
        with open(replay_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f)

        # Record in DuckDB
        record_replay(
            replay_id=duel_id,
            agent_p0=p0_label,
            agent_p1=p1_label,
            steps=steps,
            p0_bank=p0_bank,
            p1_bank=p1_bank,
            winner=winner,
            margin=margin,
            replay_path=str(replay_path.relative_to(repo_root)),
            title=f"{p0_label} vs {p1_label} ({steps}s)",
            source="live_duel",
        )

        with DUELS_LOCK:
            ACTIVE_DUELS[duel_id].update({
                "status": "completed",
                "replay_id": duel_id,
                "winner": winner,
                "margin": margin,
                "p0_bank": p0_bank,
                "p1_bank": p1_bank,
                "duration_s": round(duration_s, 2),
                "end_time": time.time(),
                "progress": f"{steps}/{steps}",
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        with DUELS_LOCK:
            ACTIVE_DUELS[duel_id].update({
                "status": "error",
                "error": str(e),
                "end_time": time.time(),
            })


@app.route("/api/run_match", methods=["POST"])
def api_run_match():
    """Starts a match (sandbox or 1v1 duel) and tracks it in ACTIVE_DUELS for live viewing."""
    data = request.json or {}
    p0_name = data.get("agent_p0", "starter")
    p1_name = data.get("agent_p1", "llm_labor_magnate_v1")
    steps = int(data.get("steps", 72))
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except ValueError:
            seed = None

    duel_id = f"duel_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    p0_label = p0_name
    if p0_name.startswith("sub_"):
        if p0_name not in SUBMISSION_REGISTRY:
            try:
                _load_submission_callable(p0_name)
            except Exception:
                pass
        if p0_name in SUBMISSION_REGISTRY:
            p0_label = f"Submissão ({SUBMISSION_REGISTRY[p0_name]['filename']})"

    p1_label = p1_name
    if p1_name.startswith("sub_"):
        if p1_name not in SUBMISSION_REGISTRY:
            try:
                _load_submission_callable(p1_name)
            except Exception:
                pass
        if p1_name in SUBMISSION_REGISTRY:
            p1_label = f"Submissão ({SUBMISSION_REGISTRY[p1_name]['filename']})"

    with DUELS_LOCK:
        ACTIVE_DUELS[duel_id] = {
            "duel_id": duel_id,
            "match_id": duel_id,
            "replay_id": duel_id,
            "agent_p0": p0_label,
            "agent_p1": p1_label,
            "steps": steps,
            "seed": seed,
            "status": "starting",
            "current_step": 0,
            "total_steps": steps,
            "progress": f"0/{steps}",
            "p0_bank": 0.0,
            "p1_bank": 0.0,
            "winner": None,
            "start_time": time.time(),
        }

    t = threading.Thread(
        target=_execute_duel_worker,
        args=(duel_id, p0_name, p1_name, steps, seed),
        daemon=True,
    )
    t.start()

    return jsonify({
        "status": "started",
        "duel_id": duel_id,
        "match_id": duel_id,
        "replay_id": duel_id,
        "agent_p0": p0_label,
        "agent_p1": p1_label,
        "steps": steps,
        "message": f"Duelo iniciado ao vivo: {p0_label} vs {p1_label} ({steps} passos)",
    })


@app.route("/api/run_live_round", methods=["POST"])
def api_run_live_round():
    """Starts simultaneous live arena duels in parallel so the user can watch the arena in action."""
    data = request.json or {}
    steps = int(data.get("steps", 72))

    pairs = [
        ("starter", "random"),
        ("starter", "starter"),
    ]

    launched = []
    for p0, p1 in pairs:
        duel_id = f"live_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        with DUELS_LOCK:
            ACTIVE_DUELS[duel_id] = {
                "duel_id": duel_id,
                "match_id": duel_id,
                "replay_id": duel_id,
                "agent_p0": p0,
                "agent_p1": p1,
                "steps": steps,
                "status": "starting",
                "current_step": 0,
                "total_steps": steps,
                "progress": f"0/{steps}",
                "p0_bank": 0.0,
                "p1_bank": 0.0,
                "winner": None,
                "start_time": time.time(),
            }
        t = threading.Thread(
            target=_execute_duel_worker,
            args=(duel_id, p0, p1, steps, None),
            daemon=True,
        )
        t.start()
        launched.append(duel_id)

    return jsonify({
        "status": "started",
        "launched_duels": launched,
        "count": len(launched),
        "message": f"{len(launched)} duelos ao vivo iniciados simultaneamente na arena!",
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns runtime diagnostic information."""
    with DUELS_LOCK:
        active_count = sum(1 for d in ACTIVE_DUELS.values() if d.get("status") == "running")
    return jsonify({
        "status": "operational",
        "duckdb": str(repo_root / "kaggriculture" / "data" / "arena.duckdb"),
        "replays_count": len(list(REPLAYS_DIR.glob("*.json"))),
        "submissions_staged": len(SUBMISSION_REGISTRY),
        "live_duels_running": active_count,
        "target_hardware": "NVIDIA GeForce GTX 1050 Ti (GPU 1 via DXGI Shim)",
        "timestamp": datetime.datetime.now().isoformat(),
    })


def main():
    """CLI Entrypoint for the Dashboard."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    initialize_schema()
    port = int(os.environ.get("PORT", 8080))
    print(f"\n=======================================================")
    print(f"[*] KAGGRICULTURE ARENA REPLAY & SUBMISSION DASHBOARD")
    print(f"[*] Servidor ativo em: http://127.0.0.1:{port}")
    print(f"[*] Banco DuckDB: {repo_root / 'kaggriculture' / 'data' / 'arena.duckdb'}")
    print(f"[*] Replays: {REPLAYS_DIR}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
