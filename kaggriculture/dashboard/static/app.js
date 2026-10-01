// Kaggriculture Arena Dashboard Frontend Application

let livePollInterval = null;
let currentReplayId = null;
let homeFilterTimeout = null;
let fullFilterTimeout = null;

document.addEventListener("DOMContentLoaded", () => {
    // Restore saved tab or default to Leaderboard & Podium
    let initialTab = "leaderboard";
    try {
        initialTab = localStorage.getItem("kaggriculture_active_tab") || "leaderboard";
    } catch (e) {}

    switchTab(initialTab);
    loadLeaderboard();
    loadHomeMatches();
    loadFullMatches();
    loadLiveDuels();
    setupDropzone();

    // Poll live duels every 3 seconds
    livePollInterval = setInterval(loadLiveDuels, 3000);
});

// TAB SWITCHING
function switchTab(tabName) {
    if (!tabName) tabName = "leaderboard";
    try {
        localStorage.setItem("kaggriculture_active_tab", tabName);
    } catch (e) {}

    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    const tabBtn = document.getElementById(`tabbtn-${tabName}`);
    if (tabBtn) tabBtn.classList.add("active");

    const content = document.getElementById(`tab-${tabName}`);
    if (content) content.classList.add("active");

    if (tabName === "leaderboard") {
        loadLeaderboard();
        loadHomeMatches();
    } else if (tabName === "matches") {
        loadFullMatches();
    } else if (tabName === "live") {
        loadLiveDuels();
    }
}

// 1. LEADERBOARD & PODIUM
async function loadLeaderboard() {
    try {
        const res = await fetch("/api/leaderboard");
        const data = await res.json();
        const tbody = document.getElementById("leaderboard-tbody");
        if (!tbody) return;

        const list = data.leaderboard || [];
        tbody.innerHTML = list.map((a, idx) => {
            const winRate = a.matches_played > 0 
                ? `${((a.wins / a.matches_played) * 100).toFixed(1)}%` 
                : "0.0%";

            const posBadge = idx === 0 ? "🥇 1º" : idx === 1 ? "🥈 2º" : idx === 2 ? "🥉 3º" : `${idx + 1}º`;
            const cleanId = a.agent_id.replace(/^llm_/, "").replace(/_v1$/, "");

            return `
                <tr>
                    <td><strong>${posBadge}</strong></td>
                    <td><code>${escapeHtml(a.name || a.agent_id)}</code></td>
                    <td>
                        <span class="tile-stage-badge clickable" onclick="filterByLeague('${escapeHtml(a.league)}')" title="Filtrar todas as partidas da ${escapeHtml(a.league)}">
                            🔍 ${escapeHtml(a.league)}
                        </span>
                    </td>
                    <td><strong style="color:var(--accent-amber);">${a.elo.toFixed(1)}</strong></td>
                    <td>${a.matches_played}</td>
                    <td>${a.wins} - ${a.draws} - ${a.losses}</td>
                    <td>${winRate}</td>
                    <td>${a.total_coins.toLocaleString()} moedas</td>
                    <td style="text-align: right; white-space: nowrap;">
                        <button class="btn btn-sm btn-primary" onclick="filterByAgentAndScroll('${cleanId}')" title="Ver todos os 40 confrontos de ${escapeHtml(a.name)}">
                            <span>🎮</span> Ver Partidas (${a.matches_played})
                        </button>
                    </td>
                </tr>
            `;
        }).join("");
    } catch (err) {
        console.error("Erro carregando leaderboard:", err);
    }
}

// FILTER BY AGENT & SCROLL TO GAMES LIST
function filterByAgentAndScroll(agentId) {
    // 1. Update inputs
    const homeInput = document.getElementById("home-filter-agent");
    if (homeInput) homeInput.value = agentId;

    const fullInput = document.getElementById("match-filter-agent");
    if (fullInput) fullInput.value = agentId;

    // 2. Update quick chips
    syncQuickChips(agentId);

    // 3. Reload home matches
    loadHomeMatches();
    loadFullMatches();

    // 4. Scroll smoothly to the matches section
    const target = document.getElementById("leaderboard-matches-section");
    if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

function filterByAgent(agentId) {
    filterByAgentAndScroll(agentId);
}

// FILTER BY LEAGUE
function filterByLeague(leagueName) {
    let val = "";
    if (leagueName.includes("48") || leagueName.toLowerCase().includes("hybrid")) {
        val = "h48";
    } else if (leagueName.includes("24") || leagueName.toLowerCase().includes("macro")) {
        val = "h24";
    }

    const homeLeagueSelect = document.getElementById("home-filter-league");
    if (homeLeagueSelect) homeLeagueSelect.value = val;

    const fullLeagueSelect = document.getElementById("match-filter-league");
    if (fullLeagueSelect) fullLeagueSelect.value = val;

    loadHomeMatches();
    loadFullMatches();

    const target = document.getElementById("leaderboard-matches-section");
    if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

// QUICK CHIP SELECTORS
function selectQuickHomeAgent(agentId, btnEl) {
    const parent = document.getElementById("home-quick-chips");
    if (parent) {
        parent.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
        if (btnEl) btnEl.classList.add("active");
    }

    const homeInput = document.getElementById("home-filter-agent");
    if (homeInput) homeInput.value = agentId;

    loadHomeMatches();
}

function selectQuickFullAgent(agentId, btnEl) {
    const parent = document.getElementById("full-quick-chips");
    if (parent) {
        parent.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
        if (btnEl) btnEl.classList.add("active");
    }

    const fullInput = document.getElementById("match-filter-agent");
    if (fullInput) fullInput.value = agentId;

    loadFullMatches();
}

function syncQuickChips(agentId) {
    ["home-quick-chips", "full-quick-chips"].forEach(containerId => {
        const container = document.getElementById(containerId);
        if (!container) return;
        const chips = container.querySelectorAll(".chip");
        chips.forEach(c => c.classList.remove("active"));
        if (!agentId) {
            chips[0]?.classList.add("active");
        } else {
            let found = false;
            chips.forEach(c => {
                if (c.getAttribute("onclick") && c.getAttribute("onclick").includes(`'${agentId}'`)) {
                    c.classList.add("active");
                    found = true;
                }
            });
            if (!found) chips[0]?.classList.add("active");
        }
    });
}

// 2. CONNECTED HOME MATCHES LIST
async function loadHomeMatches() {
    const stage = document.getElementById("home-filter-stage")?.value || "all";
    const league = document.getElementById("home-filter-league")?.value || "";
    const agent = document.getElementById("home-filter-agent")?.value || "";

    try {
        const url = `/api/matches?limit=150&stage=${encodeURIComponent(stage)}&league=${encodeURIComponent(league)}&agent=${encodeURIComponent(agent)}`;
        const res = await fetch(url);
        const data = await res.json();

        const tbody = document.getElementById("home-matches-tbody");
        if (!tbody) return;

        if (!data.matches || data.matches.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:32px; color:var(--text-muted);">Nenhuma partida encontrada com os filtros informados.</td></tr>`;
            return;
        }

        tbody.innerHTML = data.matches.map((m, idx) => renderMatchRow(m, idx)).join("");
    } catch (err) {
        console.error("Erro carregando partidas da home:", err);
    }
}

function filterHomeMatchesDebounced() {
    clearTimeout(homeFilterTimeout);
    homeFilterTimeout = setTimeout(loadHomeMatches, 250);
}

function resetHomeMatchesFilter() {
    const stage = document.getElementById("home-filter-stage");
    if (stage) stage.value = "all";

    const league = document.getElementById("home-filter-league");
    if (league) league.value = "";

    const agent = document.getElementById("home-filter-agent");
    if (agent) agent.value = "";

    syncQuickChips("");
    loadHomeMatches();
}

// 3. DEDICATED FULL MATCHES LIST (TAB 2)
async function loadFullMatches() {
    const stage = document.getElementById("match-filter-stage")?.value || "all";
    const league = document.getElementById("match-filter-league")?.value || "";
    const agent = document.getElementById("match-filter-agent")?.value || "";

    try {
        const url = `/api/matches?limit=150&stage=${encodeURIComponent(stage)}&league=${encodeURIComponent(league)}&agent=${encodeURIComponent(agent)}`;
        const res = await fetch(url);
        const data = await res.json();

        const tbody = document.getElementById("matches-tbody");
        const empty = document.getElementById("matches-empty");
        const countBadge = document.getElementById("matches-count");

        if (countBadge) countBadge.textContent = data.total || 0;
        if (!tbody) return;

        if (!data.matches || data.matches.length === 0) {
            tbody.innerHTML = "";
            if (empty) empty.style.display = "block";
            return;
        }
        if (empty) empty.style.display = "none";

        tbody.innerHTML = data.matches.map((m, idx) => renderMatchRow(m, idx)).join("");
    } catch (err) {
        console.error("Erro carregando todas as partidas:", err);
    }
}

function filterFullMatchesDebounced() {
    clearTimeout(fullFilterTimeout);
    fullFilterTimeout = setTimeout(loadFullMatches, 250);
}

function resetFullMatchesFilter() {
    const stage = document.getElementById("match-filter-stage");
    if (stage) stage.value = "all";

    const league = document.getElementById("match-filter-league");
    if (league) league.value = "";

    const agent = document.getElementById("match-filter-agent");
    if (agent) agent.value = "";

    syncQuickChips("");
    loadFullMatches();
}

// REUSABLE MATCH ROW BUILDER
function renderMatchRow(m, idx) {
    const p0Win = m.winner === m.agent_p0;
    const p1Win = m.winner === m.agent_p1;
    const isDraw = m.winner === "Draw";

    const stageBadge = `<span class="tile-stage-badge">${escapeHtml(m.stage)} (${m.steps}s)</span>`;
    const epochBadge = `<span style="font-family:var(--font-mono); font-weight:700; color:#8b5cf6;">Época ${m.epoch}</span>`;

    const winnerDisplay = isDraw 
        ? `<span style="color:var(--text-secondary); font-weight:600;">Empate</span>`
        : `<span class="winner-tag">👑 ${escapeHtml(m.winner)}</span>`;

    return `
        <tr class="match-row">
            <td style="font-family:var(--font-mono); color:var(--text-muted); font-size:11px;">#${idx + 1}</td>
            <td>${epochBadge}</td>
            <td>${stageBadge}</td>
            <td>
                <strong class="${p0Win ? 'winner-tag' : ''}">${escapeHtml(m.agent_p0)}</strong>
                <span style="color:var(--text-muted); margin: 0 4px;">vs</span>
                <strong class="${p1Win ? 'winner-tag' : ''}">${escapeHtml(m.agent_p1)}</strong>
            </td>
            <td>
                <span class="coin-pill">${m.p0_bank.toLocaleString()} vs ${m.p1_bank.toLocaleString()}</span>
            </td>
            <td>${winnerDisplay}</td>
            <td style="font-family:var(--font-mono); font-size:11px; color: ${m.margin > 0 ? '#10b981' : 'var(--text-muted)'};">
                ${m.margin > 0 ? '+' + m.margin.toLocaleString() : '0'}
            </td>
            <td>
                <span style="font-size:11px; color:var(--text-muted);">${escapeHtml(m.bracket)}</span>
            </td>
            <td style="text-align: right; white-space: nowrap;">
                <button class="btn btn-sm btn-primary" onclick="openZoomModal('${m.match_id}', '${escapeHtml(m.agent_p0)} vs ${escapeHtml(m.agent_p1)}', 'Época ${m.epoch} • ${escapeHtml(m.stage)} (${m.steps} passos)')" title="Abrir player interativo">
                    <span>▶️</span> Assistir Replay
                </button>
                <button class="btn btn-sm btn-secondary" onclick="window.open('/player/${m.match_id}', '_blank')" title="Abrir em Nova Janela">
                    <span>↗</span>
                </button>
                <button class="btn btn-sm btn-outline" onclick="window.open('/api/replays/${m.match_id}', '_blank')" title="Baixar JSON">
                    <span>⬇</span>
                </button>
            </td>
        </tr>
    `;
}

// 4. LIVE ARENA MONITOR (TILES VIEW - STRICTLY WHEN ARENA IS RUNNING)
async function loadLiveDuels() {
    try {
        const res = await fetch("/api/live");
        const data = await res.json();

        const activeContainer = document.getElementById("live-active-container");
        const idleContainer = document.getElementById("live-idle-container");
        const liveBadge = document.getElementById("live-badge");
        const liveIndicator = document.getElementById("live-indicator");

        if (data.active && data.duels && data.duels.length > 0) {
            if (activeContainer) activeContainer.style.display = "block";
            if (idleContainer) idleContainer.style.display = "none";
            if (liveBadge) {
                liveBadge.style.display = "inline-block";
                liveBadge.textContent = data.has_running ? "AO VIVO" : `${data.count} RECENTES`;
            }
            if (liveIndicator) liveIndicator.style.display = "inline-block";

            renderLiveTiles(data.duels);
        } else {
            if (activeContainer) activeContainer.style.display = "none";
            if (idleContainer) idleContainer.style.display = "block";
            if (liveBadge) liveBadge.style.display = "none";
            if (liveIndicator) liveIndicator.style.display = "none";
        }
    } catch (err) {
        console.error("Erro consultando duelos ao vivo:", err);
    }
}

function renderLiveTiles(duels) {
    const grid = document.getElementById("live-tiles-grid");
    if (!grid) return;

    const currentDuelIds = new Set(duels.map(d => d.duel_id));

    // Remove any tiles that are no longer active
    Array.from(grid.children).forEach(child => {
        const id = child.id.replace("tile-", "");
        if (!currentDuelIds.has(id)) {
            child.remove();
        }
    });

    // Update or create each duel tile
    duels.forEach(d => {
        let tile = document.getElementById(`tile-${d.duel_id}`);
        const isRunning = d.status === "running" || d.status === "starting";
        const isError = d.status === "error";
        const hasReplay = !!d.replay_id;

        const p0IsWinner = d.winner === d.agent_p0;
        const p1IsWinner = d.winner === d.agent_p1;

        const stageLabel = d.steps === 72 ? "Sprint (72s)" :
                           d.steps === 144 ? "Expansion (144s)" :
                           d.steps === 240 ? "Scaling (240s)" :
                           d.steps === 720 ? "Full Season (720s)" : `${d.steps} passos`;

        const repId = d.replay_id || d.duel_id;

        if (!tile) {
            tile = document.createElement("div");
            tile.id = `tile-${d.duel_id}`;
            tile.className = `game-tile ${isRunning ? 'live' : ''}`;

            tile.innerHTML = `
                <div class="tile-header">
                    <span class="tile-stage-badge">${stageLabel}</span>
                    <span class="tile-status-slot" id="tile-status-${d.duel_id}">
                        ${isError 
                            ? `<span class="tile-stage-badge" style="color:#ef4444; border-color:rgba(239,68,68,0.4);">⚠️ ERRO</span>`
                            : (isRunning 
                                ? `<span class="live-pill"><span class="dot red"></span> AO VIVO NA GPU</span>` 
                                : `<span class="tile-stage-badge" style="color: #10b981; border-color: rgba(16,185,129,0.3);">CONCLUÍDO</span>`)
                        }
                    </span>
                </div>

                <div class="tile-vs">
                    <div class="player-box p0">
                        <span class="player-name ${p0IsWinner ? 'winner' : ''}" id="tile-p0-name-${d.duel_id}">
                            ${p0IsWinner ? '👑 ' : ''}${escapeHtml(d.agent_p0)}
                        </span>
                        <span class="player-coins" id="tile-p0-bank-${d.duel_id}">${(d.p0_bank || 0).toLocaleString()} moedas</span>
                    </div>

                    <span class="vs-divider">VS</span>

                    <div class="player-box p1">
                        <span class="player-name ${p1IsWinner ? 'winner' : ''}" id="tile-p1-name-${d.duel_id}">
                            ${escapeHtml(d.agent_p1)}${p1IsWinner ? ' 👑' : ''}
                        </span>
                        <span class="player-coins" id="tile-p1-bank-${d.duel_id}">${(d.p1_bank || 0).toLocaleString()} moedas</span>
                    </div>
                </div>

                <div class="tile-viewport" id="tile-viewport-${d.duel_id}">
                    ${hasReplay && !isError
                        ? `<iframe src="/player/${repId}?mini=true" loading="lazy" scrolling="no"></iframe>
                           <div class="tile-overlay-click" onclick="openZoomModal('${repId}', '${escapeHtml(d.agent_p0)} vs ${escapeHtml(d.agent_p1)}', '${stageLabel}')">
                                <span class="tile-overlay-btn">🔍 Expandir / Zoom</span>
                           </div>` 
                        : (isError
                            ? `<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#ef4444; padding:16px; text-align:center; gap:8px;">
                                   <span style="font-size:24px;">⚠️</span>
                                   <span style="font-size:12px; font-weight:600;">Falha na execução do agente</span>
                                   <span style="font-size:10px; font-family:var(--font-mono); color:#94a3b8;">${escapeHtml(d.error || 'Erro desconhecido')}</span>
                               </div>`
                            : `<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#94a3b8; gap:10px;">
                                   <div class="spinner"></div>
                                   <span style="font-size:12px; font-family:var(--font-mono);">Iniciando motor na GTX 1050 Ti...</span>
                               </div>`)
                    }
                </div>

                <div class="tile-footer">
                    <span class="tile-date" id="tile-footer-status-${d.duel_id}">
                        ${isRunning ? `Simulação: <strong>Passo ${d.current_step || 0}/${d.steps}</strong>` : (isError ? `Erro: ${escapeHtml(d.error || '')}` : `Vencedor: <strong>${escapeHtml(d.winner || 'Empate')}</strong>`)}
                    </span>
                    <div class="tile-footer-actions">
                        ${hasReplay ? `
                            <button class="btn btn-sm btn-secondary" onclick="window.open('/player/${repId}', '_blank')">
                                <span>↗</span> Nova Janela
                            </button>
                        ` : ''}
                    </div>
                </div>
            `;
            grid.appendChild(tile);
        } else {
            // Update tile in-place without reloading iframe
            if (isRunning) {
                tile.classList.add("live");
            } else {
                tile.classList.remove("live");
            }

            const statusSlot = document.getElementById(`tile-status-${d.duel_id}`);
            if (statusSlot) {
                statusSlot.innerHTML = isError
                    ? `<span class="tile-stage-badge" style="color:#ef4444; border-color:rgba(239,68,68,0.4);">⚠️ ERRO</span>`
                    : (isRunning 
                        ? `<span class="live-pill"><span class="dot red"></span> AO VIVO (${d.current_step || 0}/${d.steps})</span>`
                        : `<span class="tile-stage-badge" style="color: #10b981; border-color: rgba(16,185,129,0.3);">CONCLUÍDO</span>`);
            }

            const p0BankEl = document.getElementById(`tile-p0-bank-${d.duel_id}`);
            if (p0BankEl) p0BankEl.textContent = `${(d.p0_bank || 0).toLocaleString()} moedas`;

            const p1BankEl = document.getElementById(`tile-p1-bank-${d.duel_id}`);
            if (p1BankEl) p1BankEl.textContent = `${(d.p1_bank || 0).toLocaleString()} moedas`;

            const p0NameEl = document.getElementById(`tile-p0-name-${d.duel_id}`);
            if (p0NameEl) {
                p0NameEl.className = `player-name ${p0IsWinner ? 'winner' : ''}`;
                p0NameEl.textContent = `${p0IsWinner ? '👑 ' : ''}${d.agent_p0}`;
            }

            const p1NameEl = document.getElementById(`tile-p1-name-${d.duel_id}`);
            if (p1NameEl) {
                p1NameEl.className = `player-name ${p1IsWinner ? 'winner' : ''}`;
                p1NameEl.textContent = `${d.agent_p1}${p1IsWinner ? ' 👑' : ''}`;
            }

            const footerStatus = document.getElementById(`tile-footer-status-${d.duel_id}`);
            if (footerStatus) {
                footerStatus.innerHTML = isRunning 
                    ? `Simulação: <strong>Passo ${d.current_step || 0}/${d.steps}</strong>` 
                    : (isError ? `Erro: ${escapeHtml(d.error || '')}` : `Vencedor: <strong>${escapeHtml(d.winner || 'Empate')}</strong>`);
            }
        }
    });
}

async function launchLiveRound(steps = 72) {
    try {
        const res = await fetch("/api/run_live_round", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ steps: steps })
        });
        const data = await res.json();
        switchTab("live");
        loadLiveDuels();
    } catch (err) {
        alert("Erro iniciando rodada ao vivo: " + err.message);
    }
}

// 5. ZOOM MODAL & PLAYER
window.addEventListener("message", (event) => {
    if (event.data && event.data.type === "KAGGRICULTURE_LIVE_STEP") {
        updateGameHUD(event.data);
    }
});

function updateGameHUD(data) {
    const dayEl = document.getElementById("hud-day-val");
    const turnEl = document.getElementById("hud-turn-val");
    const stepEl = document.getElementById("hud-step-val");
    const totalEl = document.getElementById("hud-total-steps");
    if (dayEl) dayEl.textContent = (data.totalDays ? `${data.day} / ${data.totalDays}` : data.day);
    if (turnEl) turnEl.textContent = `${data.turn || data.hour || 1} / 24`;
    if (stepEl) stepEl.textContent = data.currentStep !== undefined ? data.currentStep : data.step;
    if (totalEl) totalEl.textContent = data.totalSteps;

    const p0Coins = document.getElementById("hud-p0-coins");
    const p1Coins = document.getElementById("hud-p1-coins");
    if (p0Coins) p0Coins.textContent = Math.round(data.p0_bank || 0).toLocaleString();
    if (p1Coins) p1Coins.textContent = Math.round(data.p1_bank || 0).toLocaleString();

    // Commodities prices
    const prices = data.prices || {};
    const commodities = [
        { id: "mkt-wheat", val: prices.WHEAT },
        { id: "mkt-carrot", val: prices.CARROT },
        { id: "mkt-tomato", val: prices.TOMATO },
        { id: "mkt-strawberry", val: prices.STRAWBERRY },
        { id: "mkt-melon", val: prices.MELON },
        { id: "mkt-egg", val: prices.EGG },
        { id: "mkt-milk", val: prices.MILK },
        { id: "mkt-wool", val: prices.WOOL },
    ];
    commodities.forEach(c => {
        const el = document.getElementById(c.id);
        if (el) el.textContent = c.val !== undefined ? c.val : "--";
    });

    // Inventories
    renderInventory("hud-p0-inventory", data.p0_inv || { seeds: data.p0_seeds, shed: data.p0_shed });
    renderInventory("hud-p1-inventory", data.p1_inv || { seeds: data.p1_seeds, shed: data.p1_shed });
}

function renderInventory(containerId, invData = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const items = [];
    const seedIcons = { WHEAT: "🌾", CARROT: "🥕", TOMATO: "🍅", STRAWBERRY: "🍓", MELON: "🍉" };
    const shedIcons = { WHEAT: "🌾", CARROT: "🥕", TOMATO: "🍅", STRAWBERRY: "🍓", MELON: "🍉", EGG: "🥚", MILK: "🥛", WOOL: "🐑" };

    const carried = invData.carried || {};
    const seeds = invData.seeds || {};
    const shed = invData.shed || {};

    for (const [k, v] of Object.entries(carried)) {
        if (v > 0) items.push(`${seedIcons[k] || "🎒"} ${k.toLowerCase()} x${v}`);
    }
    for (const [k, v] of Object.entries(seeds)) {
        if (v > 0) items.push(`${seedIcons[k] || "🌱"} sem. ${k.toLowerCase()} x${v}`);
    }
    for (const [k, v] of Object.entries(shed)) {
        if (v > 0) items.push(`${shedIcons[k] || "📦"} ${k.toLowerCase()}: ${v}`);
    }

    if (items.length === 0) {
        container.innerHTML = `<span class="hud-inv-pill" style="opacity:0.6;">Bolsa vazia</span>`;
    } else {
        container.innerHTML = items.map(it => `<span class="hud-inv-pill">${escapeHtml(it)}</span>`).join("");
    }
}

function openZoomModal(replayId, title, meta) {
    currentReplayId = replayId;
    const modal = document.getElementById("zoom-modal");
    const iframe = document.getElementById("modal-iframe");
    const titleEl = document.getElementById("modal-match-title");
    const metaEl = document.getElementById("modal-match-meta");
    const popoutBtn = document.getElementById("btn-popout");

    if (titleEl) titleEl.textContent = title || `Replay ${replayId}`;
    if (metaEl) metaEl.textContent = meta || "";
    if (popoutBtn) popoutBtn.onclick = () => window.open(`/player/${replayId}`, "_blank");

    // Parse player names from title
    let p0 = "Player 0";
    let p1 = "Player 1";
    if (title && title.includes(" vs ")) {
        const parts = title.split(" vs ");
        p0 = parts[0].trim();
        p1 = parts[1].trim();
    }
    const p0NameEl = document.getElementById("hud-p0-name");
    const p1NameEl = document.getElementById("hud-p1-name");
    if (p0NameEl) p0NameEl.textContent = p0;
    if (p1NameEl) p1NameEl.textContent = p1;

    // Reset temporary placeholders
    const p0Coins = document.getElementById("hud-p0-coins");
    const p1Coins = document.getElementById("hud-p1-coins");
    if (p0Coins) p0Coins.textContent = "...";
    if (p1Coins) p1Coins.textContent = "...";

    if (iframe) iframe.src = `/player/${replayId}`;
    if (modal) modal.style.display = "flex";
}

function closeZoomModal() {
    const modal = document.getElementById("zoom-modal");
    const iframe = document.getElementById("modal-iframe");
    if (iframe) iframe.src = "about:blank";
    if (modal) modal.style.display = "none";
}

// 6. UPLOAD REPLAY MODAL
function openUploadModal() {
    const m = document.getElementById("upload-modal");
    if (m) m.style.display = "flex";
}
function closeUploadModal() {
    const m = document.getElementById("upload-modal");
    if (m) m.style.display = "none";
    const s = document.getElementById("upload-status");
    if (s) s.style.display = "none";
}

async function handleReplayUpload(input) {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const status = document.getElementById("upload-status");

    if (status) {
        status.style.display = "block";
        status.textContent = "Validando e indexando replay JSON no DuckDB...";
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/upload_replay", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.status === "success") {
            if (status) status.textContent = "Replay indexado com sucesso!";
            setTimeout(() => {
                closeUploadModal();
                loadHomeMatches();
                loadFullMatches();
                openZoomModal(data.replay_id, data.title, `${data.winner} venceu (${data.steps} passos)`);
            }, 600);
        } else {
            if (status) status.textContent = `Erro: ${data.error || "Falha no envio"}`;
        }
    } catch (err) {
        if (status) status.textContent = `Erro: ${err.message}`;
    }
}

// 7. SUBMISSION SANDBOX
function setupDropzone() {
    const dropzone = document.getElementById("sub-dropzone");
    if (!dropzone) return;

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, e => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, e => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        }, false);
    });

    dropzone.addEventListener("drop", e => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleSubFile(dt.files[0]);
        }
    });
}

function handleSubFileSelect(input) {
    if (input.files && input.files.length > 0) {
        handleSubFile(input.files[0]);
    }
}

let selectedSubFile = null;

function handleSubFile(file) {
    selectedSubFile = file;
    const info = document.getElementById("sub-file-info");
    const name = document.getElementById("sub-file-name");
    const size = document.getElementById("sub-file-size");
    const btn = document.getElementById("btn-run-match");

    if (info) info.style.display = "inline-flex";
    if (name) name.textContent = file.name;
    if (size) size.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    if (btn) btn.disabled = false;
}

let currentSandboxDuel = null;
let currentSandboxOpponent = "";
let currentSandboxSteps = 72;
let sandboxPollTimer = null;
let sandboxStartTime = null;

async function handleRunSubmission(e) {
    e.preventDefault();
    if (!selectedSubFile) return;

    const btn = document.getElementById("btn-run-match");
    if (btn) btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append("file", selectedSubFile);

        const uploadRes = await fetch("/api/upload_submission", {
            method: "POST",
            body: formData
        });
        const uploadData = await uploadRes.json();
        if (uploadData.status !== "success") {
            throw new Error(uploadData.error || "Falha no upload do arquivo.");
        }

        const subId = uploadData.submission_id;
        const opponent = document.getElementById("sub-opponent").value;
        const steps = parseInt(document.getElementById("sub-steps").value);
        const seedVal = document.getElementById("sub-seed").value;
        const seed = seedVal ? parseInt(seedVal) : null;

        const matchRes = await fetch("/api/run_match", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                agent_p0: subId,
                agent_p1: opponent,
                steps: steps,
                seed: seed
            })
        });

        const matchData = await matchRes.json();
        if (matchData.status !== "started") {
            throw new Error(matchData.message || "Erro ao iniciar duelo.");
        }

        currentSandboxDuel = matchData.duel_id;
        currentSandboxOpponent = opponent;
        currentSandboxSteps = steps;
        sandboxStartTime = Date.now();

        // Reveal and initialize Sandbox live progress card in-place
        const statusCard = document.getElementById("sandbox-duel-status");
        if (statusCard) {
            statusCard.style.display = "flex";
            const titleEl = document.getElementById("sandbox-match-title");
            const badgeEl = document.getElementById("sandbox-status-badge");
            const p0NameEl = document.getElementById("sb-p0-name");
            const p1NameEl = document.getElementById("sb-p1-name");
            const p0CoinsEl = document.getElementById("sb-p0-coins");
            const p1CoinsEl = document.getElementById("sb-p1-coins");
            const fillEl = document.getElementById("sandbox-progress-fill");
            const stepTextEl = document.getElementById("sandbox-step-text");
            const timeEl = document.getElementById("sandbox-time-elapsed");

            if (titleEl) titleEl.textContent = `⚔️ Duelo Ao Vivo: ${selectedSubFile.name} vs ${opponent}`;
            if (badgeEl) {
                badgeEl.textContent = "AO VIVO NA GPU";
                badgeEl.className = "badge";
                badgeEl.style.color = "#ef4444";
                badgeEl.style.borderColor = "rgba(239,68,68,0.4)";
            }
            if (p0NameEl) p0NameEl.textContent = `Sua Submissão (${selectedSubFile.name})`;
            if (p1NameEl) p1NameEl.textContent = opponent;
            if (p0CoinsEl) p0CoinsEl.textContent = "💰 0 moedas";
            if (p1CoinsEl) p1CoinsEl.textContent = "💰 0 moedas";
            if (fillEl) fillEl.style.width = "0%";
            if (stepTextEl) stepTextEl.textContent = `Passo 0 / ${steps} (0%)`;
            if (timeEl) timeEl.textContent = "Tempo: 0s";

            statusCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }

        // Start live poller for this sandbox match
        if (sandboxPollTimer) clearInterval(sandboxPollTimer);
        sandboxPollTimer = setInterval(pollSandboxProgress, 500);

        // Also trigger live duels update in background
        loadLiveDuels();

    } catch (err) {
        alert("Erro iniciando teste: " + err.message);
        if (btn) btn.disabled = false;
    }
}

async function pollSandboxProgress() {
    if (!currentSandboxDuel) {
        if (sandboxPollTimer) clearInterval(sandboxPollTimer);
        return;
    }

    try {
        const res = await fetch(`/api/replay_stream/${currentSandboxDuel}`);
        if (!res.ok) return;
        const data = await res.json();

        const fillEl = document.getElementById("sandbox-progress-fill");
        const stepTextEl = document.getElementById("sandbox-step-text");
        const timeEl = document.getElementById("sandbox-time-elapsed");
        const badgeEl = document.getElementById("sandbox-status-badge");
        const p0CoinsEl = document.getElementById("sb-p0-coins");
        const p1CoinsEl = document.getElementById("sb-p1-coins");
        const runBtn = document.getElementById("btn-run-match");

        const currStep = data.current_step || 0;
        const totalSteps = data.total_steps || currentSandboxSteps;
        const pct = Math.min(100, Math.round((currStep / totalSteps) * 100));

        if (fillEl) fillEl.style.width = `${pct}%`;
        if (stepTextEl) stepTextEl.textContent = `Passo ${currStep} / ${totalSteps} (${pct}%)`;

        if (sandboxStartTime && timeEl) {
            const elapsed = Math.round((Date.now() - sandboxStartTime) / 1000);
            timeEl.textContent = `Tempo: ${elapsed}s`;
        }

        // Check if steps array has latest coins
        if (data.steps && data.steps.length > 0) {
            const lastSt = data.steps[data.steps.length - 1];
            if (lastSt && lastSt.length >= 2) {
                const r0 = lastSt[0].reward !== undefined ? lastSt[0].reward : (lastSt[0].observation?.farms?.[0]?.money || 0);
                const r1 = lastSt[1].reward !== undefined ? lastSt[1].reward : (lastSt[1].observation?.farms?.[1]?.money || 0);
                if (p0CoinsEl) p0CoinsEl.textContent = `💰 ${Math.round(r0).toLocaleString()} moedas`;
                if (p1CoinsEl) p1CoinsEl.textContent = `💰 ${Math.round(r1).toLocaleString()} moedas`;
            }
        }

        if (data.error) {
            clearInterval(sandboxPollTimer);
            if (badgeEl) {
                badgeEl.textContent = `⚠️ ERRO: ${data.error}`;
                badgeEl.style.color = "#ef4444";
                badgeEl.style.borderColor = "rgba(239,68,68,0.5)";
            }
            if (runBtn) runBtn.disabled = false;
        } else if (data.is_done) {
            clearInterval(sandboxPollTimer);
            if (badgeEl) {
                badgeEl.textContent = "CONCLUÍDO";
                badgeEl.style.color = "#10b981";
                badgeEl.style.borderColor = "rgba(16,185,129,0.5)";
            }
            if (runBtn) runBtn.disabled = false;
            // Also refresh match lists
            loadHomeMatches();
            loadFullMatches();
        }
    } catch (e) {
        console.error("Erro no polling da submissão:", e);
    }
}

function openCurrentSandboxModal() {
    if (!currentSandboxDuel) return;
    openZoomModal(
        currentSandboxDuel,
        `Sua Submissão vs ${currentSandboxOpponent}`,
        `Duelo Sandbox Ao Vivo (${currentSandboxSteps} passos)`
    );
}

function openCurrentSandboxPopout() {
    if (!currentSandboxDuel) return;
    window.open(`/player/${currentSandboxDuel}`, "_blank");
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
