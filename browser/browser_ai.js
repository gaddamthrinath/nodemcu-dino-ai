/*
 * Dino AI: Play Chrome Dino with Trained Neural Network (model.json)
 * ------------------------------------------------------------------
 * 1. Open chrome://dino (or disconnect internet and press Space).
 * 2. Press F12 -> Console tab.
 * 3. Paste this entire script and press Enter.
 * 4. Click the "📁 Pick model.json" button at top-left (or it will use the embedded trained model).
 *
 * Console Commands:
 *   dinoAI.stats()   -> View run count, high score, and average
 *   dinoAI.stop()    -> Stop the AI
 *   dinoAI.start()   -> Resume playing
 *   dinoAI.load(...) -> Load a model directly via JS object / JSON string
 */
(() => {
    // Stop existing instances if running
    if (window.dinoAI && window.dinoAI.stop) window.dinoAI.stop();
    if (window.dino && window.dino.stop) window.dino.stop();

    // ---------- 1. Find Game Runner ----------
    function findRunner() {
        const isOk = (v) => v && typeof v === 'object' && v.tRex && v.horizon;
        try {
            if (isOk(window.Runner && Runner.instance_)) return Runner.instance_;
            if (typeof Runner.getInstance === 'function' && isOk(Runner.getInstance())) return Runner.getInstance();
            for (const k of Object.getOwnPropertyNames(Runner || {})) {
                try { if (isOk(Runner[k])) return Runner[k]; } catch (e) { }
            }
        } catch (e) { }
        for (const k of Object.getOwnPropertyNames(window)) {
            try { if (isOk(window[k])) return window[k]; } catch (e) { }
        }
        return null;
    }

    const runner = findRunner();
    if (!runner) {
        console.error('[dinoAI] Could not find the Chrome Dino Runner instance. Make sure the Dino game is loaded.');
        return;
    }

    // ---------- 2. Keyboard Simulator ----------
    const KEY = { JUMP: 38, DUCK: 40, START: 32 };
    const CODE = { 38: 'ArrowUp', 40: 'ArrowDown', 32: 'Space', 13: 'Enter' };

    function fire(type, keyCode) {
        const event = new KeyboardEvent(type, {
            keyCode,
            which: keyCode,
            code: CODE[keyCode] || 'Space',
            key: keyCode === 32 ? ' ' : (CODE[keyCode] || 'Space'),
            bubbles: true,
            cancelable: true
        });
        if (event.keyCode !== keyCode) {
            Object.defineProperty(event, 'keyCode', { get: () => keyCode });
            Object.defineProperty(event, 'which', { get: () => keyCode });
        }
        document.dispatchEvent(event);
    }

    const down = (k) => fire('keydown', k);
    const up = (k) => fire('keyup', k);

    // ---------- 3. Obstacle Classification & State Reader ----------
    function classify(o) {
        const cfg = o.typeConfig || {};
        const t = String(cfg.type || '').toLowerCase().replace(/[^a-z]/g, '');
        if (t.includes('small')) return 'cactusSmall';
        if (t.includes('large')) return 'cactusLarge';
        if (t.includes('ptero') || t.includes('bird')) return 'pterodactyl';
        if (cfg.speedOffset !== undefined || (cfg.numFrames || 1) > 1) return 'pterodactyl';
        return ((cfg.height || o.height || 0) >= 45) ? 'cactusLarge' : 'cactusSmall';
    }

    function readState() {
        const dino = runner.tRex;
        const dinoX = dino.xPos;
        const obstacles = (runner.horizon.obstacles || [])
            .filter((o) => o.xPos + o.width > dinoX)
            .sort((a, b) => a.xPos - b.xPos);

        const speed = runner.currentSpeed || 6;
        const ducking = dino.ducking ? 1 : 0;
        const jumping = dino.jumping ? 1 : 0;

        if (!obstacles.length) {
            const dist = 999;
            return {
                dist,
                speed,
                obs_small: 0,
                obs_large: 0,
                obs_bird: 0,
                obs_y: 0,
                obs_h: 0,
                obs_w: 0,
                ducking,
                ttc: dist / Math.max(speed, 0.1),
                jumping,
                rawType: 'none',
            };
        }

        const o = obstacles[0];
        const type = classify(o);
        const dist = Math.max(0, o.xPos - dinoX);
        const height = (o.typeConfig && o.typeConfig.height) || o.height || 0;
        const width = o.width || 0;
        const yPos = o.yPos || 0;

        return {
            dist,
            speed,
            obs_small: type === 'cactusSmall' ? 1 : 0,
            obs_large: type === 'cactusLarge' ? 1 : 0,
            obs_bird: type === 'pterodactyl' ? 1 : 0,
            obs_y: yPos,
            obs_h: height,
            obs_w: width,
            ducking,
            ttc: dist / Math.max(speed, 0.1),
            jumping,
            rawType: type,
        };
    }

    // ---------- 4. Neural Network Inference ----------
    let model = null;

    function predict(state) {
        if (!model || !model.layers) return 0;
        const clip = (v) => (v > 1 ? 1 : v < -1 ? -1 : v);

        // 1. Normalization & clipping
        let x = model.features.map((f, i) => clip((state[f] || 0) / model.scale[i]));

        // 2. Forward pass across layers (Dense + ReLU on hidden layers)
        model.layers.forEach((L, li) => {
            const y = L.b.map((bias, j) => {
                let sum = bias;
                const wRow = L.W[j];
                for (let i = 0; i < x.length; i++) {
                    sum += wRow[i] * x[i];
                }
                return sum;
            });
            x = li < model.layers.length - 1 ? y.map((v) => (v > 0 ? v : 0)) : y;
        });

        // 3. Argmax output
        let best = 0;
        for (let j = 1; j < x.length; j++) {
            if (x[j] > x[best]) best = j;
        }
        return best; // 0 = NOTHING, 1 = JUMP, 2 = DUCK
    }

    // ---------- 5. Key Actuation ----------
    let jumpHeld = false;
    let duckHeld = false;

    function act(action) {
        const t = runner.tRex;

        // Release jump once grounded or transitioning
        if (jumpHeld && !t.jumping) {
            up(KEY.JUMP);
            jumpHeld = false;
        }

        if (action === 2) {
            // DUCK
            if (jumpHeld) { up(KEY.JUMP); jumpHeld = false; }
            if (!duckHeld) { down(KEY.DUCK); duckHeld = true; }
        } else if (action === 1) {
            // JUMP
            if (duckHeld) { up(KEY.DUCK); duckHeld = false; }
            if (!t.jumping && !jumpHeld) {
                down(KEY.JUMP);
                jumpHeld = true;
            }
        } else {
            // NOTHING / RUN
            if (duckHeld) { up(KEY.DUCK); duckHeld = false; }
            if (jumpHeld && !t.jumping) { up(KEY.JUMP); jumpHeld = false; }
        }
    }

    function releaseKeys() {
        if (jumpHeld) { up(KEY.JUMP); jumpHeld = false; }
        if (duckHeld) { up(KEY.DUCK); duckHeld = false; }
    }

    // ---------- 6. Game Loop & Score Tracking ----------
    const scores = [];
    const trace = [];
    let lastScore = 0;
    let wasCrashed = false;
    let crashAt = 0;
    let lastRestartTry = 0;
    let timer = null;

    function currentScore() {
        try {
            return runner.distanceMeter.getActualDistance(Math.ceil(runner.distanceRan));
        } catch (e) {
            return Math.round((runner.distanceRan || 0) * 0.025);
        }
    }

    function tick() {
        if (runner.crashed) {
            if (!wasCrashed) {
                wasCrashed = true;
                crashAt = Date.now();
                releaseKeys();
                scores.push(lastScore);
                const best = Math.max(...scores);
                console.log(`%c[dinoAI] Crashed at score: ${lastScore} | Best: ${best} | Runs: ${scores.length}`, 'color: #ff5555; font-weight: bold;');
                updateHUD('Crashed', lastScore, best);
            }
            const now = Date.now();
            if (now - crashAt > 1200 && now - lastRestartTry > 1000) {
                lastRestartTry = now;
                try {
                    runner.restart();
                } catch (e) {
                    down(KEY.START);
                    up(KEY.START);
                }
            }
            return;
        }

        wasCrashed = false;
        if (!runner.playing) {
            down(KEY.START);
            up(KEY.START);
            return;
        }

        lastScore = currentScore();
        const s = readState();
        const a = predict(s);

        trace.push({
            dist: Math.round(s.dist),
            speed: +s.speed.toFixed(2),
            type: s.rawType,
            action: ['NOTHING', 'JUMP', 'DUCK'][a],
            jumping: s.jumping,
        });
        if (trace.length > 30) trace.shift();

        act(a);
        updateHUD(['RUNNING', 'JUMPING', 'DUCKING'][a], lastScore, scores.length ? Math.max(...scores) : lastScore, s);
    }

    // ---------- 7. UI Overlay (HUD & File Picker) ----------
    let hud = document.getElementById('dino-ai-hud');
    if (hud) hud.remove();

    hud = document.createElement('div');
    hud.id = 'dino-ai-hud';
    Object.assign(hud.style, {
        position: 'fixed',
        top: '12px',
        left: '12px',
        zIndex: '999999',
        background: 'rgba(24, 26, 32, 0.92)',
        backdropFilter: 'blur(8px)',
        color: '#f0f0f0',
        padding: '12px 16px',
        borderRadius: '12px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
        fontFamily: 'Segoe UI, Roboto, system-ui, sans-serif',
        fontSize: '13px',
        lineHeight: '1.4',
        minWidth: '220px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
    });

    hud.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <strong style="font-size:14px; color:#4ade80; display:flex; align-items:center; gap:6px;">
        🦖 Dino AI Agent
      </strong>
      <span id="dino-status-badge" style="font-size:11px; padding:2px 6px; border-radius:6px; background:#374151; color:#9ca3af;">Ready</span>
    </div>
    <div style="display:flex; gap:6px; margin-bottom:10px;">
      <input type="file" id="dino-model-input" accept=".json" style="display:none;" />
      <button id="dino-btn-pick" style="flex:1; background:#2563eb; color:white; border:none; padding:6px 10px; border-radius:6px; font-weight:600; cursor:pointer;">📁 Pick model.json</button>
      <button id="dino-btn-toggle" style="background:#4b5563; color:white; border:none; padding:6px 10px; border-radius:6px; font-weight:600; cursor:pointer;">Pause</button>
    </div>
    <div id="dino-stats-view" style="font-family:monospace; font-size:12px; color:#d1d5db; display:grid; grid-template-columns: 1fr 1fr; gap:4px;">
      <div>Score: <b id="dino-hud-score" style="color:#60a5fa">0</b></div>
      <div>Best: <b id="dino-hud-best" style="color:#fbbf24">0</b></div>
      <div>Speed: <span id="dino-hud-speed">0</span></div>
      <div>TTC: <span id="dino-hud-ttc">0</span></div>
    </div>
  `;
    document.body.appendChild(hud);

    const fileInput = document.getElementById('dino-model-input');
    const btnPick = document.getElementById('dino-btn-pick');
    const btnToggle = document.getElementById('dino-btn-toggle');
    const badge = document.getElementById('dino-status-badge');

    btnPick.onclick = () => fileInput.click();
    fileInput.onchange = async () => {
        try {
            const file = fileInput.files[0];
            if (file) {
                const text = await file.text();
                window.dinoAI.load(text);
                btnPick.style.background = '#059669';
                btnPick.textContent = '✓ Loaded: ' + file.name.slice(0, 12);
            }
        } catch (err) {
            console.error('[dinoAI] Model load error:', err);
            alert('Failed to load model: ' + err.message);
        }
    };

    btnToggle.onclick = () => {
        if (timer) {
            window.dinoAI.stop();
            btnToggle.textContent = 'Resume';
            btnToggle.style.background = '#059669';
            badge.textContent = 'Paused';
            badge.style.background = '#ef4444';
            badge.style.color = '#fff';
        } else {
            window.dinoAI.start();
            btnToggle.textContent = 'Pause';
            btnToggle.style.background = '#4b5563';
        }
    };

    function updateHUD(action, score, best, state) {
        const sEl = document.getElementById('dino-hud-score');
        const bEl = document.getElementById('dino-hud-best');
        const spEl = document.getElementById('dino-hud-speed');
        const ttcEl = document.getElementById('dino-hud-ttc');

        if (sEl) sEl.textContent = score || 0;
        if (bEl) bEl.textContent = best || 0;
        if (spEl && state) spEl.textContent = state.speed.toFixed(1);
        if (ttcEl && state) ttcEl.textContent = state.ttc < 900 ? state.ttc.toFixed(1) : '∞';

        if (badge && timer) {
            badge.textContent = action;
            if (action === 'JUMPING') {
                badge.style.background = '#f59e0b';
                badge.style.color = '#000';
            } else if (action === 'DUCKING') {
                badge.style.background = '#8b5cf6';
                badge.style.color = '#fff';
            } else if (action === 'Crashed') {
                badge.style.background = '#ef4444';
                badge.style.color = '#fff';
            } else {
                badge.style.background = '#10b981';
                badge.style.color = '#fff';
            }
        }
    }

    // ---------- 8. Public API & Start ----------
    window.dinoAI = {
        load(json) {
            const parsed = typeof json === 'string' ? JSON.parse(json) : json;
            if (!parsed.features || !parsed.scale || !parsed.layers) {
                throw new Error('Invalid model format. Expected features, scale, and layers.');
            }
            model = parsed;
            badge.textContent = 'Playing';
            badge.style.background = '#10b981';
            badge.style.color = '#fff';
            if (!timer) timer = setInterval(tick, 16);
            console.log('%c[dinoAI] Model loaded successfully! AI is playing.', 'color: #10b981; font-weight: bold;');
        },
        start() {
            if (!timer) {
                timer = setInterval(tick, 16);
                badge.textContent = 'Playing';
                badge.style.background = '#10b981';
                badge.style.color = '#fff';
                console.log('[dinoAI] Started');
            }
        },
        stop() {
            if (timer) {
                clearInterval(timer);
                timer = null;
            }
            releaseKeys();
            console.log('[dinoAI] Stopped');
        },
        stats() {
            if (!scores.length) {
                console.log(`[dinoAI] Current run score: ${lastScore} (no finished runs yet)`);
                return;
            }
            const avg = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
            console.log(`[dinoAI] Runs: ${scores.length} | Best: ${Math.max(...scores)} | Avg: ${avg} | Recent: [${scores.slice(-8).join(', ')}]`);
        },
        scores: () => scores,
    };

    console.log('%c[dinoAI] Initialized! Click "📁 Pick model.json" at the top-left to select your model.', 'color: #38bdf8; font-weight: bold;');
})();
