/*
 * Dino AI: Web Serial Bridge for NodeMCU (No Python needed!)
 * -----------------------------------------------------------
 * 1. Flash dino_nodemcu/dino_nodemcu.ino to your NodeMCU using Arduino IDE.
 * 2. Keep NodeMCU connected via USB.
 * 3. Open chrome://dino -> F12 -> Console -> paste this script -> Enter.
 * 4. Click the "🔌 Connect NodeMCU" button at the top-left and select your NodeMCU COM port.
 * 5. NodeMCU is now controlling the Dino game live!
 */
(() => {
  if (window.dinoMCU && window.dinoMCU.stop) window.dinoMCU.stop();

  function findRunner() {
    const ok = (v) => v && typeof v === 'object' && v.tRex && v.horizon;
    try {
      if (ok(window.Runner && Runner.instance_)) return Runner.instance_;
      if (typeof Runner.getInstance === 'function' && ok(Runner.getInstance())) return Runner.getInstance();
      for (const k of Object.getOwnPropertyNames(Runner || {})) {
        try { if (ok(Runner[k])) return Runner[k]; } catch (e) { }
      }
    } catch (e) { }
    for (const k of Object.getOwnPropertyNames(window)) {
      try { if (ok(window[k])) return window[k]; } catch (e) { }
    }
    return null;
  }

  const runner = findRunner();
  if (!runner) {
    console.error('[dinoMCU] Chrome Dino game instance not found.');
    return;
  }

  const KEY = { JUMP: 38, DUCK: 40, START: 32 };
  const CODE = { 38: 'ArrowUp', 40: 'ArrowDown', 32: 'Space' };
  function fire(type, keyCode) {
    const e = new KeyboardEvent(type, {
      keyCode, which: keyCode, code: CODE[keyCode], key: keyCode === 32 ? ' ' : CODE[keyCode], bubbles: true
    });
    document.dispatchEvent(e);
  }
  const down = (k) => fire('keydown', k);
  const up = (k) => fire('keyup', k);

  function classify(o) {
    const cfg = o.typeConfig || {};
    const t = String(cfg.type || '').toLowerCase();
    if (t.includes('small')) return 'cactusSmall';
    if (t.includes('large')) return 'cactusLarge';
    if (t.includes('ptero') || t.includes('bird')) return 'pterodactyl';
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

    if (!obstacles.length) {
      return [999, speed, 0, 0, 0, 0, 0, 0, ducking];
    }
    const o = obstacles[0];
    const type = classify(o);
    const dist = Math.max(0, o.xPos - dinoX);
    const height = (o.typeConfig && o.typeConfig.height) || o.height || 0;
    const width = o.width || 0;
    const yPos = o.yPos || 0;

    return [
      dist,
      speed,
      type === 'cactusSmall' ? 1 : 0,
      type === 'cactusLarge' ? 1 : 0,
      type === 'pterodactyl' ? 1 : 0,
      yPos,
      height,
      width,
      ducking
    ];
  }

  let port = null;
  let reader = null;
  let writer = null;
  let isRunning = false;
  let jumpHeld = false;
  let duckHeld = false;

  function act(action) {
    const t = runner.tRex;
    if (jumpHeld && !t.jumping) { up(KEY.JUMP); jumpHeld = false; }
    if (action === 2) {
      if (jumpHeld) { up(KEY.JUMP); jumpHeld = false; }
      if (!duckHeld) { down(KEY.DUCK); duckHeld = true; }
    } else if (action === 1) {
      if (duckHeld) { up(KEY.DUCK); duckHeld = false; }
      if (!t.jumping && !jumpHeld) { down(KEY.JUMP); jumpHeld = true; }
    } else {
      if (duckHeld) { up(KEY.DUCK); duckHeld = false; }
      if (jumpHeld && !t.jumping) { up(KEY.JUMP); jumpHeld = false; }
    }
  }

  async function connectSerial() {
    if (!('serial' in navigator)) {
      alert('Web Serial is not supported in this browser. Please use Google Chrome or Edge.');
      return;
    }
    try {
      port = await navigator.serial.requestPort();
      await port.open({ baudRate: 115200 });

      const textEncoder = new TextEncoderStream();
      textEncoder.readable.pipeTo(port.writable);
      writer = textEncoder.writable.getWriter();

      const textDecoder = new TextDecoderStream();
      port.readable.pipeTo(textDecoder.writable);
      reader = textDecoder.readable.getReader();

      isRunning = true;
      btnConnect.style.background = '#059669';
      btnConnect.textContent = '⚡ Connected to NodeMCU';
      console.log('[dinoMCU] Connected to Serial Port!');

      startLoops();
    } catch (err) {
      console.error('[dinoMCU] Connection failed:', err);
      alert('Could not connect to Serial: ' + err.message);
    }
  }

  async function startLoops() {
    // Reading loop from NodeMCU
    (async () => {
      let buffer = '';
      while (isRunning && reader) {
        try {
          const { value, done } = await reader.read();
          if (done) break;
          if (value) {
            buffer += value;
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep remainder
            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed === '0' || trimmed === '1' || trimmed === '2') {
                const action = parseInt(trimmed, 10);
                act(action);
                updateBadge(action);
              }
            }
          }
        } catch (e) {
          console.warn('[dinoMCU] Read error:', e);
          break;
        }
      }
    })();

    // Sending loop to NodeMCU (~60 FPS)
    const interval = setInterval(async () => {
      if (!isRunning || !writer) {
        clearInterval(interval);
        return;
      }

      if (runner.crashed) {
        act(0);
        setTimeout(() => {
          try { runner.restart(); } catch (e) { down(KEY.START); up(KEY.START); }
        }, 1200);
        return;
      }

      if (!runner.playing) {
        down(KEY.START);
        up(KEY.START);
        return;
      }

      const st = readState();
      const msg = st.map(v => typeof v === 'number' ? v.toFixed(2) : v).join(',') + '\n';
      try {
        await writer.write(msg);
      } catch (e) { }
    }, 16);
  }

  function updateBadge(action) {
    const badge = document.getElementById('mcu-status-badge');
    if (!badge) return;
    const names = ['RUNNING', 'JUMPING', 'DUCKING'];
    badge.textContent = names[action] || 'RUNNING';
    badge.style.background = action === 1 ? '#f59e0b' : action === 2 ? '#8b5cf6' : '#10b981';
  }

  // UI Setup (Positioned safely at top: 75px so it NEVER covers Dino Score)
  let hud = document.getElementById('dino-mcu-hud');
  if (hud) hud.remove();

  hud = document.createElement('div');
  hud.id = 'dino-mcu-hud';
  Object.assign(hud.style, {
    position: 'fixed',
    top: '75px',
    right: '24px',
    left: 'auto',
    zIndex: '999999',
    background: 'linear-gradient(145deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.94))',
    backdropFilter: 'blur(16px)',
    color: '#f8fafc',
    padding: '18px 22px',
    borderRadius: '16px',
    boxShadow: '0 20px 40px -8px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(56, 189, 248, 0.25), 0 0 24px rgba(56, 189, 248, 0.15)',
    fontFamily: "'Segoe UI', Roboto, system-ui, -apple-system, sans-serif",
    fontSize: '13px',
    lineHeight: '1.5',
    width: '300px',
    userSelect: 'none',
    transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
  });

  hud.innerHTML = `
    <!-- Header & Drag Handle -->
    <div id="mcu-drag-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.12); padding-bottom:10px; cursor:move;">
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="font-size:18px;">⚡</span>
        <div>
          <strong style="font-size:14px; font-weight:700; background:linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; display:block;">
            Dino AI • NodeMCU
          </strong>
          <span style="font-size:10px; color:#94a3b8;">ESP8266 Hardware Brain</span>
        </div>
      </div>
      <span id="mcu-status-badge" style="font-size:10px; font-weight:700; padding:3px 8px; border-radius:20px; background:rgba(239, 68, 68, 0.2); color:#f87171; border:1px solid rgba(248, 113, 113, 0.3);">
        Disconnected
      </span>
    </div>

    <!-- Action Banner -->
    <div style="background:rgba(30, 41, 59, 0.8); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:8px 12px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
      <span style="font-size:11px; color:#94a3b8; font-weight:600; text-transform:uppercase;">Current Action</span>
      <span id="mcu-act-badge" style="font-size:12px; font-weight:800; padding:4px 12px; border-radius:8px; background:#10b981; color:#fff; box-shadow:0 2px 10px rgba(16,185,129,0.3);">
        RUNNING
      </span>
    </div>

    <button id="dino-btn-mcu-connect" style="background:linear-gradient(135deg, #2563eb, #1d4ed8); color:white; border:none; padding:9px 14px; border-radius:8px; font-weight:700; cursor:pointer; width:100%; box-shadow:0 4px 14px rgba(37,99,235,0.3); font-size:13px; margin-bottom:10px;">
      🔌 Connect NodeMCU (USB)
    </button>

    <div style="font-size:10px; color:#475569; display:flex; justify-content:space-between; align-items:center; border-top:1px solid rgba(255,255,255,0.08); padding-top:6px;">
      <span>Port: COM3 @ 115200</span>
      <span>⠿ drag to move</span>
    </div>
  `;
  document.body.appendChild(hud);

  // Draggable HUD functionality
  const dragHeader = document.getElementById('mcu-drag-header');
  let isDragging = false, startX, startY, initialLeft, initialTop;

  dragHeader.addEventListener('mousedown', (e) => {
    isDragging = true;
    const rect = hud.getBoundingClientRect();
    startX = e.clientX;
    startY = e.clientY;
    initialLeft = rect.left;
    initialTop = rect.top;
    hud.style.right = 'auto';
    hud.style.left = initialLeft + 'px';
    hud.style.top = initialTop + 'px';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    hud.style.left = Math.max(10, Math.min(window.innerWidth - hud.offsetWidth - 10, initialLeft + dx)) + 'px';
    hud.style.top = Math.max(10, Math.min(window.innerHeight - hud.offsetHeight - 10, initialTop + dy)) + 'px';
  });

  document.addEventListener('mouseup', () => { isDragging = false; });

  const btnConnect = document.getElementById('dino-btn-mcu-connect');
  btnConnect.onclick = () => connectSerial();

  window.dinoMCU = {
    stop() {
      isRunning = false;
      if (reader) reader.cancel();
      if (writer) writer.close();
      if (port) port.close();
      if (hud) hud.remove();
      console.log('[dinoMCU] Stopped');
    }
  };
})();
