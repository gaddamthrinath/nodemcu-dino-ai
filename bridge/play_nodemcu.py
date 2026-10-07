"""
NodeMCU Dino AI - High-Speed Bridge & Live Right-Side HUD
---------------------------------------------------------
1. Flashes live game state from Chrome over USB Serial to NodeMCU.
2. Displays real-time telemetry HUD on the top-right of the screen.
3. NodeMCU makes decisions on-chip and controls Dino at 60 FPS.
"""

import sys
import time
import serial
import serial.tools.list_ports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

BAUD_RATE = 115200

def find_nodemcu_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("❌ No serial devices found! Please connect your NodeMCU via USB.")
        sys.exit(1)
    
    # Check for likely USB UART bridge (CP210x / CH340 / USB Serial)
    usb_ports = [p for p in ports if any(k in p.description.lower() for k in ["cp210", "ch340", "uart", "usb serial", "ftdi", "silicon labs"])]
    if len(usb_ports) == 1:
        chosen = usb_ports[0].device
        print(f"👉 Auto-detected NodeMCU on: {chosen} ({usb_ports[0].description})")
        return chosen

    print("\n🔍 Available Ports:")
    for i, p in enumerate(ports):
        print(f"  [{i+1}] {p.device} - {p.description}")
    
    choice = input("\nEnter port number (press Enter for [1]): ").strip()
    if not choice:
        idx = 0
    else:
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(ports):
                idx = 0
        except ValueError:
            idx = 0
            
    chosen = ports[idx].device
    print(f"👉 Selected: {chosen}")
    return chosen

def main():
    port_name = find_nodemcu_port()
    print(f"Connecting to NodeMCU on {port_name} at {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=0.008, write_timeout=0.008)
        time.sleep(1.5) # Allow NodeMCU to reset
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print("✅ NodeMCU Connected with ultra-low latency serial buffers!")
    except Exception as e:
        print(f"❌ Could not open port {port_name}: {e}")
        sys.exit(1)

    print("\n🌐 Launching Chrome Dino in Full Screen...")
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-infobars")
    options.add_argument("--mute-audio")
    options.add_argument("--start-maximized")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.maximize_window()
    try:
        driver.get("chrome://dino")
    except Exception:
        pass # Expected for offline dino page
    
    time.sleep(1)

    # Inject JavaScript state reader, key simulator, and RIGHT-SIDE HUD
    inject_script = """
    // 1. Find Game Runner
    function findRunner() {
        const isOk = (v) => v && typeof v === 'object' && v.tRex && v.horizon;
        try {
            if (isOk(window.Runner && Runner.instance_)) return Runner.instance_;
            if (typeof Runner.getInstance === 'function' && isOk(Runner.getInstance())) return Runner.getInstance();
            for (const k of Object.getOwnPropertyNames(Runner || {})) {
                try { if (isOk(Runner[k])) return Runner[k]; } catch (e) {}
            }
        } catch (e) {}
        for (const k of Object.getOwnPropertyNames(window)) {
            try { if (isOk(window[k])) return window[k]; } catch (e) {}
        }
        return null;
    }

    // 2. Read game state
    window.dinoState = function() {
        const r = findRunner();
        if (!r) return null;
        const dino = r.tRex;
        const dinoX = dino.xPos;
        const obstacles = (r.horizon.obstacles || [])
            .filter(o => o.xPos + o.width > dinoX)
            .sort((a, b) => a.xPos - b.xPos);
        
        const speed = r.currentSpeed || 6;
        const ducking = dino.ducking ? 1 : 0;
        
        let score = 0;
        try {
            score = r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan));
        } catch (e) {
            score = Math.round((r.distanceRan || 0) * 0.025);
        }

        if (!obstacles.length) {
            return [999, speed, 0, 0, 0, 0, 0, 0, ducking, r.crashed ? 1 : 0, r.playing ? 1 : 0, score, 'none'];
        }
        
        const o = obstacles[0];
        const cfg = o.typeConfig || {};
        const t = String(cfg.type || '').toLowerCase();
        let small = 0, large = 0, bird = 0, rawType = 'cactusSmall';
        if (t.includes('small')) { small = 1; rawType = 'cactusSmall'; }
        else if (t.includes('large')) { large = 1; rawType = 'cactusLarge'; }
        else if (t.includes('ptero') || t.includes('bird')) { bird = 1; rawType = 'pterodactyl'; }
        else if ((cfg.height || 0) >= 45) { large = 1; rawType = 'cactusLarge'; }
        else { small = 1; rawType = 'cactusSmall'; }
        
        const dist = Math.max(0, o.xPos - dinoX);
        const y = o.yPos || 0;
        const h = (cfg.height || o.height || 0);
        const w = o.width || 0;
        
        return [dist, speed, small, large, bird, y, h, w, ducking, r.crashed ? 1 : 0, r.playing ? 1 : 0, score, rawType];
    };

    // 3. Key Actuation with state tracking
    let jumpHeld = false;
    let duckHeld = false;

    window.dinoAct = function(action, latencyMs) {
        const r = findRunner();
        const t = r ? r.tRex : null;
        function fire(type, code) {
            const CODE = { 38: 'ArrowUp', 40: 'ArrowDown', 32: 'Space' };
            document.dispatchEvent(new KeyboardEvent(type, { keyCode: code, which: code, code: CODE[code], bubbles: true }));
        }

        if (jumpHeld && t && !t.jumping) {
            fire('keyup', 38);
            jumpHeld = false;
        }

        if (action === 2) { // DUCK
            if (jumpHeld) { fire('keyup', 38); jumpHeld = false; }
            if (!duckHeld) { fire('keydown', 40); duckHeld = true; }
        } else if (action === 1) { // JUMP
            if (duckHeld) { fire('keyup', 40); duckHeld = false; }
            if (t && !t.jumping && !jumpHeld) {
                fire('keydown', 38);
                jumpHeld = true;
            }
        } else { // RUN
            if (duckHeld) { fire('keyup', 40); duckHeld = false; }
            if (jumpHeld && t && !t.jumping) { fire('keyup', 38); jumpHeld = false; }
        }

        updateHUD(action, latencyMs);
    };

    window.dinoRestart = function() {
        if (jumpHeld) { document.dispatchEvent(new KeyboardEvent('keyup', { keyCode: 38, bubbles: true })); jumpHeld = false; }
        if (duckHeld) { document.dispatchEvent(new KeyboardEvent('keyup', { keyCode: 40, bubbles: true })); duckHeld = false; }
        
        const r = findRunner();
        if (r && r.crashed) {
            try { r.restart(); } catch (e) {}
        }
        document.dispatchEvent(new KeyboardEvent('keydown', { keyCode: 32, which: 32, code: 'Space', bubbles: true }));
        setTimeout(() => document.dispatchEvent(new KeyboardEvent('keyup', { keyCode: 32, which: 32, code: 'Space', bubbles: true })), 80);
        updateHUD(-1, 0);
    };

    // 4. Inject Right-Side HUD UI (Positioned safely at top: 75px so it NEVER covers Dino Score)
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
            <span id="mcu-conn-status" style="font-size:10px; font-weight:700; padding:3px 8px; border-radius:20px; background:rgba(16, 185, 129, 0.2); color:#34d399; border:1px solid rgba(52, 211, 153, 0.3); display:flex; align-items:center; gap:4px;">
                <span style="width:6px; height:6px; border-radius:50%; background:#10b981; display:inline-block; box-shadow:0 0 6px #10b981;"></span>
                ONLINE
            </span>
        </div>

        <!-- Action Banner -->
        <div id="mcu-action-banner" style="background:rgba(30, 41, 59, 0.8); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:8px 12px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:11px; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Current Action</span>
            <span id="mcu-badge" style="font-size:12px; font-weight:800; padding:4px 12px; border-radius:8px; background:#10b981; color:#fff; letter-spacing:0.5px; box-shadow:0 2px 10px rgba(16,185,129,0.3);">
                RUNNING
            </span>
        </div>

        <!-- Metrics Grid -->
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:12px;">
            <!-- Score Card -->
            <div style="background:rgba(15, 23, 42, 0.6); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:8px 10px;">
                <div style="font-size:10px; color:#94a3b8; text-transform:uppercase; margin-bottom:2px;">🏆 Game Score</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span id="mcu-score" style="font-size:17px; font-weight:800; color:#60a5fa; font-family:monospace;">0</span>
                    <span style="font-size:10px; color:#fbbf24;">HI <b id="mcu-best" style="font-family:monospace;">0</b></span>
                </div>
            </div>

            <!-- Latency Card -->
            <div style="background:rgba(15, 23, 42, 0.6); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:8px 10px;">
                <div style="font-size:10px; color:#94a3b8; text-transform:uppercase; margin-bottom:2px;">⚡ MCU Latency</div>
                <div style="font-size:17px; font-weight:800; color:#f472b6; font-family:monospace;">
                    <span id="mcu-latency">0.0</span> <span style="font-size:10px; color:#94a3b8; font-weight:normal;">ms</span>
                </div>
            </div>

            <!-- Obstacle Distance -->
            <div style="background:rgba(15, 23, 42, 0.6); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:8px 10px;">
                <div style="font-size:10px; color:#94a3b8; text-transform:uppercase; margin-bottom:2px;">🎯 Obstacle</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span id="mcu-dist" style="font-size:14px; font-weight:700; color:#e2e8f0; font-family:monospace;">None</span>
                    <span id="mcu-type-icon" style="font-size:12px;">🌵</span>
                </div>
            </div>

            <!-- Speed & TTC -->
            <div style="background:rgba(15, 23, 42, 0.6); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:8px 10px;">
                <div style="font-size:10px; color:#94a3b8; text-transform:uppercase; margin-bottom:2px;">🚀 Speed / TTC</div>
                <div style="font-size:13px; font-weight:700; color:#e2e8f0; font-family:monospace;">
                    <span id="mcu-speed">6.0</span> <span style="font-size:10px; color:#64748b;">spd</span> • <span id="mcu-ttc" style="color:#34d399;">∞</span>
                </div>
            </div>
        </div>

        <!-- Footer specs & Drag Hint -->
        <div style="font-size:11px; color:#64748b; display:flex; justify-content:space-between; align-items:center; border-top:1px solid rgba(255,255,255,0.08); padding-top:8px;">
            <span>Port: <b style="color:#94a3b8;">COM3</b> (115.2k)</span>
            <span style="font-size:10px; color:#475569; display:flex; align-items:center; gap:3px;">
                ⠿ drag to move
            </span>
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

    let bestScore = 0;
    window.updateHUD = function(action, latencyMs) {
        const state = window.dinoState();
        if (!state) return;

        const score = state[11] || 0;
        if (score > bestScore) bestScore = score;

        const dist = Math.round(state[0]);
        const speed = state[1].toFixed(1);
        const ttc = state[0] < 900 ? (state[0] / Math.max(state[1], 0.1)).toFixed(1) : '∞';
        const rawType = state[12] || 'cactusSmall';

        const bScore = document.getElementById('mcu-score');
        const bBest = document.getElementById('mcu-best');
        const bDist = document.getElementById('mcu-dist');
        const bSpeed = document.getElementById('mcu-speed');
        const bTtc = document.getElementById('mcu-ttc');
        const bLat = document.getElementById('mcu-latency');
        const badge = document.getElementById('mcu-badge');
        const typeIcon = document.getElementById('mcu-type-icon');

        if (bScore) bScore.textContent = score;
        if (bBest) bBest.textContent = bestScore;
        if (bDist) bDist.textContent = dist < 900 ? dist + 'px' : 'Clear';
        if (bSpeed) bSpeed.textContent = speed;
        if (bTtc) bTtc.textContent = ttc;
        if (bLat && latencyMs !== undefined) bLat.textContent = latencyMs.toFixed(1);

        if (typeIcon) {
            if (dist >= 900) typeIcon.textContent = '✨';
            else if (rawType === 'pterodactyl') typeIcon.textContent = '🦅 Bird';
            else if (rawType === 'cactusLarge') typeIcon.textContent = '🌵 Big';
            else typeIcon.textContent = '🌵 Small';
        }

        if (badge) {
            if (action === 1) {
                badge.textContent = '⬆️ JUMPING';
                badge.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
                badge.style.boxShadow = '0 2px 14px rgba(245, 158, 11, 0.4)';
                badge.style.color = '#fff';
            } else if (action === 2) {
                badge.textContent = '⬇️ DUCKING';
                badge.style.background = 'linear-gradient(135deg, #a855f7, #7c3aed)';
                badge.style.boxShadow = '0 2px 14px rgba(168, 85, 247, 0.4)';
                badge.style.color = '#fff';
            } else if (action === -1 || state[9]) {
                badge.textContent = '💥 CRASHED';
                badge.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
                badge.style.boxShadow = '0 2px 14px rgba(239, 68, 68, 0.4)';
                badge.style.color = '#fff';
            } else {
                badge.textContent = '🏃 RUNNING';
                badge.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                badge.style.boxShadow = '0 2px 14px rgba(16, 185, 129, 0.3)';
                badge.style.color = '#fff';
            }
        }
    }
    """
    driver.execute_script(inject_script)
    print("🎮 Ready! NodeMCU High-Speed Bridge Active. Telemetry HUD mounted on TOP-RIGHT of Chrome.")
    print("=" * 70)
    print(f"{'SCORE':<8} | {'ACTION':<10} | {'DIST':<8} | {'SPEED':<8} | {'TTC':<8} | {'LATENCY':<8}")
    print("=" * 70)

    last_log_time = 0
    scores = []
    last_score = 0
    was_crashed = False

    # Main ultra-low latency loop (~60 Hz)
    try:
        while True:
            t0 = time.perf_counter()
            state = driver.execute_script("return window.dinoState();")
            if not state:
                time.sleep(0.02)
                continue
            
            crashed = state[9]
            playing = state[10]
            score = state[11]

            if crashed:
                if not was_crashed:
                    was_crashed = True
                    scores.append(score)
                    best = max(scores)
                    avg = sum(scores) // len(scores)
                    print(f"\n💥 CRASH! Score: {score} | Best: {best} | Avg: {avg} | Runs: {len(scores)}")
                    try:
                        driver.execute_script("if (window.updateHUD) window.updateHUD(-1, 0);")
                    except Exception:
                        pass
                
                time.sleep(1.2)
                driver.execute_script("window.dinoRestart();")
                continue

            was_crashed = False

            if not playing:
                driver.execute_script("window.dinoRestart();")
                time.sleep(0.8)
                continue

            last_score = score
            dist = state[0]
            speed = state[1]
            ttc = dist / max(speed, 0.1)

            # Send 9 float features to NodeMCU: dist,speed,obs_small,obs_large,obs_bird,obs_y,obs_h,obs_w,ducking
            packet = ",".join(f"{v:.2f}" for v in state[:9]) + "\n"
            ser.write(packet.encode('ascii'))
            
            # Non-blocking read from NodeMCU
            resp = ser.readline().decode('ascii', errors='ignore').strip()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            action = 0
            if resp in ("0", "1", "2"):
                action = int(resp)
            
            # Execute action in Chrome & update right-side HUD
            driver.execute_script(f"window.dinoAct({action}, {latency_ms});")

            # Console logging
            action_names = ["RUN", "JUMP", "DUCK"]
            action_str = action_names[action]

            now = time.time()
            if action != 0 or now - last_log_time > 0.5:
                last_log_time = now
                dist_str = f"{dist:.0f}px" if dist < 900 else "None"
                ttc_str = f"{ttc:.1f}" if dist < 900 else "∞"
                print(f"{score:<8} | {action_str:<10} | {dist_str:<8} | {speed:<8.1f} | {ttc_str:<8} | {latency_ms:<6.1f}ms")

            time.sleep(0.012) # ~60-80 Hz refresh rate

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        ser.close()
        driver.quit()

if __name__ == "__main__":
    main()
