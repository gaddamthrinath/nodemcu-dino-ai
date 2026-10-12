# Module 04: Real-Time Bridge & Real-World Edge AI Applications

> **Goal:** Understand the complete full-duplex communication loop between Chrome, Python, and NodeMCU, understand how the communication pipeline stays fast enough for real-time responsiveness, master stateful keypress injection and HUD telemetry, and learn how to adapt this entire pipeline to real-world industrial IoT and robotics products.

---

## 1. Full-Duplex System Architecture

The following diagram illustrates the complete data lifecycle of every single game frame:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. GOOGLE CHROME BROWSER                                               │
│    Runs Chrome Dino @ 60-80 FPS                                        │
│    JS snippet extracts: [dist, speed, type, y, h, w, ducking, ttc]     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Fast IPC via Selenium execute_script
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. PYTHON SERIAL BRIDGE (play_nodemcu.py)                              │
│    Formats 10-float CSV string: "120.5,9.2,1,0,0,100,35,20,0,13.1\n"   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ High-speed USB UART @ 115200 Baud
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. NODEMCU ESP8266 (dino_nodemcu.ino)                                  │
│    • Non-blocking ring buffer accumulates packet                       │
│    • Bare-metal C matrix multiplication in Flash ROM: dino_predict()   │
│    • Fast on-chip decision                                             │
│    • Transmits reply: "PRED:1,4.82,82\n" (Action: JUMP)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ High-speed USB UART @ 115200 Baud
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. PYTHON ACTION DISPATCHER                                            │
│    • Parses action (0=RUN, 1=JUMP, 2=DUCK)                             │
│    • Executes stateful key event: Space down or Down-Arrow down        │
│    • Updates Live Glassmorphic HUD overlay in Chrome                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Real-Time Communication & Responsiveness: Fast Enough for Edge Control

In real-time systems (whether jumping over a cactus at maximum game speed or triggering emergency braking in an autonomous vehicle), **responsiveness and communication speed are critical**.

At higher game speeds, obstacles approach rapidly. If communication between the browser, Python bridge, and microcontroller is slow or blocking, the dinosaur crashes before the jump key is ever registered. The entire architecture is designed so data flows **fast enough to keep pace with the game loop in real time**.

### What Makes the Communication Fast Enough?

Instead of relying on heavy network stacks or slow polling, every step of the pipeline is stripped down to maximize throughput and maintain high responsiveness:

| Pipeline Stage | Implementation | Why It Communicates Fast Enough |
| :--- | :--- | :--- |
| **1. Browser State Read** | Selenium `execute_script()` reading `window.dinoState()` | Accesses internal game variables directly from memory without the overhead of screen scraping or visual OCR. |
| **2. PC $\to$ NodeMCU Stream** | PySerial UART @ 115200 Baud with small buffer flush | Sends a compact ASCII CSV string containing only the essential numeric features. |
| **3. Hardware Inference** | Bare-metal C MLP compiled into ESP8266 Flash ROM | Pure arithmetic matrix math executed directly on the CPU with zero runtime interpreter or OS overhead. |
| **4. NodeMCU $\to$ PC Reply** | Non-blocking serial reply (`Serial.println(action)`) | Minimal single-token action payload returned immediately upon calculation. |
| **5. Action Dispatch** | Native browser event dispatch + DOM HUD update | Instant dispatch directly targeting the active browser window. |

### Practical Timing & Responsiveness Considerations

In real-world setups across various USB-to-UART bridge chips (CH340, CP2102), operating system schedulers, and driver buffers:

* **The Core Goal:** The objective is not chasing theoretical or unrealistic latency figures, but ensuring the complete roundtrip communicates **fast enough** to easily stay ahead of the game's reaction window.
* **Keeping Pace with 60 FPS:** The Chrome Dino game updates at approximately 60 frames per second. As long as state extraction, serial transmission, and action execution finish comfortably within each frame cycle, gameplay remains responsive with zero noticeable control lag.
* **Zero Queue Accumulation:** Non-blocking serial buffers and quick packet handling prevent stale backlogs, ensuring the AI is always reacting to the freshest game state.

---

## 3. Deep Dive: Python Bridge & Stateful Keypresses

The bridge script [`bridge/play_nodemcu.py`](../bridge/play_nodemcu.py) handles four critical responsibilities:

### 1. Launching Offline Chrome Dino via Selenium

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--disable-infobars")
options.add_argument("--mute-audio")
options.add_argument("--no-default-browser-check")

driver = webdriver.Chrome(options=options)

# Chrome Dino triggers an ERR_INTERNET_DISCONNECTED error page by design.
# We wrap driver.get() in a try/except so Selenium doesn't crash on load!
try:
    driver.get("chrome://dino")
except Exception:
    pass
```

---

### 2. Stateful Keypress Controller

A common beginner mistake in game bots is calling `press_and_release()` on every frame. 
* If the AI wants to **DUCK under a long pterodactyl**, repeatedly pressing and releasing the Down arrow causes the dinosaur to stand up mid-flight and crash!
* If the AI wants to **JUMP**, it needs to tap Space once, not spam 60 keydown events per second.

```python
class KeyController:
    def __init__(self, body_element):
        self.body = body_element
        self.is_ducking = False

    def execute_action(self, action):
        if action == 1:  # JUMP
            if self.is_ducking:
                self.body.send_keys(Keys.NULL) # Release duck
                self.is_ducking = False
            self.body.send_keys(Keys.SPACE)     # Trigger jump

        elif action == 2:  # DUCK
            if not self.is_ducking:
                self.body.send_keys(Keys.ARROW_DOWN) # Hold duck
                self.is_ducking = True

        elif action == 0:  # RUN
            if self.is_ducking:
                self.body.send_keys(Keys.NULL) # Release duck
                self.is_ducking = False
```

---

### 3. Glassmorphic In-Browser Live HUD

To monitor hardware telemetry in real time without obscuring gameplay, the bridge injects a draggable HTML/CSS HUD overlay directly into the browser's Document Object Model (DOM):

```javascript
// Injected into Chrome DOM via driver.execute_script()
const hud = document.createElement("div");
hud.id = "dino-ai-hud";
hud.style.cssText = `
    position: fixed; top: 16px; right: 16px; z-index: 999999;
    background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px;
    padding: 14px 18px; color: #f8fafc; font-family: monospace;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
`;
hud.innerHTML = `
    <div style="font-weight: bold; color: #38bdf8;">NodeMCU Edge AI Telemetry</div>
    <div>FPS: <span id="hud-fps" style="color: #4ade80;">--</span></div>
    <div>Action: <span id="hud-action" style="color: #facc15;">RUN</span></div>
    <div>Response: <span id="hud-resp" style="color: #cbd5e1;">Fast / Live</span></div>
`;
document.body.appendChild(hud);
```

---

## 4. Browser Direct AI: Zero-Hardware Fallback

In [`browser/browser_ai.js`](../browser/browser_ai.js), we implement the exact same neural network architecture in pure JavaScript using [`models/model.json`](../models/model.json).

```javascript
function predictJS(inputs, model) {
    // 1. Normalize
    const norm = inputs.map((val, i) => {
        let n = (val - model.means[i]) / (model.stds[i] || 1.0);
        return Math.max(-5.0, Math.min(5.0, n));
    });

    // 2. Hidden Layer 1 (Weights + Bias + ReLU)
    const h1 = model.b1.map((bias, j) => {
        let sum = bias;
        for (let i = 0; i < 10; i++) sum += model.w1[j][i] * norm[i];
        return Math.max(0, sum);
    });

    // 3. Hidden Layer 2 (Weights + Bias + ReLU)
    const h2 = model.b2.map((bias, j) => {
        let sum = bias;
        for (let i = 0; i < 16; i++) sum += model.w2[j][i] * h1[i];
        return Math.max(0, sum);
    });

    // 4. Output Layer & Argmax
    let maxVal = -Infinity, bestAction = 0;
    model.b3.forEach((bias, j) => {
        let sum = bias;
        for (let i = 0; i < 16; i++) sum += model.w3[j][i] * h2[i];
        if (sum > maxVal) { maxVal = sum; bestAction = j; }
    });

    return bestAction;
}
```

This allows full verification and training validation on any browser without needing hardware connected.

---

## 5. Applying Your Edge AI Knowledge to Real-World Problems

You now understand the complete pipeline:
1. **Feature Engineering & Normalization**
2. **PyTorch Training with Class Weighting**
3. **C Code Generation (`PROGMEM`)**
4. **Non-blocking Embedded Inference**
5. **Real-Time Actuation & Feedback**

Here are 4 industry blueprints you can build today using this exact blueprint:

---

### Blueprint 1: Industrial Predictive Maintenance (Factory Motors)

```
[Vibration Sensor + Current Clamp] ──> [ESP32 / STM32] ──> [Flash ROM C MLP] ──> [Relay / Alarm]
```

* **Problem:** Factory motor bearings wear out over time. If a bearing seizes unexpectedly, an entire manufacturing line halts, costing thousands of dollars per hour.
* **Sensor Inputs (8 Features):**
  1. `vibration_rms` (from ADXL345 accelerometer)
  2. `vibration_peak_freq` (from FFT)
  3. `vibration_kurtosis`
  4. `motor_current_amps` (from CT current clamp)
  5. `temperature_celsius` (from DS18B20 temp sensor)
  6. `rpm_actual` (from Hall effect sensor)
  7. `run_hours_continuous`
  8. `ambient_temp_celsius`
* **Microcontroller Actions:**
  * `0`: **Normal** (Keep running)
  * `1`: **Warning / Maintenance Due** (Send MQTT telemetry over WiFi)
  * `2`: **Critical Failure Imminent** (Open contactor relay to safely shut down motor)

---

### Blueprint 2: Smart Agriculture & Precision Irrigation

```
[Soil + Sunlight Sensors] ──> [Solar-Powered NodeMCU] ──> [C Model] ──> [Solenoid Valve Relay]
```

* **Problem:** Traditional farm timers water crops on fixed schedules, wasting water during rain or under-watering during heatwaves.
* **Sensor Inputs (6 Features):**
  1. `soil_moisture_pct` (Capacitive soil sensor)
  2. `air_temperature` (DHT22)
  3. `air_humidity` (DHT22)
  4. `solar_irradiance_lux` (BH1750)
  5. `soil_temperature`
  6. `forecast_rain_prob` (Pulled once every morning via WiFi)
* **Microcontroller Actions:**
  * `0`: **Do Nothing** (Soil moisture adequate)
  * `1`: **Open Drip Irrigation Valve** (Water plants for optimal absorption)

---

### Blueprint 3: Wearable Fall Detection in Elderly Care

```
[6-Axis IMU (MPU6050)] ──> [Low-Power Wearable MCU] ──> [C Model] ──> [Cellular SOS Alert]
```

* **Problem:** Falls among elderly individuals can be life-threatening if they are unable to reach a phone.
* **Sensor Inputs (7 Features):**
  1. `accel_magnitude = sqrt(ax^2 + ay^2 + az^2)`
  2. `gyro_magnitude = sqrt(gx^2 + gy^2 + gz^2)`
  3. `jerk_metric = d(accel)/dt`
  4. `vertical_orientation_angle`
  5. `freefall_duration`
  6. `post_impact_immobility_seconds`
  7. `heart_rate_bpm`
* **Microcontroller Actions:**
  * `0`: **Normal Activity** (Walking, sitting, lying down)
  * `1`: **Fall Detected** (Sound local buzzer, send emergency SMS via SIM800L module)

---

### Blueprint 4: Drone Collision Reflex Loop

```
[4x ToF Laser Rangefinders] ──> [Flight Controller MCU] ──> [C Model] ──> [Motor ESCs]
```

* **Problem:** Drones flying indoors or near trees cannot rely on slow GPS/cloud navigation for instantaneous obstacle avoidance.
* **Sensor Inputs (10 Features):**
  * `range_front`, `range_back`, `range_left`, `range_right` (VL53L1X Time-of-Flight sensors)
  * `vx`, `vy`, `vz` (Optical flow velocities)
  * `ttc_front`, `ttc_left`, `ttc_right` (Derived Time-to-Collision)
* **Microcontroller Actions:**
  * `0`: **Continue Nav Waypoint**
  * `1`: **Emergency Reverse Thrust**
  * `2`: **Emergency Left Bank**
  * `3`: **Emergency Right Bank**
  * `4`: **Emergency Climb**

---

## 6. Advanced Next Steps for Edge AI Engineers

When you are ready to expand your embedded AI skills beyond this project:

1. **8-bit Integer Quantization (Int8):** Convert 32-bit floats to 8-bit integers (`int8_t`) with scale factors, reducing Flash storage by 4x and speeding up 8-bit microcontrollers (like Arduino Uno / ATmega328P).
2. **On-Device Reinforcement Learning (Q-Learning):** Implement tabular Q-learning or Policy Gradient updates directly in C on the microcontroller so it learns to play games or navigate mazes without a PC training phase.
3. **Dual-Core FreeRTOS on ESP32:** Run your serial communication and sensor reading on Core 0, and run the neural network inference uninterrupted on Core 1 for zero-jitter deterministic timing.

---

### Summary of Module 4
* The full-duplex loop communicates **fast enough** to process states and trigger key actions in real time without falling behind the game pace.
* **Stateful key controls** prevent jerky transitions and allow holding the duck key through long obstacles.
* The exact same feature engineering, C code generation, and responsive communication patterns power **real-world robotics, predictive maintenance, and medical devices**.

🎉 **Congratulations! You have completed the full NodeMCU Dino Edge AI Study Course.**
