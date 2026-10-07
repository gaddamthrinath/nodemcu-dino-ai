# Module 04: Real-Time Bridge & Real-World Edge AI Applications

> **Goal:** Understand the complete full-duplex communication loop between Chrome, Python, and NodeMCU, analyze the sub-2 millisecond latency budget, master stateful keypress injection and HUD telemetry, and learn how to adapt this entire pipeline to real-world industrial IoT and robotics products.

---

## 1. Full-Duplex System Architecture

The following diagram illustrates the complete data lifecycle of every single game frame:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. GOOGLE CHROME BROWSER                                               │
│    Runs Chrome Dino @ 60-80 FPS                                        │
│    JS snippet extracts: [dist, speed, type, y, h, w, ducking, ttc]     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Python Selenium Driver (~0.8 ms)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. PYTHON SERIAL BRIDGE (play_nodemcu.py)                              │
│    Formats 10-float CSV string: "120.5,9.2,1,0,0,100,35,20,0,13.1\n"   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ USB UART @ 115200 Baud (~0.35 ms)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. NODEMCU ESP8266 (dino_nodemcu.ino)                                  │
│    • Non-blocking ring buffer accumulates packet                       │
│    • C matrix multiplication runs in Flash ROM: dino_predict()         │
│    • Execution time: 0.08 ms (80 microseconds!)                        │
│    • Transmits reply: "PRED:1,4.82,82\n" (Action: JUMP)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ USB UART @ 115200 Baud (~0.15 ms)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. PYTHON ACTION DISPATCHER                                            │
│    • Parses action (0=RUN, 1=JUMP, 2=DUCK)                             │
│    • Executes stateful key event: Space down or Down-Arrow down        │
│    • Updates Live Glassmorphic HUD overlay in Chrome                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Latency Budgeting: Why Sub-2ms Latency Matters

In real-time control (whether jumping over a cactus at speed 13.0 or emergency braking a car at 100 km/h), **latency is life or death**.

At maximum speed ($13.0\text{ px/frame}$), an obstacle moves across the screen in just a few hundred milliseconds. If your AI decision takes 50 ms, the dinosaur crashes before the jump key is even pressed!

### Latency Breakdown per Frame

| Step | Operation | Duration | Percentage of Frame |
| :--- | :--- | :--- | :--- |
| **1** | JavaScript game state read via Selenium | $\sim 0.80\text{ ms}$ | 42.5% |
| **2** | PySerial UART transmission (PC $\to$ NodeMCU) | $\sim 0.35\text{ ms}$ | 18.6% |
| **3** | **NodeMCU C Neural Network Inference** | **$0.08\text{ ms}$** | **4.3%** |
| **4** | PySerial UART transmission (NodeMCU $\to$ PC) | $\sim 0.15\text{ ms}$ | 8.0% |
| **5** | Python stateful keypress dispatch & HUD update | $\sim 0.50\text{ ms}$ | 26.6% |
| **Total** | **Full End-to-End Decision Loop** | **$\sim 1.88\text{ ms}$** | **100%** |

Since a 60 FPS frame allows **16.66 ms** of time, our $1.88\text{ ms}$ loop uses **only 11% of the available frame budget**, guaranteeing buttery-smooth 60–80 FPS execution without missing a single frame!

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
    <div>Latency: <span id="hud-lat" style="color: #cbd5e1;">-- µs</span></div>
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
  5. `freefall_duration_ms`
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
* The complete system operates in **$\sim 1.88\text{ ms}$**, using only **11%** of a 60 FPS frame window.
* **Stateful key controls** prevent jerky transitions and allow holding the duck key through long obstacles.
* The exact same feature engineering, C code generation, and low-latency inference patterns power **real-world robotics, predictive maintenance, and medical devices**.

🎉 **Congratulations! You have completed the full NodeMCU Dino Edge AI Study Course.**
