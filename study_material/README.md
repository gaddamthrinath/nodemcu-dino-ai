# Edge AI & Microcontroller Neural Networks: Comprehensive Study Course

Welcome to the comprehensive, step-by-step study curriculum for the **NodeMCU Dino Edge AI** project.

This course teaches you how to design, train, export, and run deep learning neural networks directly on resource-constrained microcontrollers (ESP8266 / ESP32 / ARM Cortex-M) in pure C with sub-millisecond real-time performance.

---

## 📚 Curriculum Roadmap

The course is divided into 4 modular, self-contained study guides:

### [Module 01: Data & Feature Engineering](01_data_and_feature_engineering.md)
* Why compact feature engineering beats raw pixel computer vision on microcontrollers.
* Anatomy of the Chrome Dino internal state (`Runner.instance_`).
* The 10 mathematical input features.
* The physics of Time-To-Collision ($\text{TTC} = \text{dist} / \text{speed}$).
* Data cleaning: Dropping intro frames, grouping by obstacle IDs to prevent data leakage, and discarding mid-air jump frames.
* **Real-World Parallels:** Autonomous drone collision avoidance, automotive ADAS / smart braking.

### [Module 02: Neural Network Math & PyTorch Training](02_neural_network_math_and_pytorch.md)
* Multi-Layer Perceptron (MLP) architecture ($10 \to 16 \to 16 \to 3$, 499 parameters).
* Step-by-step forward pass math: Linear layers, non-linear activations, and logit outputs.
* Why ReLU is the ideal activation function for microcontrollers (zero exponential math).
* Solving extreme class imbalance with inverse frequency loss weighting.
* Input normalization, standard deviation scaling, and clipping for embedded stability.
* PyTorch training pipeline and C code generation (`model.h`).
* **Real-World Parallels:** Industrial vibration anomaly detection, smart agricultural soil monitoring.

### [Module 03: Embedded C & Hardware Inference](03_embedded_c_and_hardware_inference.md)
* Microcontroller hardware architecture: Xtensa LX106 32-bit core, SRAM vs Flash ROM.
* Why raw weights in C `PROGMEM` outperform runtime interpreters (TensorFlow Lite Micro / ONNX).
* Memory layout: Zero dynamic heap allocation (`malloc`), zero RAM footprint for weights.
* Pure C forward pass walkthrough (`dino_predict`) taking $< 0.08\text{ ms}$ ($80\text{ µs}$).
* Zero-cost Argmax selection on raw logits without computing Softmax.
* Non-blocking high-speed serial packet accumulation and tokenization (`strtok`).
* **Real-World Parallels:** Hard real-time control (airbags, active suspension, industrial pneumatic sorting).

### [Module 04: Real-Time Bridge & Real-World Edge AI Applications](04_realtime_bridge_and_realworld_applications.md)
* Full-duplex system architecture: Chrome JS $\leftrightarrow$ Python Bridge $\leftrightarrow$ NodeMCU C.
* Latency budgeting: Tracing a frame through the $< 1.88\text{ ms}$ roundtrip loop.
* Python Selenium automation and stateful keypress controller (duck hold vs jump tap).
* In-browser draggable glassmorphic HUD telemetry injection.
* Client-side JavaScript neural network runner (`browser_ai.js`).
* **4 Complete Industrial Blueprints:**
  1. Factory Motor Bearing Predictive Maintenance.
  2. Smart Solar-Powered Farm Irrigation.
  3. Wearable Fall & Cardiac Arrhythmia Detection.
  4. Autonomous Drone Obstacle Reflex System.
* Advanced horizons: Int8 fixed-point quantization, on-device Q-learning, and dual-core FreeRTOS.

---

## 🎯 Learning Outcomes

After completing this study course, you will be able to:
1. **Analyze any real-time system** (game, robot, factory sensor, vehicle) and distill high-dimensional data into compact numerical feature representations.
2. **Train PyTorch models** resilient to real-world class imbalances and noise.
3. **Write pure C inference engines** that execute matrix operations and activations without depending on heavy third-party libraries.
4. **Deploy AI models on $3 microcontrollers** with deterministic microsecond-level execution times.
5. **Architect full-stack real-time pipelines** connecting hardware peripherals, microcontrollers, PC bridges, and web applications.

---

## 🛠️ Practical Exercises & Experiments

Try these hands-on challenges to test your understanding:

1. **Feature Ablation Experiment:** In [`training/train.py`](../training/train.py), remove the `ttc` feature and train with only 9 features. Test the resulting model in the game. At what speed does the dinosaur begin to mistime its jumps?
2. **Quantization Challenge:** Convert the `float` arrays in [`models/model.h`](../models/model.h) to scaled 8-bit integers (`int8_t`). How much did the Flash ROM size decrease?
3. **Blinky Reflex Indicator:** In [`firmware/dino_nodemcu/dino_nodemcu.ino`](../firmware/dino_nodemcu/dino_nodemcu.ino), turn on the built-in onboard LED (`LED_BUILTIN`) whenever the AI outputs `JUMP` or `DUCK`, and turn it off during `RUN`. Measure the visual synchronization with the screen!
