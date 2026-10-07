# Module 01: Data & Feature Engineering

> **Goal:** Understand how the AI "perceives" the game environment, why feature engineering beats raw pixel computer vision for microcontrollers, how Time-To-Collision (TTC) physics works, and how to apply this to real-world edge robotics.

---

## 1. The Core Philosophy: Raw Pixels vs. Compact Features

When beginners approach AI for games or robotics, the first thought is usually:
> *"Let's take a screenshot / camera feed of the screen and feed it into a Convolutional Neural Network (CNN)!"*

### Why Raw Pixels Fail on Microcontrollers
If you take a tiny grayscale screenshot of $100 \times 100$ pixels:
* **Input size:** $10,000$ numbers per frame.
* **Model size:** A standard CNN needs **500,000 to 5,000,000 parameters** (~5 MB to 50 MB of memory).
* **Processing time:** Running a CNN on a small microcontroller takes **500 ms to 2,000 ms** per frame ($0.5$ to $2$ frames per second).
* **Result:** The dinosaur crashes before the computer even finishes processing the first image!

### The Edge AI Approach: Feature Extraction
Instead of passing millions of blank white background pixels, we extract the **exact mathematical state** the dinosaur needs to make a decision:
* **Input size:** Just **10 numbers** per frame.
* **Model size:** **499 parameters** (~1.9 KB of Flash memory).
* **Processing time:** **< 0.1 ms** (10,000+ predictions per second on an ESP8266!).

```
[Raw Pixel Approach - Heavy & Slow]
Canvas (10,000 Pixels) ──> Heavy CNN (20 MB) ──> Takes 500 ms ──> CRASH!

[Edge AI Feature Approach - Ultra-Fast]
Game State ──> 10 Float Numbers ──> Tiny MLP (1.9 KB) ──> Takes 0.08 ms ──> SUCCESS!
```

---

## 2. Anatomy of the Chrome Dino Game State

Inside Google Chrome, the Dino game is managed by an internal JavaScript object called `Runner.instance_`.

### Key Game Objects:
1. `Runner.instance_.tRex`: Represents the dinosaur.
   * `xPos`: Horizontal position on canvas (typically `21px`).
   * `yPos`: Vertical position (elevates when jumping).
   * `jumping`: Boolean (`true` when in air).
   * `ducking`: Boolean (`true` when duck key is held).

2. `Runner.instance_.horizon.obstacles`: An array of upcoming obstacles.
   * `xPos`: Distance from left side of screen.
   * `yPos`: Elevation of obstacle (cacti are at ground level `~100px`, birds fly at `50px`, `75px`, or `100px`).
   * `width` & `height`: Size of the obstacle.
   * `typeConfig.type`: `"CACTUS_SMALL"`, `"CACTUS_LARGE"`, or `"PTERODACTYL"`.

3. `Runner.instance_.currentSpeed`: The game scroll velocity (starts at `6.0` and ramps up to `13.0` maximum).

---

## 3. The 10 Features: What Each Input Means

We transform raw game coordinates into a structured **10-dimensional input vector**:

```
Input Vector = [dist, speed, obs_small, obs_large, obs_bird, obs_y, obs_h, obs_w, ducking, ttc]
```

| Feature | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `dist` | Float | `0 to 600` | Distance from dinosaur's nose to obstacle: `obstacle.x - dino.x` |
| `speed` | Float | `6.0 to 13.0` | Game acceleration velocity (pixels per frame) |
| `obs_small` | Binary (0/1) | `0 or 1` | 1 if obstacle is a small cactus, else 0 |
| `obs_large` | Binary (0/1) | `0 or 1` | 1 if obstacle is a large cactus, else 0 |
| `obs_bird` | Binary (0/1) | `0 or 1` | 1 if obstacle is a flying pterodactyl, else 0 |
| `obs_y` | Float | `0 to 150` | Vertical height of obstacle (critical for bird ducking) |
| `obs_h` | Float | `0 to 50` | Height in pixels of obstacle hitbox |
| `obs_w` | Float | `0 to 75` | Width in pixels of obstacle (single vs double/triple cluster) |
| `ducking` | Binary (0/1) | `0 or 1` | Current dinosaur posture (1 if already ducking) |
| `ttc` | Float | `0 to 80` | **Time-To-Collision**: Frames until impact |

---

## 4. The Secret Weapon: Time-To-Collision (TTC)

Why is `dist` alone not enough?

### The Physics Dilemma:
* At the start of the game, `speed = 6.0`. An obstacle at `120px` away takes **20 frames** to reach you.
* At high speed, `speed = 13.0`. An obstacle at `120px` away takes **9.2 frames** to reach you.

If the neural network only looked at `dist`, a jump timed at `120px` would jump **way too early** at low speed, or **way too late** at high speed!

### The Solution:
We derive the kinematic Time-To-Collision feature:

$$\text{TTC} = \frac{\text{dist}}{\max(\text{speed}, 0.1)}$$

TTC tells the neural network **exactly how many frames of life the dinosaur has before impact**, regardless of game speed!

---

## 5. Data Cleaning Pipeline (`prepare_data.py`)

Raw human or bot gameplay recordings contain noisy data. If you train on dirty data, the AI learns bad habits.

Here is how [`training/prepare_data.py`](../training/prepare_data.py) cleans the raw recording:

```python
# 1. Drop the Intro Animation
# At game start, the dinosaur slides in from x=0 to x=21.
# Those frames are not real gameplay, so we drop them.
normal_x = Counter(s["state"]["dino"]["x"] for s in samples).most_common(1)[0][0]
start = next(i for i, s in enumerate(samples) if s["state"]["dino"]["x"] == normal_x)
samples = samples[start:]
```

```python
# 2. Prevent Data Leakage (Obstacle ID Grouping)
# Consecutive frames look 99% identical. If we did a random train/test split,
# the validation set would cheat by seeing neighboring frames from training.
# We group frames by Obstacle ID so whole obstacles are kept together.
obstacle_id = []
current = 0
for prev, cur in zip([None] + samples[:-1], samples):
    if prev is not None and cur["state"]["obstacle"]["x"] > prev["state"]["obstacle"]["x"] + 5:
        current += 1
    obstacle_id.append(current)
```

```python
# 3. Drop Bad Actions (Crash Obstacle & Mid-Air Frames)
for s, oid in zip(samples, obstacle_id):
    if oid == last_id:
        continue  # The last obstacle ended in a crash: do NOT teach the model bad decisions!

    if s["state"]["dino"]["jumping"]:
        continue  # When airborne, holding jump does nothing. Only teach decisions on the ground!
```

---

## 6. Real-World Applications: How to Apply This Anywhere

The exact same feature extraction and TTC pipeline is used across modern engineering:

### 1. Autonomous Drone & Robot Obstacle Avoidance
Instead of feeding full HD camera feeds into a flight controller, a drone uses **LiDAR / Sonar range sensors**:
* `dist`: LiDAR distance reading to wall.
* `speed`: IMU accelerometer velocity.
* `TTC = dist / speed`: Triggers emergency braking when $\text{TTC} < 0.5\text{ seconds}$.

### 2. Automotive ADAS & Smart Braking (Radar + Vision)
Cars like Tesla and Mobileye extract bounding boxes:
* Bounding Box Width $\to$ Target vehicle size.
* Range rate $\to$ Relative speed.
* Time-to-Collision $\to$ Autonomous Emergency Braking (AEB) trigger.

### 3. Industrial Conveyor Belt Sorting
Cameras extract box dimensions (`width`, `height`, `speed`) and send 5 numerical features to a PLC microcontroller to actuate pneumatic pushers in milliseconds.

---

### Summary of Module 1
* **Engineered features** allow neural networks to be **thousands of times smaller and faster** than image-based models.
* **Time-to-Collision ($\text{TTC} = \text{dist} / \text{speed}$)** standardizes physics across varying speeds.
* **Cleaning data** (removing intro frames, mid-air frames, and crash actions) is essential for high accuracy.

➡️ **Next Module:** `02_neural_network_math_and_pytorch.md` (How the MLP computes predictions and how PyTorch trains it).
