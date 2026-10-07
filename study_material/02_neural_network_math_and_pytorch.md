# Module 02: Neural Network Math & PyTorch Training

> **Goal:** Understand how Multi-Layer Perceptrons (MLP) process numerical features, master the underlying linear algebra and activation math, solve extreme class imbalance with loss weighting, normalize sensor inputs for microcontroller stability, and train the model using PyTorch.

---

## 1. Neural Network Architecture: The Multi-Layer Perceptron (MLP)

For tabular and low-dimensional sensor data (such as our 10 game features), a **Multi-Layer Perceptron (MLP)** is the ideal deep learning architecture. It provides high non-linear expressiveness with minimal memory and compute requirements.

### Architecture Overview

```
Input Layer (10 features)
       │
   [10 × 16 Weights + 16 Biases] ──> Linear Layer 1
       │
   [ReLU Activation]
       │
   Hidden Layer 1 (16 neurons)
       │
   [16 × 16 Weights + 16 Biases] ──> Linear Layer 2
       │
   [ReLU Activation]
       │
   Hidden Layer 2 (16 neurons)
       │
   [16 × 3 Weights + 3 Biases]   ──> Linear Layer 3 (Output)
       │
   Output Logits (3 neurons: [RUN, JUMP, DUCK])
       │
   [Argmax Selection] ─────────────> Winning Action (0, 1, or 2)
```

### Parameter Count Breakdown

| Layer | Input Size | Output Size | Weights | Biases | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Layer 1** | 10 | 16 | $10 \times 16 = 160$ | 16 | 176 |
| **Linear Layer 2** | 16 | 16 | $16 \times 16 = 256$ | 16 | 272 |
| **Linear Layer 3** | 16 | 3 | $16 \times 3 = 48$ | 3 | 51 |
| **Total Model** | - | - | **464** | **35** | **499 floats** (~1.99 KB) |

A 499-parameter model fits comfortably inside the 4 MB Flash ROM of a $3 NodeMCU ESP8266 microcontroller!

---

## 2. The Forward Pass: Step-by-Step Math

Let us trace how 10 numbers turn into a jump decision.

### Step 1: Input Normalization
Raw features have vastly different physical units:
* `dist` is $0$ to $600$ pixels.
* `speed` is $6.0$ to $13.0$ px/frame.
* `ducking` is $0$ or $1$.

If we pass unnormalized numbers into a neural network, large numbers (`dist = 500`) dominate small numbers (`ducking = 1`), destabilizing gradient updates and causing numerical instability on 32-bit embedded hardware.

We normalize each feature $x_i$ using its dataset mean $\mu_i$ and standard deviation $\sigma_i$:

$$x_{\text{norm}, i} = \frac{x_i - \mu_i}{\sigma_i + \epsilon}$$

We then clip the normalized value to $[-5.0, 5.0]$ to eliminate extreme outliers:

$$x_{\text{clipped}, i} = \max\left(-5.0, \min(5.0, x_{\text{norm}, i})\right)$$

---

### Step 2: Dense Layer Computation (Matrix Multiplication)
For a layer with input vector $\vec{x}$ of size $N$ and output size $M$:

$$\vec{z} = \mathbf{W} \cdot \vec{x} + \vec{b}$$

For each individual neuron $j$ from $0$ to $M-1$:

$$z_j = \sum_{i=0}^{N-1} \left( W_{j, i} \cdot x_i \right) + b_j$$

---

### Step 3: Non-Linear Activation (ReLU)
Without non-linear activation functions, stacking 100 linear layers is mathematically equivalent to just one single linear layer ($W_2(W_1 x + b_1) + b_2 = W' x + b'$). 

To allow the network to learn complex decision boundaries (e.g., *"If speed is high AND distance is medium AND obstacle is tall, then JUMP"*), we apply the **Rectified Linear Unit (ReLU)**:

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

Why ReLU is preferred for Edge AI:
* **Zero expensive transcendental operations:** Unlike Sigmoid ($\frac{1}{1 + e^{-z}}$) or Tanh ($\frac{e^z - e^{-z}}{e^z + e^{-z}}$), ReLU requires **zero floating-point exponent calculations**.
* On an 80 MHz microcontroller, calculating $e^x$ takes dozens of clock cycles. A ReLU check takes **1 assembly instruction** (`if (z < 0) z = 0;`).
* Solves the vanishing gradient problem during training.

---

### Step 4: Output Layer and Action Selection
The final layer outputs 3 unconstrained numbers called **Logits**:

$$\vec{z}_{\text{out}} = [z_{\text{RUN}}, z_{\text{JUMP}}, z_{\text{DUCK}}]$$

During training, we convert logits into probabilities using **Softmax**:

$$P(\text{action} = k) = \frac{e^{z_k}}{\sum_{j=0}^{2} e^{z_j}}$$

During **microcontroller inference**, we do **NOT** compute Softmax! Because $e^x$ is strictly monotonically increasing:

$$\operatorname{argmax}_k \left( \frac{e^{z_k}}{\sum e^{z_j}} \right) \equiv \operatorname{argmax}_k (z_k)$$

We simply pick the index of the highest logit. This saves precious CPU cycles on hardware.

```
Logits: [RUN: 1.2, JUMP: 4.8, DUCK: -0.5]
Max value is 4.8 at Index 1 ──> Action: JUMP (1)
```

---

## 3. The Big Machine Learning Challenge: Class Imbalance

In a typical 10-minute game recording:
* The dinosaur spends **85%** of the time running on the ground (`RUN = 0`).
* It spends **12%** of the time jumping over obstacles (`JUMP = 1`).
* It spends **3%** of the time ducking under low birds (`DUCK = 2`).

```
[Dataset Distribution]
RUN:  ██████████████████████████████████████ 85%
JUMP: █████ 12%
DUCK: █ 3%
```

### The "Stupid AI" Trap
If you train standard Cross-Entropy Loss on this raw dataset, the neural network learns a lazy trick: **Always output RUN (0)!**
* Accuracy: **85%** (looks great on paper!).
* Game Performance: **Crashes into the very first cactus within 3 seconds.**

### The Solution: Inverse Class Frequency Loss Weighting
In [`training/train.py`](../training/train.py), we calculate the inverse frequency of each class and pass it into PyTorch's loss function:

$$w_c = \frac{1}{\text{count}(c)}$$

$$\bar{w}_c = \frac{w_c}{\sum_k w_k} \times 3$$

```python
# Calculate inverse frequency weights
class_counts = np.bincount(y_train, minlength=3)
total_samples = len(y_train)

# Classes with fewer samples get much higher penalty when misclassified
class_weights = total_samples / (3.0 * class_counts.astype(np.float32))
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Pass weights to CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
```

If the model misses a single rare `DUCK` frame, it receives a heavy penalty in the loss gradient, forcing it to pay equal attention to jumping and ducking.

---

## 4. PyTorch Training Pipeline Walkthrough

Here is how the training pipeline in [`training/train.py`](../training/train.py) is implemented:

### Defining the PyTorch Model

```python
import torch
import torch.nn as nn

class DinoMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16, output_dim=3):
        super(DinoMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
```

### Training Loop with Adam Optimizer and Early Stopping

```python
model = DinoMLP(input_dim=10, hidden_dim=16, output_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_loss = float('inf')

for epoch in range(1, 101):
    model.train()
    total_train_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # Backward pass (gradient calculation)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # Validation evaluation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            val_loss += criterion(logits, batch_y).item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    val_accuracy = (correct / total) * 100.0
    print(f"Epoch {epoch:03d} | Train Loss: {total_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
```

---

## 5. Exporting to C: Why Code Generation Beats Binary Formats

Standard machine learning deployments export models to `.onnx` or `.tflite` files.

However, loading a binary `.tflite` file on an ESP8266 requires:
1. Embedding a filesystem driver (SPIFFS / LittleFS).
2. Parsing the flatbuffer binary at startup.
3. Allocating a dynamic tensor arena in limited SRAM.

### Our Approach: Direct C Header Generation
In [`training/train.py`](../training/train.py), we extract the PyTorch tensor weights and format them directly as valid C code:

```python
# Extract weights and biases as numpy arrays
w1 = model.net[0].weight.detach().cpu().numpy() # Shape: (16, 10)
b1 = model.net[0].bias.detach().cpu().numpy()   # Shape: (16,)
w2 = model.net[2].weight.detach().cpu().numpy() # Shape: (16, 16)
b2 = model.net[2].bias.detach().cpu().numpy()   # Shape: (16,)
w3 = model.net[4].weight.detach().cpu().numpy() # Shape: (3, 16)
b3 = model.net[4].bias.detach().cpu().numpy()   # Shape: (3,)

# Format into a C header file (model.h)
header_code = f"""// Auto-generated neural network weights
#ifndef MODEL_H
#define MODEL_H

#include <Arduino.h>

#define N_INPUTS   10
#define N_HIDDEN1  16
#define N_HIDDEN2  16
#define N_OUTPUTS  3

// Normalization parameters
const float PROGMEM FEATURE_MEANS[10] = {{{", ".join(f"{m:.6f}f" for m in means)}}};
const float PROGMEM FEATURE_STDS[10]  = {{{", ".join(f"{s:.6f}f" for s in stds)}}};

// Layer 1
const float PROGMEM WEIGHT1[16][10] = ...;
const float PROGMEM BIAS1[16] = ...;

// Layer 2
const float PROGMEM WEIGHT2[16][16] = ...;
const float PROGMEM BIAS2[16] = ...;

// Layer 3
const float PROGMEM WEIGHT3[3][16] = ...;
const float PROGMEM BIAS3[3] = ...;

#endif
"""
```

This turns the neural network into **static constant ROM code**, compiled directly into the microcontroller binary with **zero runtime overhead**.

---

## 6. Real-World Applications: Applying This Math to Industry

The exact same MLP + Feature Normalization + C-export workflow powers mission-critical edge devices:

### 1. Industrial Predictive Maintenance (Bearing Vibration Anomaly)
* **Features (8 inputs):** Accelerometer RMS vibration, peak frequency, kurtosis, motor temperature, RPM, current draw, run hours, ambient temp.
* **Architecture:** 8 $\to$ 16 $\to$ 8 $\to$ 3 (Healthy, Warning, Critical Bearing Fault).
* **Benefit:** Evaluated on an STM32/ESP32 sensor attached to factory motors running 24/7 without cloud dependency.

### 2. Smart Wearables (Fall Detection in Elderly Care)
* **Features (6 inputs):** 3-axis accelerometer + 3-axis gyroscope derivatives, total jerk metric.
* **Architecture:** 6 $\to$ 16 $\to$ 16 $\to$ 2 (Normal Motion vs Sudden Fall).
* **Benefit:** Ultra-low power consumption (<100 µA), instant alert dispatch.

### 3. Smart Agricultural Irrigation (Soil Moisture Trigger)
* **Features (5 inputs):** Soil moisture sensor, air humidity, ambient temperature, solar UV index, historical rain probability.
* **Architecture:** 5 $\to$ 12 $\to$ 2 (Solenoid Valve Open/Close).
* **Benefit:** Solar-powered microcontroller operates autonomously for years in remote farm fields.

---

### Summary of Module 2
* An **MLP with 10 $\to$ 16 $\to$ 16 $\to$ 3 neurons** (499 parameters) provides ample non-linear capacity.
* **ReLU activations** avoid expensive exponential calculations on microcontrollers.
* **Inverse class frequency weighting** stops the model from falling into the "always run" trap.
* **Code-generating C headers (`model.h`)** eliminates the need for heavy runtime interpreters.

➡️ **Next Module:** `03_embedded_c_and_hardware_inference.md` (Compiling and running the C neural network on the ESP8266 microcontroller).
