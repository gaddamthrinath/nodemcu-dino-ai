# Module 03: Embedded C & Hardware Inference

> **Goal:** Understand the internal architecture of the ESP8266 NodeMCU, discover why embedding raw weights as C `PROGMEM` constants outperforms heavy TinyML interpreters, walk through the pure C neural network forward pass, and master non-blocking high-speed serial packet parsing.

---

## 💡 Quick Intuition for ML Developers (The Bookshelf vs. Desk Analogy)

If you are coming from **PyTorch, TensorFlow, or Python**, you are used to loading your entire model into memory with `model(x)`. On a laptop or server with 8 to 16 GB of RAM, memory limits rarely cross your mind.

On a microcontroller like the NodeMCU (ESP8266), the situation is drastically different:
* You only have **~80 KB of RAM** (that is smaller than a single low-resolution image!).
* If you tried to load neural network weights into RAM, the chip would instantly run out of memory and crash.

### How this project runs it with zero RAM wasted:
Think of the microcontroller's memory like an office:
* **Flash ROM (4 MB) is like a giant Bookshelf:** The trained neural network weights are printed in a book and stored permanently on the shelf.
* **SRAM (80 KB) is like a tiny Desk:** It only has room for a small scratchpad.

Instead of dumping the entire heavy book onto the tiny desk, the microcontroller acts like a human doing mental math:
1. It points its finger to **one weight in the book on the shelf** (Flash).
2. Multiplies that weight by an input in its head.
3. Adds the result to a running total.
4. **Immediately forgets the weight** and moves its finger to the next one!
5. Only the final answers for each layer (16 numbers for Layer 1, 16 numbers for Layer 2) are written down on the scratchpad (RAM).

**The result:** The model's ~500 weights consume **0 bytes of RAM**, and the decision is computed virtually instantaneously in bare-metal C without runtime overhead!

---

## 1. The Microcontroller Architecture: ESP8266 NodeMCU

The NodeMCU is powered by the **Espressif ESP8266EX** system-on-a-chip.

```
┌────────────────────────────────────────────────────────┐
│                   ESP8266 NodeMCU                      │
├───────────────────────────────┬────────────────────────┤
│ CPU: Tensilica Xtensa LX106   │ SRAM: ~80 KB usable    │
│ 32-bit RISC @ 80 / 160 MHz    │ Flash ROM: 4 MB SPI    │
├───────────────────────────────┴────────────────────────┤
│ Hardware UART: Serial at 115200 to 921600 baud         │
└────────────────────────────────────────────────────────┘
```

### The Memory Dilemma: SRAM vs. Flash ROM
* **SRAM (Static RAM - ~80 KB):** Fast working memory used for running variables, the call stack, and temporary buffers. It is tiny and volatile.
* **Flash ROM (~4 MB):** Non-volatile storage where your compiled Arduino firmware binary lives. It is **50x larger** than SRAM!

If you load neural network weights as standard C variables (`float weight1[16][10]`), the compiler copies them into **SRAM**, consuming valuable dynamic memory and risking heap overflow.

### The Solution: The `PROGMEM` Keyword
By declaring all model parameters with the `PROGMEM` attribute:
```c
const float PROGMEM WEIGHT1[16][10] = { ... };
```
The weights are stored permanently in **Flash ROM**. When the CPU needs a weight during matrix multiplication, it reads it directly from Flash memory on-demand using the `pgm_read_float()` hardware macro, leaving **SRAM 100% free**!

### Execution Model: Does It Load All Weights at Once or Layer-by-Layer?

A key embedded systems architectural question is:  
> *"For each prediction, does the microcontroller copy all weights into SRAM at once, or copy them layer-by-layer?"*

**The Answer:** **Neither! It streams weights scalar-by-scalar (weight-by-weight) directly into the CPU on-demand.**

1. **No Bulk Array Allocation in RAM:** There is no `memcpy()` or intermediate array in SRAM to hold weights.
2. **On-Demand Single Float Reads:** Inside the inner loop (`s += pgm_read_float(&W1[j][i]) * x[i];`), the CPU reads a single weight directly from Flash, multiplies it, adds it to the sum, and immediately discards the weight before reading the next one.
3. **Layer-by-Layer Activations Only:** The **computation** proceeds layer-by-layer, but the only data stored in SRAM are the tiny intermediate neuron outputs: `h1[16]` (64 bytes) and `h2[16]` (64 bytes) allocated on the execution stack.

| Strategy | Memory Mechanism | SRAM (RAM) Cost |
| :--- | :--- | :--- |
| **All weights at once** | Bulk copy 499 floats into SRAM buffer before predicting | **~2,000 bytes** duplicated in RAM |
| **Layer-by-layer weights** | Copy Layer 1 weights into RAM, then Layer 2 weights | **~1,024 bytes** buffer in RAM |
| **On-Demand Streaming (Our Approach)** | Read 1 float at a time directly into CPU registers | **0 bytes** of RAM for weights! |

---

### 🎨 Simple Visual Flow: How Inference Actually Runs

#### Step 1: Weight-by-Weight Processing (Zero RAM Used)

```text
  4 MB Flash Memory (The Bookshelf)
  ┌───────────────────────┐
  │ Weight 1 (e.g.  1.14) │ ──> [ Multiply by Input ] ──> Add to Running Sum
  ├───────────────────────┤               │
  │ Weight 2 (e.g. -0.71) │               ▼
  ├───────────────────────┤     Instantly Forget Weight 1!
  │ Weight 3 ...          │     (Point finger to Weight 2 in the same spot)
  └───────────────────────┘
```

#### Step 2: The Layer-by-Layer Forward Pass

```text
    [ 10 Game Inputs ]
            │
            ▼  (Stream 160 weights from Flash one-by-one)
    [ Hidden Layer 1: 16 Neurons ]  ──> Only 16 numbers saved to RAM (64 bytes)
            │
            ▼  (Stream 256 weights from Flash one-by-one)
    [ Hidden Layer 2: 16 Neurons ]  ──> Only 16 numbers saved to RAM (64 bytes)
            │
            ▼  (Stream 48 weights from Flash one-by-one)
    [ 3 Output Action Scores     ]  ──> Highest score wins: RUN, JUMP, or DUCK!
```

---

### 📌 In Simple Words (Conclusion for Anyone)

* **No RAM is wasted on weights:** All 500 weights live permanently on the Flash "bookshelf". They consume **0 bytes of RAM**.
* **Stream, compute, and forget:** The CPU looks at one weight at a time, multiplies it with an input, adds it to the running sum, and immediately forgets it before reading the next one.
* **Only layer outputs touch RAM:** The only numbers written down in RAM are the small layer outputs (16 numbers for Layer 1, 16 numbers for Layer 2).
* **Blazing fast:** Because the chip does this in pure C without any heavy background software, it computes the decision at bare-metal speed with virtually zero overhead.

---

## 2. Why Raw C Arrays Beat TinyML / TensorFlow Lite Micro

Many developers reflexively reach for **TensorFlow Lite Micro (TFLM)** or **microTVM** when doing Edge AI. Here is why writing pure C is far superior for small microcontrollers:

```
[TensorFlow Lite Micro / Runtime Approach]
Flatbuffer Parser ──> Dynamic Tensor Arena (malloc) ──> Graph Interpreter ──> Op Kernels
Overhead: ~40 KB Flash + 15 KB SRAM + Heavy interpreter overhead

[Direct C Header Approach (Our Project)]
Raw Matrix Multiplications in Compiled C Code (PROGMEM)
Overhead: ~2 KB Flash + 0 KB SRAM + Direct compiled C execution (much faster!)
```

### Comparison Matrix

| Feature | TensorFlow Lite for Microcontrollers | Direct C `PROGMEM` (Our Project) |
| :--- | :--- | :--- |
| **Flash Binary Footprint** | 30 KB – 80 KB interpreter engine | **~1.9 KB total** (weights + code) |
| **Dynamic SRAM Usage** | 10 KB – 30 KB "Tensor Arena" | **0 bytes** (weights stay in Flash) |
| **Dynamic Allocation (`malloc`)** | Often required for tensor buffers | **Zero** (100% deterministic stack) |
| **Inference Overhead** | Heavy interpreter graph traversal | **Direct compiled C arithmetic** |
| **Execution Efficiency** | Complex multi-layer runtime | **Zero interpreter layers** |
| **External Dependencies** | Heavy C++ libraries & toolchains | **Standard C / Arduino core only** |

### The Arduino Macro Gotcha: `BIAS1` vs `B1`
When designing the C header generator, we encountered an interesting embedded bug:
In the Arduino core file `binary.h`, standard binary constants are defined as preprocessor macros:
```c
#define B0 0
#define B1 1
#define B10 2
```
If you name your bias arrays `B1`, `B2`, `B3`, the C preprocessor expands `const float B1[16]` into `const float 1[16]`, throwing a cryptic syntax error!
Naming them `BIAS1`, `BIAS2`, `BIAS3` avoids this conflict.

---

## 3. The Pure C Inference Engine Walkthrough

Inside [`firmware/dino_nodemcu/dino_nodemcu.ino`](../firmware/dino_nodemcu/dino_nodemcu.ino) and [`models/model.h`](../models/model.h), the complete inference engine is implemented in just **45 lines of pure C**:

### Step 1: Input Normalization in C

```c
int dino_predict(const float raw_inputs[10], float* out_confidence) {
    float norm_in[N_INPUTS];

    // Normalize and clip inputs: (x - mean) / std, clamped to [-5.0, 5.0]
    for (int i = 0; i < N_INPUTS; i++) {
        float mean = pgm_read_float(&FEATURE_MEANS[i]);
        float std_val = pgm_read_float(&FEATURE_STDS[i]);
        if (std_val < 1e-6f) std_val = 1e-6f; // Avoid division by zero

        float val = (raw_inputs[i] - mean) / std_val;
        if (val > 5.0f) val = 5.0f;
        if (val < -5.0f) val = -5.0f;
        norm_in[i] = val;
    }
```

---

### Step 2: Hidden Layer 1 (Matrix Multiplication + ReLU)

```c
    float h1[N_HIDDEN1];
    for (int j = 0; j < N_HIDDEN1; j++) {
        // Initialize accumulator with the neuron's bias from Flash
        float sum = pgm_read_float(&BIAS1[j]);

        // Dot product: row j of WEIGHT1 dot norm_in
        for (int i = 0; i < N_INPUTS; i++) {
            float w = pgm_read_float(&WEIGHT1[j][i]);
            sum += w * norm_in[i];
        }

        // Fast ReLU activation
        if (sum < 0.0f) sum = 0.0f;
        h1[j] = sum;
    }
```

---

### Step 3: Hidden Layer 2 (Matrix Multiplication + ReLU)

```c
    float h2[N_HIDDEN2];
    for (int j = 0; j < N_HIDDEN2; j++) {
        float sum = pgm_read_float(&BIAS2[j]);
        for (int i = 0; i < N_HIDDEN1; i++) {
            float w = pgm_read_float(&WEIGHT2[j][i]);
            sum += w * h1[i];
        }

        // Fast ReLU activation
        if (sum < 0.0f) sum = 0.0f;
        h2[j] = sum;
    }
```

---

### Step 4: Output Layer & Zero-Cost Argmax Selection

```c
    float out_logits[N_OUTPUTS];
    int best_action = 0;
    float max_logit = -1e9f;

    for (int j = 0; j < N_OUTPUTS; j++) {
        float sum = pgm_read_float(&BIAS3[j]);
        for (int i = 0; i < N_HIDDEN2; i++) {
            float w = pgm_read_float(&WEIGHT3[j][i]);
            sum += w * h2[i];
        }
        out_logits[j] = sum;

        // In-line Argmax search: find highest score without computing Softmax
        if (sum > max_logit) {
            max_logit = sum;
            best_action = j;
        }
    }

    if (out_confidence != NULL) {
        *out_confidence = max_logit;
    }

    return best_action; // Returns: 0 = RUN, 1 = JUMP, 2 = DUCK
}
```

---

## 4. High-Speed Non-Blocking Serial Communication

Chrome Dino runs at **60 to 80 frames per second**. That gives our entire system a **budget of only 12 to 16 milliseconds per frame**.

If your microcontroller code uses `delay(10)` or blocking calls like `Serial.readStringUntil('\n')`, it will stall the CPU, drop frames, and cause the dinosaur to crash.

### The Non-Blocking Character Accumulator
We use a small 128-byte static buffer that collects incoming UART bytes as they arrive without stopping the main loop:

```c
static char rx_buf[128];
static uint8_t rx_idx = 0;

void loop() {
    // Non-blocking read: Process whatever bytes are currently in the UART FIFO
    while (Serial.available() > 0) {
        char c = (char)Serial.read();

        // End of packet reached (newline received)
        if (c == '\n' || c == '\r') {
            if (rx_idx > 0) {
                rx_buf[rx_idx] = '\0'; // Null-terminate string
                process_command(rx_buf);
                rx_idx = 0;            // Reset buffer pointer
            }
        } 
        else if (rx_idx < sizeof(rx_buf) - 1) {
            rx_buf[rx_idx++] = c;      // Accumulate byte
        }
    }
}
```

---

### High-Speed Tokenization with `strtok()`

When a packet like `"120.5,9.2,1,0,0,100.0,35.0,20.0,0,13.09\n"` arrives, we parse the 10 float values using standard C `strtok()`:

```c
void process_command(char* line) {
    // Check for ping / handshake
    if (strcmp(line, "PING") == 0) {
        Serial.println("PONG:NODEMCU_DINO_AI_V1");
        return;
    }

    float inputs[10];
    int count = 0;

    // Fast comma tokenization
    char* token = strtok(line, ",");
    while (token != NULL && count < 10) {
        inputs[count++] = (float)atof(token);
        token = strtok(NULL, ",");
    }

    if (count == 10) {
        unsigned long t_start = micros();
        float confidence = 0.0f;

        // Run inference on bare metal
        int action = dino_predict(inputs, &confidence);
        unsigned long elapsed_us = micros() - t_start;

        // Reply immediately to bridge: PRED:<action>,<confidence>,<micros>
        Serial.print("PRED:");
        Serial.print(action);
        Serial.print(",");
        Serial.print(confidence, 2);
        Serial.print(",");
        Serial.println(elapsed_us);
    }
}
```

---

## 5. Real-World Applications: Hard Real-Time Edge Systems

The techniques demonstrated here—`PROGMEM` static arrays, pure C forward pass, and zero-allocation ring buffers—are the foundation of commercial embedded systems:

### 1. Automotive Airbag & Active Suspension ECUs
* **Requirement:** Must decide to trigger actuators within $< 1.0\text{ ms}$ of impact.
* **Architecture:** Static C neural network evaluating 3-axis crash accelerometer inputs directly on an automotive-grade MCU. Dynamic memory allocation (`malloc`) is **strictly forbidden** in automotive ISO 26262 standards.

### 2. High-Speed Industrial Sorting Flaps
* **Requirement:** Optical sensors track pills or nuts moving at 5 meters/sec down a chute.
* **Architecture:** An 80 MHz microcontroller parses optical sensor values, predicts defective items in real time, and fires a pneumatic solenoid air jet to blow the defective pill into a reject bin.

### 3. Drone Flight Stabilization (Neuro-PID)
* **Requirement:** 1,000 Hz attitude update rate (1.0 ms budget).
* **Architecture:** Microcontroller computes sensor fusion (gyro + accel) and runs a tiny neural net to adjust motor PWM pulses against wind gusts.

---

### Summary of Module 3
* `PROGMEM` stores neural network weights in **Flash ROM**, keeping dynamic **SRAM free**.
* Pure C inference executes directly on bare metal, eliminating runtime interpreter overhead.
* Skipping Softmax during inference and using direct **Argmax on logits** saves CPU clock cycles.
* **Non-blocking UART buffers + `strtok()`** ensure fast, non-blocking serial communication at high frame rates.

➡️ **Next Module:** `04_realtime_bridge_and_realworld_applications.md` (Connecting Python Selenium, Serial communication, and translating this to industry projects).
