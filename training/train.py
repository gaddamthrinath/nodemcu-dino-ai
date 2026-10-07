"""
STEP 2: train the tiny MLP on dino_clean.csv (the output of prepare_data.py).

    pip install torch numpy
    python train_dino_model.py                 # reads dino_clean.csv
    python train_dino_model.py other.csv

Outputs:
    model.h     C header for the NodeMCU (weights in flash + a ready forward pass)
    model.json  the same weights, for testing the model in the browser first
"""
import csv
import json
import sys

import numpy as np
import torch
import torch.nn as nn

import os
import shutil

# Flexible dataset location
if len(sys.argv) > 1:
    CSV = sys.argv[1]
else:
    candidates = [
        "data/dino_clean.csv",
        "../data/dino_clean.csv",
        "dino_clean.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "dino_clean.csv"),
        os.path.join(os.path.dirname(__file__), "dino_clean.csv"),
    ]
    CSV = next((p for p in candidates if os.path.exists(p)), "dino_clean.csv")

# Fixed divisors that squash every input to roughly -1..1. They are written into model.h, so the
# NodeMCU normalises exactly like this script did. Inputs are also clipped to -1..1.
SCALE_BY_NAME = {
    "dist": 500, "speed": 13, "obs_small": 1, "obs_large": 1, "obs_bird": 1,
    "obs_y": 150, "obs_h": 50, "obs_w": 75, "ducking": 1, "ttc": 80,
}
HIDDEN = 16
EPOCHS = 120
BATCH = 256
LR = 3e-3
TRAIN_FRACTION = 0.8      # first 80% of OBSTACLES train, last 20% validate
NAMES = ["nothing", "jump", "duck"]

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------- 1. load the CSV
with open(CSV) as f:
    reader = csv.reader(f)
    header = next(reader)
    table = np.array([[float(v) for v in row] for row in reader], dtype=np.float32)

FEATURES = header[2:-1]  # between (obstacle_id, frame) and label
assert FEATURES[0] == "dist" and FEATURES[1] == "speed" and FEATURES[-1] == "ttc", \
    "feature order changed: model.h assumes dist, speed first and ttc last"
SCALE = np.array([SCALE_BY_NAME[n] for n in FEATURES], dtype=np.float32)

obstacle_id = table[:, 0].astype(int)
X = np.clip(table[:, 2:-1] / SCALE, -1, 1)  # normalise + clip
y = table[:, -1].astype(np.int64)
print(f"{len(X)} rows, {len(FEATURES)} inputs: {', '.join(FEATURES)}")
print(f"nothing={np.sum(y == 0)} jump={np.sum(y == 1)} duck={np.sum(y == 2)}")

# ---------------------------------------------------------------- 2. split by obstacle, in time order
# Neighbouring frames of the same obstacle are nearly identical, so a random row split would let the
# model "see" its validation rows in training. Splitting by obstacle (earlier vs later) avoids that.
n_obstacles = obstacle_id.max() + 1
cut_id = int(n_obstacles * TRAIN_FRACTION)
is_train = obstacle_id < cut_id
Xtr, ytr = torch.tensor(X[is_train]), torch.tensor(y[is_train])
Xva, yva = torch.tensor(X[~is_train]), torch.tensor(y[~is_train])
print(f"train: {len(Xtr)} rows (obstacles 0..{cut_id - 1}) | validation: {len(Xva)} rows "
      f"(obstacles {cut_id}..{n_obstacles - 1})")

# ---------------------------------------------------------------- 3. model
model = nn.Sequential(
    nn.Linear(len(FEATURES), HIDDEN), nn.ReLU(),
    nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
    nn.Linear(HIDDEN, 3),
)
n_params = sum(p.numel() for p in model.parameters())
print(f"model: {len(FEATURES)} -> {HIDDEN} -> {HIDDEN} -> 3 | {n_params} parameters (~{n_params * 4 / 1024:.1f} KB)")

# JUMP is rare (~1 row per obstacle: the frame the key goes down). Weight classes inversely to their
# frequency so the loss does not just learn "always nothing".
counts = np.bincount(ytr.numpy(), minlength=3).astype(np.float32)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * np.maximum(counts, 1))))
opt = torch.optim.Adam(model.parameters(), lr=LR)


def evaluate(Xs, ys):
    """Per-class recall and the confusion matrix (rows = true class, cols = predicted class)."""
    with torch.no_grad():
        pred = model(Xs).argmax(1).numpy()
    truth = ys.numpy()
    conf = np.zeros((3, 3), dtype=int)
    for t, p in zip(truth, pred):
        conf[t, p] += 1
    return conf.diagonal() / np.maximum(conf.sum(1), 1), conf


# ---------------------------------------------------------------- 4. training loop
best_score, best_state = -1.0, None
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr))
    total = 0.0
    for i in range(0, len(Xtr), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)

    model.eval()
    recall, _ = evaluate(Xva, yva)
    present = np.bincount(yva.numpy(), minlength=3) > 0
    score = float(recall[present].mean())  # mean recall over the classes that exist in validation
    if score > best_score:
        best_score = score
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch:3d} | train loss {total / len(Xtr):.4f} | "
              f"val recall nothing/jump/duck = {recall[0]:.3f}/{recall[1]:.3f}/{recall[2]:.3f}")

model.load_state_dict(best_state)  # keep the best epoch, not just the last one
model.eval()

# ---------------------------------------------------------------- 5. final report
for name, (Xs, ys) in {"TRAIN": (Xtr, ytr), "VALIDATION": (Xva, yva)}.items():
    recall, conf = evaluate(Xs, ys)
    print(f"\n{name}: recall per class")
    for n, r in zip(NAMES, recall):
        print(f"  {n:8s} {r:.3f}")
    print("  confusion matrix (rows = true, cols = predicted: nothing, jump, duck)")
    print(conf)

# ---------------------------------------------------------------- 6. export for the NodeMCU
layers = [m for m in model if isinstance(m, nn.Linear)]
W = [l.weight.detach().numpy() for l in layers]  # shape [out, in]
B = [l.bias.detach().numpy() for l in layers]


def c_array(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    dims = "".join(f"[{d}]" for d in arr.shape)

    def fmt(a):
        if a.ndim == 1:
            return "{" + ", ".join(f"{v:.8e}f" for v in a) + "}"
        return "{\n    " + ",\n    ".join(fmt(sub) for sub in a) + "\n  }"

    return f"const float {name}{dims} PROGMEM = {fmt(arr)};\n"


n_in, n_raw = len(FEATURES), len(FEATURES) - 1
header_text = f"""// Generated by train_dino_model.py. Do not edit by hand.
// raw[] input order ({n_raw} floats): {", ".join(FEATURES[:-1])}
// dino_predict() computes the last input itself: ttc = dist / speed (frames until impact).
// Output: 0 = nothing, 1 = jump, 2 = duck
#pragma once
#include <pgmspace.h>

#define N_RAW {n_raw}
#define N_IN {n_in}
#define N_H1 {HIDDEN}
#define N_H2 {HIDDEN}
#define N_OUT 3

// Each input is divided by SCALE[i] and clipped to -1..1 before the network sees it.
{c_array("SCALE", SCALE)}
{c_array("W1", W[0])}{c_array("BIAS1", B[0])}{c_array("W2", W[1])}{c_array("BIAS2", B[1])}{c_array("W3", W[2])}{c_array("BIAS3", B[2])}
inline int dino_predict(const float *raw) {{
  float in[N_IN], x[N_IN], h1[N_H1], h2[N_H2], o[N_OUT];
  for (int i = 0; i < N_RAW; i++) in[i] = raw[i];
  in[N_RAW] = raw[0] / (raw[1] > 0.1f ? raw[1] : 0.1f);  // ttc = dist / speed
  for (int i = 0; i < N_IN; i++) {{
    float v = in[i] / pgm_read_float(&SCALE[i]);
    x[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
  }}
  for (int j = 0; j < N_H1; j++) {{
    float s = pgm_read_float(&BIAS1[j]);
    for (int i = 0; i < N_IN; i++) s += pgm_read_float(&W1[j][i]) * x[i];
    h1[j] = s > 0 ? s : 0;
  }}
  for (int j = 0; j < N_H2; j++) {{
    float s = pgm_read_float(&BIAS2[j]);
    for (int i = 0; i < N_H1; i++) s += pgm_read_float(&W2[j][i]) * h1[i];
    h2[j] = s > 0 ? s : 0;
  }}
  int best = 0;
  for (int j = 0; j < N_OUT; j++) {{
    float s = pgm_read_float(&BIAS3[j]);
    for (int i = 0; i < N_H2; i++) s += pgm_read_float(&W3[j][i]) * h2[i];
    o[j] = s;
    if (o[j] > o[best]) best = j;
  }}
  return best;
}}
"""
# Auto-export to models/ and firmware/ folders
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

out_h_paths = [
    "model.h",
    os.path.join(root_dir, "models", "model.h"),
    os.path.join(root_dir, "firmware", "dino_nodemcu", "model.h"),
]

for p in out_h_paths:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w") as f:
            f.write(header_text)
    except Exception:
        pass

out_json_paths = [
    "model.json",
    os.path.join(root_dir, "models", "model.json"),
]

for p in out_json_paths:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w") as f:
            json.dump(
                {
                    "features": FEATURES,
                    "scale": SCALE.tolist(),
                    "labels": NAMES,
                    "layers": [{"W": w.tolist(), "b": b.tolist()} for w, b in zip(W, B)],
                },
                f,
            )
    except Exception:
        pass

print("\n✅ Exported model.h and model.json to models/ and firmware/dino_nodemcu/ directories")