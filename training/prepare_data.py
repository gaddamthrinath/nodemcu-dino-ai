"""
STEP 1: clean the raw dino recording and flatten it into a CSV the trainer can read.

Run it in the folder that contains your recording:
    python prepare_data.py                 # auto-finds dino-training-*.json in this folder
    python prepare_data.py my_file.json    # or name the file yourself

Output: dino_clean.csv  (one row per frame where the dino is on the ground)

Only the standard library is needed.
"""
import csv
import glob
import json
import sys
from collections import Counter

OUT = "dino_clean.csv"

# The model's inputs, in this exact order. The trainer and the NodeMCU code rely on it.
# The last one (ttc) is derived: time-to-collision = dist / speed, in frames.
FEATURES = ["dist", "speed", "obs_small", "obs_large", "obs_bird",
            "obs_y", "obs_h", "obs_w", "ducking", "ttc"]

ACTION_ID = {"NOTHING": 0, "JUMP": 1, "DUCK": 2}
KNOWN_TYPES = {"cactusSmall", "cactusLarge", "pterodactyl"}
NEW_OBSTACLE_JUMP_PX = 5  # obstacle.x only ever decreases; a rise means a new obstacle appeared

# ------------------------------------------------------------------ 1. load
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    found = sorted(glob.glob("dino-training-*.json"))
    if not found:
        sys.exit("No dino-training-*.json in this folder. Pass the file name: python prepare_data.py file.json")
    path = found[-1]  # newest by name
print(f"reading {path}")

with open(path) as f:
    data = json.load(f)
samples = data["samples"]
print(f"{len(samples)} raw frames | stopped because: {data['metadata'].get('stoppedBecause')}")

# ------------------------------------------------------------------ 2. drop the intro animation
# At the start the dino slides in from x=0 to its normal x. Those frames are not real gameplay.
normal_x = Counter(s["state"]["dino"]["x"] for s in samples).most_common(1)[0][0]
start = next(i for i, s in enumerate(samples) if s["state"]["dino"]["x"] == normal_x)
samples = samples[start:]
print(f"dropped {start} intro frames (dino.x != {normal_x})")

# ------------------------------------------------------------------ 3. give every obstacle an id
# The recording only has "the nearest obstacle". When its x jumps UP, a new obstacle took over.
# The id lets us split train/validation by obstacle later (neighbouring frames look almost identical,
# so a random frame-level split would leak).
obstacle_id = []
current = 0
for prev, cur in zip([None] + samples[:-1], samples):
    if prev is not None and cur["state"]["obstacle"]["x"] > prev["state"]["obstacle"]["x"] + NEW_OBSTACLE_JUMP_PX:
        current += 1
    obstacle_id.append(current)
last_id = current
print(f"{last_id + 1} obstacles found")

# ------------------------------------------------------------------ 4. build the rows
rows = []
skipped_air = 0
for s, oid in zip(samples, obstacle_id):
    if oid == last_id:
        continue  # the last obstacle is the one that ended in the crash: those decisions were bad
    st = s["state"]
    dino, obs = st["dino"], st["obstacle"]
    if dino["jumping"]:
        # In the air the recorded "action" is just the key still being held. The bot cannot make a new
        # decision there, so these frames would only teach "in the air => JUMP". Skip them.
        skipped_air += 1
        continue
    if obs["type"] not in KNOWN_TYPES:
        sys.exit(f"unknown obstacle type {obs['type']!r}")

    dist = obs["x"] - dino["x"]
    speed = st["speed"]
    row = {
        "obstacle_id": oid,
        "frame": s["frame"],
        "dist": dist,
        "speed": round(speed, 4),
        "obs_small": int(obs["type"] == "cactusSmall"),
        "obs_large": int(obs["type"] == "cactusLarge"),
        "obs_bird": int(obs["type"] == "pterodactyl"),
        "obs_y": obs["y"],
        "obs_h": obs["height"],
        "obs_w": obs["width"],
        "ducking": int(dino["ducking"]),
        "ttc": round(dist / max(speed, 0.1), 4),
        "label": ACTION_ID[s["action"]],
    }
    rows.append(row)

# ------------------------------------------------------------------ 5. write the CSV
columns = ["obstacle_id", "frame"] + FEATURES + ["label"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    w.writerows(rows)

# ------------------------------------------------------------------ 6. summary
labels = Counter(r["label"] for r in rows)
dists = [r["dist"] for r in rows]
print(f"\nskipped {skipped_air} in-air frames and the final (crash) obstacle")
print(f"wrote {len(rows)} rows -> {OUT}")
print(f"  NOTHING={labels[0]}  JUMP={labels[1]}  DUCK={labels[2]}")
print(f"  dist range: {min(dists)} .. {max(dists)}")
print(f"  columns: {', '.join(columns)}")