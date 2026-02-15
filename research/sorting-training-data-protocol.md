# Training Data Collection & Active Learning Protocol
## Rough Blue Sapphire Melee Sorting System

**Document Version:** 1.0  
**Date:** 2026-02-15  
**Status:** Reference Guide  
**Companion to:** `sorting-system-architecture.md`

---

## Table of Contents

1. [Data Collection Strategy](#1-data-collection-strategy)
2. [Image Capture Specifications](#2-image-capture-specifications)
3. [Annotation Workflow](#3-annotation-workflow)
4. [Data Augmentation](#4-data-augmentation)
5. [Active Learning Loop](#5-active-learning-loop)
6. [Edge Cases & Hard Problems](#6-edge-cases--hard-problems)
7. [Dataset Management](#7-dataset-management)
8. [Benchmarking & Progress Tracking](#8-benchmarking--progress-tracking)
9. [Synthetic Data Considerations](#9-synthetic-data-considerations)

---

## 1. Data Collection Strategy

### 1.1 Phased Dataset Targets

| Phase | Per-Class Target | Total Images | Purpose | Timeline |
|---|---|---|---|---|
| **Phase 1** — Proof of Concept | 100 | 900 | Validate pipeline, test model feasibility | 2–3 days |
| **Phase 2** — Usable Model | 500 | 4,500 | Train model to ≥85% accuracy, begin sorting | 1–2 weeks |
| **Phase 3** — Production | 2,000+ | 18,000+ | Target ≥92% accuracy, active learning maturity | Ongoing (months) |

### 1.2 Time Estimates

**Photography throughput** (solo operator, using vibratory feeder + continuous capture):

| Activity | Rate | Notes |
|---|---|---|
| Automated capture (feeder running) | ~1,200 stones/hr | Camera auto-triggers on stone detection |
| Manual placement (no feeder) | ~300 stones/hr | Tweezers + drop onto imaging area |
| Annotation (labeling) | ~1,500 stones/hr | 2-click per stone (color + clarity) |

**Phase time estimates (solo):**

| Phase | Capture Time | Annotation Time | Total |
|---|---|---|---|
| Phase 1 (900 stones) | ~45 min (feeder) or ~3 hr (manual) | ~36 min | **~1.5–4 hours** |
| Phase 2 (4,500 stones) | ~3.75 hr (feeder) | ~3 hr | **~1 full day** |
| Phase 3 (18,000 stones) | ~15 hr (feeder) | ~12 hr | **~3–4 days spread over weeks** |

**Practical note:** Phase 3 accumulates organically through production sorting via the active learning loop. You don't need to sit down and photograph 18,000 stones — most come from flagged production stones being human-labeled.

### 1.3 Handling Class Imbalance

Some color×clarity combinations are naturally rare in Thai rough melee lots:

| Combination | Expected Frequency | Strategy |
|---|---|---|
| Blue + Transparent | Common (~15–20%) | No action needed |
| Blue + Translucent | Common (~20–25%) | No action needed |
| Blue + Opaque | Moderate (~5–10%) | Monitor, oversample if needed |
| Light + Transparent | Moderate (~5–10%) | May need targeted collection |
| Light + Translucent | Common (~15–20%) | No action needed |
| Light + Opaque | Moderate (~5–8%) | Monitor |
| Inky + Transparent | **Rare (~1–3%)** | **Active oversampling required** |
| Inky + Translucent | Uncommon (~5–8%) | May need targeted collection |
| Inky + Opaque | Moderate (~5–10%) | Monitor |

**Strategies for rare classes:**

1. **Targeted sorting:** When you encounter rare combination stones during manual sorting, set them aside in a labeled tray. Photograph them in dedicated capture sessions.
2. **Weighted sampling during training:** Use PyTorch's `WeightedRandomSampler` to oversample rare classes (see §4).
3. **Class-weighted loss:** Apply inverse-frequency weights to the cross-entropy loss.
4. **Minimum viable count:** Don't train until every class has at least 50 samples. Below that, the model learns noise.
5. **Augmentation emphasis:** Apply heavier augmentation (more transforms per image) to rare classes to boost effective count.

```python
# Calculate class weights from label counts
import numpy as np

label_counts = {
    'blue_transparent': 180, 'blue_translucent': 230, 'blue_opaque': 85,
    'light_transparent': 90, 'light_translucent': 175, 'light_opaque': 70,
    'inky_transparent': 15, 'inky_translucent': 60, 'inky_opaque': 95,
}

total = sum(label_counts.values())
n_classes = len(label_counts)

# Inverse frequency weighting
weights = {k: total / (n_classes * v) for k, v in label_counts.items()}
# Normalize so mean weight = 1.0
mean_w = np.mean(list(weights.values()))
weights = {k: v / mean_w for k, v in weights.items()}

print("Class weights:")
for k, v in sorted(weights.items()):
    print(f"  {k}: {v:.2f}")
# inky_transparent will have very high weight (~7-8×)
```

### 1.4 Multi-Lot Diversity

**Critical:** Don't train on stones from a single lot. Different lots have different color/clarity distributions and visual characteristics (origin, heat treatment history, surface texture).

- Capture from at least **3–5 different lots** before Phase 2 training
- Log `lot_id` in metadata for every stone
- Ensure validation/test splits include stones from lots NOT in training
- Track per-lot accuracy to detect lot-specific bias

---

## 2. Image Capture Specifications

### 2.1 Camera Settings

| Parameter | Value | Rationale |
|---|---|---|
| Resolution | 2448×2048 (5MP) full, or ROI-cropped ~800×800 | Full for collection, ROI for production speed |
| Format | PNG (lossless) for training data; JPEG Q95 for production logging | JPEG compression artifacts destroy subtle color info |
| Color space | Capture as BGR (camera native) → store as sRGB PNG | Standard, reproducible |
| Bit depth | 8-bit per channel (24-bit color) | Sufficient for classification; 12-bit raw if doing advanced color science later |
| White balance | Fixed (from calibration), NOT auto | Auto WB would shift colors between sessions |
| Exposure | Fixed per lighting mode (calibrated at session start) | Auto exposure changes apparent brightness = changes apparent color |
| Gain | Fixed, as low as possible (minimize noise) | Higher gain = more noise = harder classification |

### 2.2 Dual-Lighting Protocol

**Recommendation: Capture TWO images per stone (sequential strobe)**

| Mode | Lighting | Reveals | Exposure (typical) |
|---|---|---|---|
| **Mode A — Backlit** | Backlight ON, ring light OFF | Clarity: transparent stones glow, opaque block light, translucent partially transmit | 500–2,000 µs |
| **Mode B — Top-lit** | Ring light ON, backlight OFF | Color: blue vs light vs inky under standardized reflected light | 1,000–5,000 µs |

**Why two images instead of one combined:**

| Approach | Pros | Cons |
|---|---|---|
| **Dual capture (recommended)** | Clean signal separation; backlit image is pure clarity info, top-lit is pure color info; easier for model to learn | 2× storage; slight complexity; ~10ms delay between captures |
| **Combined (both lights on)** | Simpler, faster, one image | Mixed signals — backlight bleed affects apparent color; dark opaque stones look different than dark inky stones; model must disentangle |
| **Verdict** | **Start with dual capture.** You can always try combined later as an optimization once you have a working model. Dual gives you a clean baseline. |

**Capture sequence per stone:**

```
Stone detected in imaging zone → t₀
  │
  t₀ + 0ms:  Trigger backlight strobe + camera capture → Mode A frame
  t₀ + 10ms: Backlight OFF, ring light ON + camera capture → Mode B frame  
  t₀ + 20ms: Ring light OFF
  │
  Both frames saved with shared stone_id
  Total capture time: ~20ms (stone transit through zone is 50-200ms at typical speeds)
```

**Model input options:**

1. **Two-image input:** Stack Mode A + Mode B as a 6-channel input (H×W×6). Model learns from both simultaneously.
2. **Side-by-side:** Concatenate horizontally into one wide image.
3. **Separate models:** One for color (Mode B), one for clarity (Mode A). Simpler but loses cross-feature learning.
4. **Recommended:** Option 1 (6-channel input) — it's the cleanest and most information-rich.

```python
import cv2
import numpy as np

def load_stone_pair(stone_id, data_dir):
    """Load backlit + toplit pair as 6-channel tensor."""
    backlit = cv2.imread(f"{data_dir}/backlit/{stone_id}.png")   # H×W×3 BGR
    toplit = cv2.imread(f"{data_dir}/toplit/{stone_id}.png")     # H×W×3 BGR
    
    # Convert BGR → RGB
    backlit = cv2.cvtColor(backlit, cv2.COLOR_BGR2RGB)
    toplit = cv2.cvtColor(toplit, cv2.COLOR_BGR2RGB)
    
    # Stack to 6-channel: [R_back, G_back, B_back, R_top, G_top, B_top]
    combined = np.concatenate([backlit, toplit], axis=2)  # H×W×6
    return combined
```

**Note on MobileNetV3 modification for 6-channel input:**

```python
import torch
import torchvision.models as models

model = models.mobilenet_v3_small(pretrained=True)

# Replace first conv layer: 3 channels → 6 channels
old_conv = model.features[0][0]
new_conv = torch.nn.Conv2d(
    6, old_conv.out_channels, 
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride, 
    padding=old_conv.padding, 
    bias=old_conv.bias is not None
)
# Initialize: copy pretrained weights for first 3 channels, duplicate for channels 4-6
with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    new_conv.weight[:, 3:] = old_conv.weight  # Initialize backlit channels same as RGB
model.features[0][0] = new_conv
```

### 2.3 Background Requirements

| Requirement | Specification | Rationale |
|---|---|---|
| Surface | Matte, non-reflective | Prevents glare/reflections contaminating stone image |
| Color (Mode A, backlit) | White/bright — the backlight panel IS the background | Transparent stones blend with bright background; opaque stand out as dark |
| Color (Mode B, top-lit) | **Neutral dark gray (18% gray)** | Doesn't cast color onto stone; not pure black (would lose dark inky stones) |
| Implementation | Clear acrylic channel over backlight; matte gray paint or tape on channel walls above stone level | Channel itself serves as the imaging surface |
| Consistency | Same background for ALL images | Model learns background-relative features; changing it invalidates training data |

### 2.4 Session Consistency Protocol

**Before every capture session (5-minute checklist):**

1. **Power on lights 5 minutes early** — LEDs stabilize thermally in ~2-3 minutes
2. **Capture reference frame** — photograph empty channel (backlit + top-lit modes)
3. **Check backlight intensity** — compare reference area to baseline value (stored from calibration)
   - If drift > 5%: adjust LED driver current or re-calibrate exposure
4. **Photograph color reference** — X-Rite ColorChecker or reference gray card under ring light
   - Software computes white balance correction matrix
   - If ΔE > 3.0 from baseline: investigate (dirty lens? LED aging?)
5. **Photograph reference stones** — run 5 known reference stones (one per challenging class)
   - Visual sanity check: do they look the same as last session?
   - If not: stop and troubleshoot before collecting data

**Reference stone kit:**
- Keep 5–10 stones permanently labeled and stored in a compartmented box
- At least one stone per extreme: best Blue+Transparent, worst Inky+Opaque, and a borderline stone
- Photograph them on Day 1 as the gold standard comparison
- Replace if damaged or lost

**Session metadata to log:**

```json
{
    "session_id": "2026-02-15_001",
    "timestamp_start": "2026-02-15T09:30:00+07:00",
    "lot_id": "LOT-2026-007",
    "lot_description": "Thai melee, Chanthaburi market, mixed grade",
    "backlight_exposure_us": 1000,
    "ringlight_exposure_us": 3000,
    "camera_gain": 2.0,
    "white_balance_matrix": [[1.02, 0, 0], [0, 1.00, 0], [0, 0, 0.97]],
    "reference_backlight_intensity": 215,
    "reference_check_passed": true,
    "ambient_temp_c": 28,
    "notes": "Stones pre-washed and air-dried"
}
```

---

## 3. Annotation Workflow

### 3.1 Labeling Interface Design

**Recommended: Simple custom web tool (FastAPI + HTML)**

For a solo operator, a folder-based system works but a minimal web UI is faster and less error-prone. Build it in ~100 lines:

```python
# annotate_server.py — Minimal annotation tool
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json, os, glob

app = FastAPI()
DATA_DIR = "data/unlabeled"
LABELS_FILE = "data/labels.jsonl"

@app.get("/", response_class=HTMLResponse)
async def annotate():
    # Find next unlabeled stone
    labeled_ids = set()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE) as f:
            for line in f:
                labeled_ids.add(json.loads(line)["stone_id"])
    
    for img in sorted(glob.glob(f"{DATA_DIR}/toplit/*.png")):
        stone_id = os.path.basename(img).replace(".png", "")
        if stone_id not in labeled_ids:
            break
    else:
        return "<h1>All stones labeled! 🎉</h1>"
    
    return f"""
    <html><body style="background:#222;color:#fff;font-family:sans-serif;text-align:center">
    <h2>Stone: {stone_id} ({len(labeled_ids)} labeled)</h2>
    <div style="display:flex;justify-content:center;gap:20px">
        <div><p>Backlit (Clarity)</p><img src="/images/backlit/{stone_id}.png" width="256"></div>
        <div><p>Top-lit (Color)</p><img src="/images/toplit/{stone_id}.png" width="256"></div>
    </div>
    <form method="POST" action="/label">
        <input type="hidden" name="stone_id" value="{stone_id}">
        <h3>Color</h3>
        <button name="color" value="blue" style="font-size:24px;padding:15px 30px;background:#4488ff;color:#fff;border:none;cursor:pointer">💎 Blue</button>
        <button name="color" value="light" style="font-size:24px;padding:15px 30px;background:#aaccff;color:#333;border:none;cursor:pointer">💡 Light</button>
        <button name="color" value="inky" style="font-size:24px;padding:15px 30px;background:#112244;color:#fff;border:none;cursor:pointer">🌑 Inky</button>
        <h3>Clarity</h3>
        <button name="clarity" value="transparent" style="font-size:24px;padding:15px 30px;background:#fff;color:#333;border:none;cursor:pointer">🔍 Transparent</button>
        <button name="clarity" value="translucent" style="font-size:24px;padding:15px 30px;background:#ccc;color:#333;border:none;cursor:pointer">🌫️ Translucent</button>
        <button name="clarity" value="opaque" style="font-size:24px;padding:15px 30px;background:#555;color:#fff;border:none;cursor:pointer">⬛ Opaque</button>
        <br><br>
        <button name="skip" value="true" style="font-size:16px;padding:10px 20px">⏭️ Skip / Uncertain</button>
    </form>
    </body></html>
    """

# Note: actual implementation needs POST handler, proper form parsing, etc.
# This is the concept — a two-click-per-stone annotation interface.
```

**Workflow:** Open browser → see stone images → click color → click clarity → next stone. ~2 seconds per stone.

**Alternative — Folder-based (simpler, no code):**

```
data/
├── labeled/
│   ├── blue_transparent/
│   │   ├── toplit/
│   │   │   ├── S00001.png
│   │   │   └── S00002.png
│   │   └── backlit/
│   │       ├── S00001.png
│   │       └── S00002.png
│   ├── blue_translucent/
│   ├── blue_opaque/
│   ├── light_transparent/
│   ├── light_translucent/
│   ├── light_opaque/
│   ├── inky_transparent/
│   ├── inky_translucent/
│   └── inky_opaque/
└── unlabeled/
    ├── toplit/
    └── backlit/
```

Drag-and-drop files into the correct folder using file manager. Simple but slower (~5 sec/stone) and error-prone.

**Recommendation:** Use the web tool. Build it once, save hours of labeling time.

### 3.2 Annotation Guidelines

#### 3.2.1 Color Classes

**Assessed under TOP-LIT (ring light) image only.**

| Class | Definition | Visual Cues | Common Confusion |
|---|---|---|---|
| **Blue** | Medium to rich blue saturation. The "good" color. Would be considered standard sapphire blue by a buyer. | Clear blue hue, moderate to high saturation. Not dark enough to lose hue, not pale enough to look colorless. | Borderline Light: if you'd hesitate to call it "blue sapphire," it's Light. Borderline Inky: if you can still see blue hue (not just black), it's Blue. |
| **Light** | Pale, washed-out, very light blue, near-colorless, or grayish-blue. Stones that lack color saturation. Includes heat-treatment candidates. | Looks pale, whitish-blue, grayish, or almost colorless. You can see through it but it doesn't look "blue." | Borderline Blue: compare to reference Blue stone side by side. If noticeably paler, it's Light. |
| **Inky** | Very dark blue to black. Over-saturated. Color appears as dark blue-black, navy, or midnight. | Looks very dark, almost black. Blue hue is barely visible or absent under standard lighting. Even under bright ring light, appears dark. | Borderline Blue: look at the darkest face — if it reads as black/near-black, it's Inky. If you can clearly see blue, it's Blue. |

**Decision tree for color:**

```
Look at stone under ring light
  │
  ├── Very dark (hard to see blue hue, appears navy/black)? → INKY
  │
  ├── Very pale (washed out, grayish, almost colorless)? → LIGHT
  │
  └── Clearly blue (medium saturation, recognizable sapphire color)? → BLUE
  
When in doubt between two:
  - Blue vs Light: compare to reference Blue. If noticeably lighter → LIGHT
  - Blue vs Inky: can you see CLEAR blue hue on any face? Yes → BLUE, No → INKY
  - Light vs Inky: this combination is rare. Very dark = Inky, very pale = Light.
```

#### 3.2.2 Clarity Classes

**Assessed under BACKLIT (LED panel) image only.**

| Class | Definition | Visual Cues | Common Confusion |
|---|---|---|---|
| **Transparent** | Light passes through freely. Stone appears bright/glowing on backlight. Internal features may be visible. | On backlit image: stone is bright, close to background brightness. You can "see through" it. Edges may be defined but interior is luminous. | Borderline Translucent: if brightness is >60% of backlight intensity → Transparent. If you see significant dark patches → Translucent. |
| **Translucent** | Light partially passes through. Stone glows but not brightly. Some areas transmit, others don't. | On backlit image: stone is dimmer than background but not dark. Appears frosted, milky, or unevenly lit. Some zones bright, some dark. | Most common class. When uncertain between Transparent and Translucent: choose Translucent (it's the safe middle). |
| **Opaque** | Light does not pass through. Stone appears as a dark silhouette on backlight. | On backlit image: stone is dark, near-black. Silhouette only. No internal glow or light transmission visible. | Borderline Translucent: look at thinnest edges — if ANY light comes through at edges → Translucent. If completely dark → Opaque. |

**Decision tree for clarity:**

```
Look at stone under backlight
  │
  ├── Stone appears bright/glowing (>60% of backlight brightness)? → TRANSPARENT
  │
  ├── Stone appears as dark silhouette (no light transmission)? → OPAQUE
  │
  └── Some light passes through but stone is clearly dimmer than background? → TRANSLUCENT

When in doubt:
  - Transparent vs Translucent: measure transmittance. >60% = Transparent.
    Or: can you clearly see THROUGH it? Yes → Transparent.
  - Translucent vs Opaque: check thinnest edge. ANY glow → Translucent.
  - Default to TRANSLUCENT when borderline (it's the middle class).
```

#### 3.2.3 Edge Cases

| Situation | Rule |
|---|---|
| Stone is borderline between two color classes | Label as the class it's "closer to." If genuinely 50/50, label as the less common class (helps balance). Add `"uncertain": true` flag. |
| Stone is borderline between two clarity classes | Default to Translucent (middle class). Add uncertainty flag. |
| Stone has mixed color zones (half blue, half light) | Label by dominant zone (>50% of visible area). Add `"mixed": true` flag. |
| Stone has mixed clarity zones | Label by overall behavior under backlight. If mostly bright with a dark patch → Transparent. If mostly dark with a bright edge → Opaque. Add flag. |
| Stone is damaged/broken | Label normally if color/clarity are still assessable. Add `"damaged": true` flag. If unassessable → Skip. |
| Not a sapphire (debris, other mineral) | Skip. Add `"reject": true` flag. These go into a separate "non-gem" training category if you want to train rejection. |
| Very small stone (<1.5mm), hard to see | Label if you can make a determination. Skip if too ambiguous. These are important training samples — don't discard them. |

### 3.3 Inter-Annotator Agreement

**Why this matters:** If Duke labels a stone "Blue" and an assistant labels it "Light," the training data is noisy. The model can't learn boundaries that don't exist consistently.

**Protocol:**

1. Duke labels 200 stones (the "gold set")
2. An assistant independently labels the same 200 stones (without seeing Duke's labels)
3. Calculate Cohen's Kappa for both color and clarity:

```python
from sklearn.metrics import cohen_kappa_score

# duke_labels and assistant_labels are lists of class names
kappa_color = cohen_kappa_score(duke_color_labels, assistant_color_labels)
kappa_clarity = cohen_kappa_score(duke_clarity_labels, assistant_clarity_labels)

print(f"Color agreement (κ): {kappa_color:.3f}")
print(f"Clarity agreement (κ): {kappa_clarity:.3f}")

# Interpretation:
# κ > 0.80: Excellent agreement — annotation guidelines are clear
# κ 0.60-0.80: Substantial — guidelines need refinement for edge cases
# κ 0.40-0.60: Moderate — significant disagreements, review and retrain annotator
# κ < 0.40: Poor — guidelines are broken, annotator doesn't understand the task
```

4. For stones where they disagree:
   - Review together, discuss the specific stone
   - Update guidelines to cover that case
   - Re-label disagreements with Duke's decision as ground truth

5. **Target: κ ≥ 0.75 for both color and clarity**

**Solo operator shortcut:** If Duke is the only annotator, do a self-consistency check. Label 100 stones, wait 2+ weeks, re-label the same stones blind. Measure self-agreement (κ). If κ < 0.85 with yourself, the class boundaries need clearer definition.

### 3.4 Label Format

**Primary format: JSONL (one line per stone)**

```jsonl
{"stone_id":"S00001","color":"blue","clarity":"transparent","lot_id":"LOT-2026-007","session_id":"2026-02-15_001","annotator":"duke","timestamp":"2026-02-15T10:05:32+07:00","uncertain":false,"mixed":false,"damaged":false,"notes":""}
{"stone_id":"S00002","color":"light","clarity":"translucent","lot_id":"LOT-2026-007","session_id":"2026-02-15_001","annotator":"duke","timestamp":"2026-02-15T10:05:34+07:00","uncertain":true,"mixed":false,"damaged":false,"notes":"borderline blue/light"}
```

**Why JSONL over CSV:** Extensible (add fields without breaking parsers), handles notes with commas, one line per record = easy append + grep + stream processing.

**Conversion to folder structure for training:**

```python
import json, shutil, os

def jsonl_to_folders(labels_file, image_dir, output_dir):
    """Convert JSONL labels to folder-per-class structure for PyTorch ImageFolder."""
    with open(labels_file) as f:
        for line in f:
            rec = json.loads(line)
            class_name = f"{rec['color']}_{rec['clarity']}"
            sid = rec['stone_id']
            
            for mode in ['backlit', 'toplit']:
                src = f"{image_dir}/{mode}/{sid}.png"
                dst_dir = f"{output_dir}/{class_name}/{mode}"
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, f"{dst_dir}/{sid}.png")

jsonl_to_folders("data/labels.jsonl", "data/images", "data/by_class")
```

### 3.5 Quality Control

| Check | Frequency | Method |
|---|---|---|
| **Spot re-labeling** | Every 500 labels | Randomly select 50 previously labeled stones, re-label blind, compare. If disagreement >10%, investigate. |
| **Class distribution check** | Every 200 labels | Print counts per class. If any class has <10% of expected, you may be systematically biased. |
| **Image quality check** | Every session start | Review 10 random captures. Check focus, lighting, background. Reject session if >20% are blurry/dark. |
| **Duplicate detection** | Before training | Hash all images. If same stone was captured twice (re-run through feeder), keep only one. |

```python
import hashlib, os, glob
from collections import defaultdict

def find_duplicates(image_dir):
    """Find duplicate images by content hash."""
    hashes = defaultdict(list)
    for path in glob.glob(f"{image_dir}/**/*.png", recursive=True):
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        hashes[h].append(path)
    
    dupes = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    print(f"Found {len(dupes)} duplicate groups ({sum(len(p)-1 for p in dupes.values())} extra copies)")
    return dupes
```

---

## 4. Data Augmentation

### 4.1 Safe Augmentations

| Augmentation | Parameters | Rationale |
|---|---|---|
| **Random rotation** | 0–360°, continuous | Stones have no fixed orientation. Essential. |
| **Horizontal flip** | 50% probability | Symmetric augmentation. |
| **Vertical flip** | 50% probability | Symmetric augmentation. |
| **Random scale** | 0.8×–1.2× | Stones vary in size; model should be scale-invariant. |
| **Random translation** | ±10% of image size | Stone position in ROI may vary slightly. |
| **Gaussian noise** | σ = 0.01–0.03 (normalized) | Simulates sensor noise, dust particles. |
| **Random erasing** | p=0.1, scale=0.02–0.05 | Simulates small occlusions (dust, debris). |

### 4.2 Augmentations to Use with EXTREME CARE

| Augmentation | Parameters | Warning |
|---|---|---|
| **Brightness jitter** | ±5% max | Color brightness IS the classification signal. Keep very subtle. Only simulates minor lighting fluctuation. |
| **Contrast jitter** | ±5% max | Same concern. Clarity features depend on contrast. |
| **CLAHE / histogram equalization** | DO NOT apply as augmentation | This changes the signal. Only apply as preprocessing if used consistently at inference too. |

### 4.3 Augmentations to NEVER USE

| Augmentation | Why NOT |
|---|---|
| **Color jitter (hue shift)** | **Color IS the classification target.** Shifting hue turns a Blue stone into a Light or Inky stone — you'd be mislabeling. |
| **Saturation jitter (>10%)** | Same — saturation distinguishes Blue from Light from Inky. |
| **Heavy Gaussian blur** | Clarity classification depends on sharpness and light transmission patterns. Blur destroys this. |
| **Style transfer / color transfer** | Changes the exact visual properties we're classifying. |
| **Cutout (large regions)** | Small stones — removing a large patch loses too much information. |
| **Mixup / CutMix** | Blending two stone images creates impossible visual patterns and ambiguous labels. |

### 4.4 Recommended Augmentation Pipeline

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training augmentation pipeline
train_transform = A.Compose([
    # Spatial transforms (SAFE — don't change color/clarity)
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,      # ±10% translation
        scale_limit=0.2,      # 0.8×–1.2× scale
        rotate_limit=180,     # Full rotation
        border_mode=0,        # Constant border (black)
        p=0.8
    ),
    
    # Very subtle brightness/contrast (CAREFUL)
    A.RandomBrightnessContrast(
        brightness_limit=0.05,  # ±5% only
        contrast_limit=0.05,    # ±5% only
        p=0.3                   # Only 30% of the time
    ),
    
    # Noise (simulates real sensor conditions)
    A.GaussNoise(
        var_limit=(5.0, 25.0),  # Subtle noise
        p=0.3
    ),
    
    # Small occlusions (simulates dust)
    A.CoarseDropout(
        max_holes=3,
        max_height=8,
        max_width=8,
        min_holes=1,
        min_height=2,
        min_width=2,
        fill_value=0,
        p=0.2
    ),
    
    # Normalize and convert
    A.Normalize(
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],  # 6-channel (ImageNet stats repeated)
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])

# Validation/test — NO augmentation (except normalize)
val_transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])
```

### 4.5 Effective Dataset Multiplier

With the above pipeline applied randomly each epoch:
- Each training image is seen ~5–8× effectively different per epoch
- Over 50 epochs of training: each image generates ~250–400 unique augmented views
- **Effective multiplier: ~6× per epoch** (conservative estimate)

| Phase | Raw Images | Effective per Epoch | Sufficient? |
|---|---|---|---|
| Phase 1 (900) | 900 | ~5,400 | Marginal — expect overfitting, useful only for feasibility |
| Phase 2 (4,500) | 4,500 | ~27,000 | Good — enough for fine-tuning MobileNetV3-Small |
| Phase 3 (18,000) | 18,000 | ~108,000 | Excellent — production-quality training |

---

## 5. Active Learning Loop

### 5.1 System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACTIVE LEARNING LOOP                          │
│                                                                 │
│  ┌──────────┐     ┌──────────────┐     ┌────────────────────┐  │
│  │Production│     │  Model       │     │ Confidence Check   │  │
│  │ Stone    │────▶│  Inference   │────▶│                    │  │
│  │          │     │ color + clar │     │ conf ≥ 0.80 → AUTO │  │
│  └──────────┘     └──────────────┘     │ conf < 0.80 → FLAG │  │
│                                        └───────┬────────────┘  │
│                                            FLAG │              │
│                                                 ▼              │
│                                        ┌────────────────┐      │
│                                        │ Review Queue   │      │
│                                        │ (Dashboard UI) │      │
│                                        └───────┬────────┘      │
│                                                │               │
│                                    Human labels│correct class  │
│                                                ▼               │
│  ┌──────────────┐     ┌──────────────┐  ┌─────────────────┐   │
│  │ Deploy New   │     │  Retrain     │  │ Training Dataset │   │
│  │ Model v(N+1) │◀────│  Model       │◀─│ + new labels     │   │
│  └──────────────┘     └──────────────┘  └─────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Confidence Thresholds

| Combined Confidence* | Action | Estimated % of Stones |
|---|---|---|
| ≥ 0.80 | **Auto-sort** — trusted classification | 70–85% (improves with model maturity) |
| 0.50–0.80 | **Sort + Flag** — sort normally but add to review queue | 10–25% |
| < 0.50 | **Divert to SELECT** — too uncertain to trust, human must decide | 3–10% |

*Combined confidence = min(max color probability, max clarity probability)*

**Why 0.80 and not 0.90 or 0.70:**
- At 0.90: too many stones flagged (30–40%), drowning the human reviewer
- At 0.70: too few flagged, missing genuinely confusing stones that would improve the model
- 0.80 is the sweet spot for a solo operator — review queue is manageable (~200–400 stones per 3,000-stone batch)

**Adjust over time:** As model improves, raise threshold to 0.85 or 0.90 to maintain a steady flow of hard cases.

### 5.3 Uncertainty Sampling Strategy

Not all low-confidence stones are equally valuable for retraining. Prioritize:

```python
def compute_uncertainty_score(color_probs, clarity_probs):
    """
    Higher score = more valuable for active learning.
    Uses entropy-based uncertainty across both classification heads.
    """
    import numpy as np
    
    def entropy(probs):
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log2(probs))
    
    color_entropy = entropy(color_probs)     # Max = log2(3) = 1.585
    clarity_entropy = entropy(clarity_probs)  # Max = log2(3) = 1.585
    
    # Normalize to [0, 1]
    max_entropy = np.log2(3)
    score = (color_entropy + clarity_entropy) / (2 * max_entropy)
    return score

# Prioritize stones by uncertainty score (highest first)
# A stone where model says [0.34, 0.33, 0.33] for color is maximally uncertain
# A stone where model says [0.90, 0.05, 0.05] is confident — low priority
```

**Additional priority signals:**

| Signal | Priority Boost | Rationale |
|---|---|---|
| Model disagrees with existing label | +HIGH | Potential mislabel in training data |
| Rare class predicted (e.g., Inky+Transparent) | +MEDIUM | Rare class samples are especially valuable |
| Near decision boundary (e.g., Blue+Transparent vs Blue+Translucent = CUT vs FLIP) | +HIGH | Errors at this boundary are costly |
| Confidence just below threshold (0.75–0.80) | +LOW | These are almost correct, less learning value |

### 5.4 Batch Retraining Schedule

| Trigger | Action |
|---|---|
| 500 new human-labeled stones accumulated | Retrain model |
| Weekly (regardless of count) | Retrain if ≥100 new labels |
| Accuracy drop detected (QA check fails) | Emergency retrain with investigation |
| New lot introduced with different characteristics | Collect 200+ from new lot, retrain |

**Retraining workflow:**

```bash
# 1. Merge new labels into dataset
python tools/merge_labels.py --new data/review_labels_batch_042.jsonl --into data/labels.jsonl

# 2. Regenerate train/val/test splits (stratified)
python tools/split_dataset.py --labels data/labels.jsonl --output data/splits/

# 3. Train new model version
python src/classification/train.py \
    --data data/splits/ \
    --model-base models/v041.pth \
    --output models/v042.pth \
    --epochs 20 \
    --lr 0.0001

# 4. Evaluate on held-out test set
python src/classification/evaluate.py \
    --model models/v042.pth \
    --test data/splits/test/

# 5. Compare to previous version
python tools/compare_models.py --old models/v041.pth --new models/v042.pth --test data/splits/test/

# 6. If better: export and deploy
python src/classification/export.py --model models/v042.pth --output models/v042.onnx
# Copy to Jetson, convert to TensorRT, restart service
```

### 5.5 Version Control

**Models:**

```
models/
├── v001.pth          # Phase 1 proof of concept
├── v001.onnx
├── v002.pth          # After first active learning batch
├── v002.onnx
├── ...
├── v042.pth          # Current production model
├── v042.onnx
├── v042.trt          # TensorRT engine (Jetson-specific)
└── model_log.jsonl   # Training metadata for every version
```

**model_log.jsonl entry:**

```json
{
    "version": "v042",
    "timestamp": "2026-04-10T14:30:00+07:00",
    "training_samples": 12450,
    "new_samples_since_last": 523,
    "epochs": 20,
    "learning_rate": 0.0001,
    "test_accuracy_color": 0.934,
    "test_accuracy_clarity": 0.912,
    "test_accuracy_combined": 0.891,
    "cost_weighted_accuracy": 0.923,
    "notes": "Added 523 stones from new Ratnapura lot"
}
```

**Dataset versioning:**

Simple approach (no DVC needed for a solo operation):

```
data/
├── labels.jsonl           # Master label file (append-only, git-tracked)
├── labels_backup/
│   ├── labels_2026-02-15.jsonl
│   ├── labels_2026-02-28.jsonl
│   └── labels_2026-03-15.jsonl
├── splits/
│   ├── v042/              # Splits used for model v042
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── v041/
└── images/                # Images never change — append-only
```

- `labels.jsonl` is tracked in git (small text file)
- Images are backed up to external drive (too large for git)
- Before each retrain, snapshot the splits into a version folder
- This gives full reproducibility: "model v042 was trained on splits/v042/train.jsonl"

---

## 6. Edge Cases & Hard Problems

### 6.1 Lighting-Dependent Appearance

**Problem:** Some stones look different under slightly different lighting angles. A blue stone may appear light or inky depending on the crystal axis facing the camera.

**Mitigation:**
- The dual-lighting protocol with fixed geometry eliminates most variation
- Random rotation augmentation means the model sees all orientations during training
- For pleochroic stones (sapphire IS pleochroic), accept that some will be ambiguous
- **Rule:** Label based on what you see in the standard imaging position. Don't rotate the stone to find the "best" color.

### 6.2 Wet vs Dry Stones

**Problem:** Surface water fills micro-cracks and scratches, making translucent stones appear more transparent. Wet stones consistently classify one step "better" in clarity.

**Mitigation:**
- **Standard operating procedure: Sort DRY stones only**
- If stones were washed: air-dry for 30+ minutes or use forced-air dryer
- If wet sorting is unavoidable: note in session metadata and consider training a separate wet model
- Monitor: if clarity distribution suddenly shifts toward transparent, check for moisture

### 6.3 Mixed-Zone Stones

**Problem:** A stone that's half blue, half light. Or part transparent with an opaque inclusion.

**Mitigation:**
- Label by the **dominant** characteristic (>50% of visible area)
- Add `"mixed": true` flag to metadata
- The model will naturally learn to handle these — they'll produce lower confidence scores and get flagged for review
- **Do NOT create additional "mixed" classes** — it fragments the already small per-class dataset
- In production, mixed stones will likely get medium confidence → flagged → human decision. This is correct behavior.

### 6.4 Very Small Stones (<1.5mm)

**Problem:** Fewer pixels to work with. At 1:1 magnification, a 1mm stone is ~290px across, but the stone image after ROI extraction and resize to 128×128 may lose subtle features.

**Mitigation:**
- Use 2:1 magnification if small stones are a large fraction of the lot (1mm → ~580px)
- Include small stones in training data — don't exclude them
- The model should learn that small stones have less visual information and may output lower confidence
- Consider a size-dependent confidence threshold: require higher confidence for small stones before auto-sorting
- **Accept:** <1.5mm stones will always have lower accuracy. This is OK — they're also lower value per stone.

### 6.5 Non-Gem Material

**Problem:** Dust, rock fragments, broken chips, and other debris mixed into the lot.

**Mitigation:**
- Pre-sieve lots to remove obvious debris and fines
- Train a binary "gem vs non-gem" pre-classifier (or threshold on stone shape/size)
- Non-gem material → REJECT bin (default trajectory, no air blast)
- The segmentation step already filters by contour area and aspect ratio — most debris is rejected there
- Keep flagged debris images for training a rejection model later

### 6.6 Model Handling Strategy

| Edge Case | Model Behavior | Bin Assignment |
|---|---|---|
| Mixed color zones | Low color confidence | Sort by dominant, flag for review |
| Mixed clarity zones | Low clarity confidence | Sort by dominant, flag for review |
| Very small stone | Lower confidence overall | Sort if conf > threshold, else SELECT |
| Wet stone | Clarity shifted toward transparent | Should be avoided (SOP: sort dry) |
| Debris/non-gem | High uncertainty on both axes | REJECT (low confidence → default bin) |
| Damaged stone | Variable | Sort normally if assessable, REJECT if not |

**No multi-label needed.** Single color + single clarity per stone. The confidence score naturally handles uncertainty — low confidence = human reviews it.

---

## 7. Dataset Management

### 7.1 Folder Structure

```
sapphire-sorter/
├── data/
│   ├── images/
│   │   ├── raw/                     # Unprocessed camera captures (archival)
│   │   │   └── session_2026-02-15_001/
│   │   │       ├── backlit/
│   │   │       │   ├── S00001.png
│   │   │       │   ├── S00002.png
│   │   │       │   └── ...
│   │   │       ├── toplit/
│   │   │       │   ├── S00001.png
│   │   │       │   └── ...
│   │   │       └── session_meta.json
│   │   │
│   │   └── processed/              # ROI-extracted, resized to 128×128
│   │       ├── backlit/
│   │       │   ├── S00001.png
│   │       │   └── ...
│   │       └── toplit/
│   │           ├── S00001.png
│   │           └── ...
│   │
│   ├── labels.jsonl                # Master label file
│   ├── labels_backup/              # Periodic snapshots
│   │
│   ├── splits/                     # Train/val/test splits per model version
│   │   ├── v001/
│   │   │   ├── train.txt           # List of stone_ids
│   │   │   ├── val.txt
│   │   │   └── test.txt
│   │   └── v042/
│   │
│   └── sessions/                   # Session metadata
│       ├── 2026-02-15_001.json
│       └── 2026-02-15_002.json
│
├── models/
│   ├── v001.pth
│   ├── v001.onnx
│   ├── v042.pth
│   ├── v042.onnx
│   ├── v042.trt
│   └── model_log.jsonl
│
└── production_logs/                # Runtime logs from sorting
    ├── batch_2026-03-01_morning/
    │   ├── stones.db               # SQLite per batch
    │   ├── thumbnails/
    │   └── flagged/                # Full-size images of flagged stones
    └── ...
```

### 7.2 File Naming Convention

```
Stone ID format: S{NNNNNN}  (zero-padded 6-digit sequential)
  Examples: S000001, S000002, ..., S999999

Session ID format: {YYYY-MM-DD}_{NNN}
  Examples: 2026-02-15_001, 2026-02-15_002

Batch ID format: {YYYY-MM-DD}_{description}
  Examples: 2026-03-01_morning, 2026-03-01_ratnapura_lot

Model version: v{NNN}
  Examples: v001, v002, ..., v042
```

### 7.3 Train/Validation/Test Split

**Strategy: 80/10/10, stratified by class, with lot-level separation**

```python
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

def create_splits(labels_file, output_dir, test_holdout_lots=None):
    """
    Create stratified train/val/test splits.
    
    Args:
        labels_file: Path to labels.jsonl
        output_dir: Where to write train.txt, val.txt, test.txt
        test_holdout_lots: List of lot_ids to reserve entirely for test set
                          (prevents data leakage from same-lot stones)
    """
    stones = []
    with open(labels_file) as f:
        for line in f:
            stones.append(json.loads(line))
    
    # Assign class labels for stratification
    for s in stones:
        s['class'] = f"{s['color']}_{s['clarity']}"
    
    # Lot-level holdout for test set (prevents leakage)
    if test_holdout_lots:
        test_stones = [s for s in stones if s['lot_id'] in test_holdout_lots]
        remaining = [s for s in stones if s['lot_id'] not in test_holdout_lots]
    else:
        # Random split (simpler, OK if stones don't re-enter feeder)
        remaining = stones
        test_stones = []
    
    # Split remaining into train + val (+ test if no lot holdout)
    ids = [s['stone_id'] for s in remaining]
    classes = [s['class'] for s in remaining]
    
    if not test_holdout_lots:
        # First split: 90% trainval, 10% test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        trainval_idx, test_idx = next(sss1.split(ids, classes))
        test_stones = [remaining[i] for i in test_idx]
        remaining = [remaining[i] for i in trainval_idx]
        ids = [s['stone_id'] for s in remaining]
        classes = [s['class'] for s in remaining]
    
    # Second split: ~89% train, ~11% val (of remaining = ~80/10 of total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.111, random_state=42)
    train_idx, val_idx = next(sss2.split(ids, classes))
    
    train_ids = [ids[i] for i in train_idx]
    val_ids = [ids[i] for i in val_idx]
    test_ids = [s['stone_id'] for s in test_stones]
    
    os.makedirs(output_dir, exist_ok=True)
    for name, id_list in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        with open(f"{output_dir}/{name}.txt", 'w') as f:
            f.write('\n'.join(id_list))
    
    # Print distribution
    for name, id_list in [('Train', train_ids), ('Val', val_ids), ('Test', test_ids)]:
        print(f"{name}: {len(id_list)} stones")

    return train_ids, val_ids, test_ids
```

### 7.4 Preventing Data Leakage

| Leakage Risk | Prevention |
|---|---|
| Same stone in train and test (re-ran through feeder) | Duplicate detection by image hash (§3.5). Remove before splitting. |
| Stones from same lot in both train and test | Use lot-level holdout: reserve 1–2 lots entirely for test set. |
| Augmented copies of test images in training | Augmentation is applied on-the-fly during training, never to raw dataset. Splits reference stone_ids, not file paths. |
| Label leakage (annotator remembers test stones) | Test set labels are assigned once and never re-annotated. Reference stone kit is separate from test set. |

### 7.5 Backup Strategy

| What | Where | Frequency | Method |
|---|---|---|---|
| `labels.jsonl` | Git repo + cloud | Every annotation session | `git commit && git push` |
| `model_log.jsonl` | Git repo | Every model version | Same |
| Model files (`.pth`, `.onnx`) | External SSD | Every model version | `rsync` to backup drive |
| Raw images | External SSD | Weekly | `rsync --checksum` |
| Processed images | **Not backed up** (can regenerate from raw) | — | — |
| Production logs + SQLite | External SSD | Daily during production runs | Automated `rsync` cron job |

**Minimum backup rule:** If your laptop/dev PC dies, you should be able to reconstruct the entire dataset from the external SSD backup within 1 hour.

```bash
# Weekly backup script
#!/bin/bash
BACKUP_DRIVE="/mnt/backup_ssd/sapphire-sorter"
PROJECT="/home/duke/sapphire-sorter"

echo "Backing up labels and models..."
rsync -av --checksum "$PROJECT/data/labels.jsonl" "$BACKUP_DRIVE/data/"
rsync -av --checksum "$PROJECT/data/labels_backup/" "$BACKUP_DRIVE/data/labels_backup/"
rsync -av --checksum "$PROJECT/models/" "$BACKUP_DRIVE/models/"

echo "Backing up raw images..."
rsync -av --checksum "$PROJECT/data/images/raw/" "$BACKUP_DRIVE/data/images/raw/"

echo "Backing up production logs..."
rsync -av --checksum "$PROJECT/production_logs/" "$BACKUP_DRIVE/production_logs/"

echo "Backup complete: $(date)"
```

---

## 8. Benchmarking & Progress Tracking

### 8.1 Metrics

| Metric | What It Measures | How to Compute |
|---|---|---|
| **Per-class accuracy** | How well each of the 9 classes is identified | Correct predictions / total for that class |
| **Overall accuracy** | Global correctness | Total correct / total predictions |
| **Confusion matrix** (color, 3×3) | Where color errors happen | sklearn `confusion_matrix` |
| **Confusion matrix** (clarity, 3×3) | Where clarity errors happen | Same |
| **Combined confusion matrix** (9×9) | Full picture | 9 color×clarity classes |
| **Cost-weighted accuracy** | Business impact of errors | See §8.2 |
| **Flagged rate** | % of stones with confidence < threshold | Count of flagged / total |
| **Rejection rate** | % of stones diverted to REJECT/SELECT for human review | Count of low-conf / total |

### 8.2 Cost-Weighted Error Matrix

Not all errors are equal. The business impact depends on which bin a stone ends up in vs where it should go.

**Decision mapping (from architecture doc):**

| Color | Clarity | Correct Bin |
|---|---|---|
| Blue | Transparent (≥2mm) | CUT |
| Blue | Transparent (<2mm) | SELECT |
| Blue | Translucent | FLIP |
| Blue | Opaque | FLIP |
| Light | Transparent | SELECT |
| Light | Translucent | FLIP |
| Light | Opaque | FLIP |
| Inky | Transparent | SELECT |
| Inky | Translucent | SELECT |
| Inky | Opaque | REJECT |

**Error costs (relative scale, 1 = minor, 10 = worst):**

| Error | Cost | Explanation |
|---|---|---|
| CUT stone → FLIP bin | **10** | Premium stone lost to bulk flip lot. Lost revenue: $0.50–5+ per stone. |
| CUT stone → SELECT bin | **3** | Not lost — human reviews SELECT and can recover it. Just extra work. |
| CUT stone → REJECT bin | **8** | Premium stone thrown away. May be recoverable if REJECT is re-checked. |
| FLIP stone → CUT bin | **7** | Bad stone gets expensive cutting. Wasted cutting fee ($0.15–0.28) + possible reputational cost. |
| FLIP stone → SELECT bin | **1** | Human reviews, sends to FLIP. Minimal cost — just extra handling. |
| FLIP stone → REJECT bin | **2** | Low-value stone discarded. Minor loss. |
| SELECT stone → CUT bin | **5** | Marginal stone gets cut — may or may not be worth it. Risky. |
| SELECT stone → FLIP bin | **3** | Heat-treatment candidate lost to bulk. Moderate loss. |
| SELECT stone → REJECT bin | **4** | Potentially valuable stone discarded. |
| REJECT stone → any other bin | **1** | Waste material in good bins. Minor contamination, caught during QC. |

**Cost-weighted accuracy formula:**

```python
import numpy as np

# Cost matrix: cost_matrix[true_bin][predicted_bin]
# Bins: CUT=0, SELECT=1, FLIP=2, REJECT=3
cost_matrix = np.array([
    # Predicted:  CUT  SEL  FLIP  REJ
    [0,    3,    10,    8],   # True: CUT
    [5,    0,     3,    4],   # True: SELECT
    [7,    1,     0,    2],   # True: FLIP
    [1,    1,     1,    0],   # True: REJECT
])

def cost_weighted_accuracy(y_true_bins, y_pred_bins, cost_matrix):
    """
    Lower is better (it's really a cost score, not accuracy).
    Normalized by max possible cost.
    """
    n = len(y_true_bins)
    total_cost = sum(cost_matrix[t][p] for t, p in zip(y_true_bins, y_pred_bins))
    max_possible_cost = sum(max(cost_matrix[t]) for t in y_true_bins)
    
    # Convert to accuracy-like metric (1 = perfect, 0 = worst)
    cost_accuracy = 1.0 - (total_cost / max_possible_cost)
    return cost_accuracy, total_cost / n

# Usage:
# cost_acc, avg_cost = cost_weighted_accuracy(true_bins, pred_bins, cost_matrix)
# print(f"Cost-weighted accuracy: {cost_acc:.3f}, avg cost per stone: {avg_cost:.3f}")
```

### 8.3 Target Milestones

| Phase | Overall Accuracy | Color Accuracy | Clarity Accuracy | Cost-Weighted Acc | Flagged Rate |
|---|---|---|---|---|---|
| **Phase 1** (POC) | ≥70% | ≥75% | ≥75% | ≥0.75 | ≤40% |
| **Phase 2** (Usable) | ≥85% | ≥88% | ≥87% | ≥0.88 | ≤20% |
| **Phase 3** (Production) | ≥92% | ≥94% | ≥93% | ≥0.93 | ≤10% |
| **Stretch** | ≥95% | ≥96% | ≥96% | ≥0.96 | ≤5% |

### 8.4 When Is the Model "Good Enough" for Production?

**Minimum requirements for unsupervised sorting:**

1. **Overall accuracy ≥ 90%** on held-out test set
2. **CUT bin purity ≥ 95%** — fewer than 5% of stones in CUT bin are non-CUT
3. **CUT recall ≥ 85%** — at least 85% of actual CUT-grade stones end up in CUT bin
4. **No class has accuracy < 70%** — model doesn't completely fail on any class
5. **Cost-weighted accuracy ≥ 0.90**
6. **QA reference stone check passes** (≥90% agreement with 50 known reference stones)

**Before reaching these thresholds:**
- Run with human verification: machine sorts → human spot-checks CUT bin (the highest-value output)
- Use the machine output as a "first pass" followed by manual refinement
- This still saves 50–70% of manual sorting time

**Tracking dashboard:**

```
┌─────────────────────────────────────────────────────────────┐
│  MODEL PERFORMANCE TRACKER (v042, 2026-04-10)               │
│                                                             │
│  Overall:  91.2% ████████████████████░░  (target: 92%)     │
│  Color:    93.8% █████████████████████░  (target: 94%)     │
│  Clarity:  91.5% ████████████████████░░  (target: 93%)     │
│  Cost-Wtd: 0.928 ████████████████████░░ (target: 0.93)    │
│  Flagged:  11.3% ███░░░░░░░░░░░░░░░░░░  (target: ≤10%)   │
│                                                             │
│  Status: ⚠️  CLOSE — 2 metrics below target                │
│  Recommendation: Collect 500 more translucent stones        │
│                  (clarity accuracy bottleneck)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Synthetic Data Considerations

### 9.1 Can We Generate Synthetic Training Data?

**Short answer:** Limited value, but some approaches are worth exploring.

| Approach | Feasibility | Value | Effort |
|---|---|---|---|
| **Existing gemstone databases** | Low | Low | Low |
| **3D rendering synthetic stones** | Medium | Medium | High |
| **GAN-generated stones** | Low (need real data first) | Medium | Very High |
| **Transfer learning from related domains** | **High** | **High** | **Low** |
| **Pre-trained backbone (ImageNet)** | **High** | **High** | **Zero** |

### 9.2 Existing Gemstone Image Databases

**Problem:** Very few public datasets exist for rough gemstone melee classification.

| Source | Content | Usefulness |
|---|---|---|
| MinDat.org | Mineral photos, mostly large museum specimens | Very low — wrong scale, wrong context |
| GIA / gemological databases | Cut gemstone images, well-lit studio photos | Low — cut stones look completely different from rough |
| Research papers on mineral classification | Some small datasets (100s of images) | Moderate — may help with transfer learning |
| Alibaba / gem dealer product photos | Rough stone lot photos | Low — inconsistent lighting, wrong resolution, mixed materials |

**Verdict:** No existing database is directly useful. Your own captured data is the only reliable source.

### 9.3 Rendering Synthetic Stones

**Approach:** Use a 3D rendering engine (Blender + Cycles) to create photorealistic rough sapphire images with controlled properties.

**Feasibility assessment:**

| Pro | Con |
|---|---|
| Can generate unlimited images | Rough stone geometry is extremely complex (fracture surfaces, crystal faces) |
| Can control color/clarity exactly | Subsurface scattering for translucency is hard to tune realistically |
| Can simulate both lighting modes | Domain gap: synthetic vs real images may look noticeably different |
| Can balance rare classes trivially | Requires significant Blender expertise |

**Recommendation:** NOT worth the effort for Phase 1–2. By the time you'd have a realistic rendering pipeline, you could have collected 10,000 real images. Consider only if you hit a hard wall on rare classes (e.g., cannot find enough Inky+Transparent stones even after months).

### 9.4 Transfer Learning (Recommended)

**This is the biggest bang for minimal effort.** Start from a model pre-trained on ImageNet.

**Why it works:**
- ImageNet features (edges, textures, colors, shapes) transfer well to gemstone classification
- Even though ImageNet has no gemstones, the low-level features (color gradients, transparency cues, surface textures) are universal
- Fine-tuning a pre-trained model on 500–1,000 gemstone images typically outperforms training from scratch on 5,000+ images

**Implementation (already in the architecture doc):**

```python
import torchvision.models as models
import torch.nn as nn

# Load ImageNet-pretrained MobileNetV3-Small
model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')

# Modify first conv for 6-channel input (see §2.2)
old_conv = model.features[0][0]
new_conv = nn.Conv2d(6, old_conv.out_channels, 
                     kernel_size=old_conv.kernel_size,
                     stride=old_conv.stride, 
                     padding=old_conv.padding, bias=False)
with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    new_conv.weight[:, 3:] = old_conv.weight
model.features[0][0] = new_conv

# Replace classifier with dual heads
num_features = model.classifier[3].in_features
model.classifier = nn.Identity()  # Remove original classifier

class DualHeadClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.color_head = nn.Linear(256, 3)    # Blue, Light, Inky
        self.clarity_head = nn.Linear(256, 3)  # Transparent, Translucent, Opaque
    
    def forward(self, x):
        shared = self.shared(x)
        return self.color_head(shared), self.clarity_head(shared)

dual_head = DualHeadClassifier(num_features)

# Full model: backbone + dual head
class SapphireClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        features = self.backbone(x)
        # MobileNetV3 returns (batch, features) after adaptive pool
        return self.head(features)

classifier = SapphireClassifier(model, dual_head)
```

**Training strategy:**

| Phase | Strategy | Learning Rate |
|---|---|---|
| First 5 epochs | Freeze backbone, train heads only | 0.001 |
| Next 15 epochs | Unfreeze all, fine-tune end-to-end | 0.0001 (backbone), 0.001 (heads) |
| Final 10 epochs | Lower LR, cosine annealing | 0.00001 → 0 |

### 9.5 Related Domain Pre-Training

If ImageNet transfer is insufficient, consider intermediate fine-tuning on:

| Domain | Dataset | Why |
|---|---|---|
| Mineral classification | [Mineral Image Recognition Dataset](https://www.kaggle.com/) (various on Kaggle) | Similar visual features: crystalline textures, transparency, color |
| Color sorting (industrial) | No public datasets, but industrial color sorter companies have proprietary data | Directly related task |
| Medical pathology (glass slides) | Various public datasets | Similar transparency/translucency classification under transmitted light |

**Realistic assessment:** ImageNet pre-training + your own 3,000+ images will be sufficient. Don't over-engineer this. The bottleneck is real gemstone data, not architecture or pre-training.

---

## Appendix: Complete Data Loading Pipeline

```python
"""
Complete PyTorch dataset and dataloader for sapphire melee classification.
Handles dual-image (backlit + toplit) input, JSONL labels, and augmentation.
"""

import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

# --- Label mapping ---
COLOR_CLASSES = ['blue', 'light', 'inky']
CLARITY_CLASSES = ['transparent', 'translucent', 'opaque']
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_CLASSES)}
CLARITY_TO_IDX = {c: i for i, c in enumerate(CLARITY_CLASSES)}


class SapphireDataset(Dataset):
    """Dataset for dual-image sapphire classification."""
    
    def __init__(self, stone_ids, labels_dict, image_dir, transform=None, img_size=128):
        """
        Args:
            stone_ids: List of stone_id strings
            labels_dict: Dict mapping stone_id → {'color': str, 'clarity': str}
            image_dir: Path to processed images (has backlit/ and toplit/ subdirs)
            transform: Albumentations transform (applied to 6-channel combined image)
            img_size: Target image size (square)
        """
        self.stone_ids = stone_ids
        self.labels = labels_dict
        self.image_dir = image_dir
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.stone_ids)
    
    def __getitem__(self, idx):
        sid = self.stone_ids[idx]
        
        # Load both images
        backlit_path = os.path.join(self.image_dir, 'backlit', f'{sid}.png')
        toplit_path = os.path.join(self.image_dir, 'toplit', f'{sid}.png')
        
        backlit = cv2.imread(backlit_path)
        toplit = cv2.imread(toplit_path)
        
        if backlit is None or toplit is None:
            raise FileNotFoundError(f"Missing image for {sid}")
        
        # Convert BGR → RGB
        backlit = cv2.cvtColor(backlit, cv2.COLOR_BGR2RGB)
        toplit = cv2.cvtColor(toplit, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        backlit = cv2.resize(backlit, (self.img_size, self.img_size))
        toplit = cv2.resize(toplit, (self.img_size, self.img_size))
        
        # Stack to 6-channel H×W×6
        combined = np.concatenate([backlit, toplit], axis=2)
        
        # Apply augmentation
        if self.transform:
            # Albumentations works with numpy arrays
            transformed = self.transform(image=combined)
            combined = transformed['image']  # Now a tensor
        else:
            combined = torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0
        
        # Labels
        label = self.labels[sid]
        color_idx = COLOR_TO_IDX[label['color']]
        clarity_idx = CLARITY_TO_IDX[label['clarity']]
        
        return combined, color_idx, clarity_idx


def load_labels(labels_file):
    """Load JSONL labels into a dict."""
    labels = {}
    with open(labels_file) as f:
        for line in f:
            rec = json.loads(line)
            labels[rec['stone_id']] = {
                'color': rec['color'],
                'clarity': rec['clarity'],
            }
    return labels


def load_split(split_file):
    """Load stone_ids from a split file (one ID per line)."""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def get_class_weights(stone_ids, labels_dict):
    """Compute sample weights for WeightedRandomSampler (handles class imbalance)."""
    # Combined class = color_clarity
    class_counts = Counter()
    for sid in stone_ids:
        l = labels_dict[sid]
        class_counts[f"{l['color']}_{l['clarity']}"] += 1
    
    # Weight = 1 / count (inverse frequency)
    total = len(stone_ids)
    n_classes = len(class_counts)
    class_weights = {c: total / (n_classes * count) for c, count in class_counts.items()}
    
    # Per-sample weight
    sample_weights = []
    for sid in stone_ids:
        l = labels_dict[sid]
        c = f"{l['color']}_{l['clarity']}"
        sample_weights.append(class_weights[c])
    
    return torch.DoubleTensor(sample_weights)


def create_dataloaders(data_dir, labels_file, splits_dir, batch_size=32, img_size=128):
    """Create train/val/test dataloaders with augmentation and class balancing."""
    
    labels = load_labels(labels_file)
    
    train_ids = load_split(os.path.join(splits_dir, 'train.txt'))
    val_ids = load_split(os.path.join(splits_dir, 'val.txt'))
    test_ids = load_split(os.path.join(splits_dir, 'test.txt'))
    
    # Augmentation pipelines
    train_transform = A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=180,
                           border_mode=0, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.CoarseDropout(max_holes=3, max_height=8, max_width=8, 
                        min_holes=1, min_height=2, min_width=2, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406] * 2, std=[0.229, 0.224, 0.225] * 2),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406] * 2, std=[0.229, 0.224, 0.225] * 2),
        ToTensorV2(),
    ])
    
    # Datasets
    train_ds = SapphireDataset(train_ids, labels, os.path.join(data_dir, 'images/processed'),
                                train_transform, img_size)
    val_ds = SapphireDataset(val_ids, labels, os.path.join(data_dir, 'images/processed'),
                              val_transform, img_size)
    test_ds = SapphireDataset(test_ids, labels, os.path.join(data_dir, 'images/processed'),
                               val_transform, img_size)
    
    # Weighted sampler for class imbalance
    sample_weights = get_class_weights(train_ids, labels)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ids), replacement=True)
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# --- Usage ---
if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir='data/',
        labels_file='data/labels.jsonl',
        splits_dir='data/splits/v001/',
        batch_size=32,
        img_size=128,
    )
    
    # Test one batch
    for images, color_labels, clarity_labels in train_loader:
        print(f"Batch: images={images.shape}, colors={color_labels.shape}, clarity={clarity_labels.shape}")
        # images: (32, 6, 128, 128) — 6 channels, 128×128
        # color_labels: (32,) — indices 0-2
        # clarity_labels: (32,) — indices 0-2
        break
```

---

*This document is a living reference guide. Update it as the data collection process evolves and lessons are learned.*
