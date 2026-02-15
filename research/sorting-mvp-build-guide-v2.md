# MVP Build Guide v2: Rough Gemstone Melee Sorting Station
## "Classify & Display" — Validate Vision/AI in 7 Days

**Date:** 2026-02-15  
**Version:** 2.0 (Revised with competitive research findings)  
**Status:** Actionable Build Guide  
**Location:** Shenzhen (华强北 + 1688/淘宝)  
**Budget:** < ¥2,000 ($280) for camera station  
**Goal:** Get real stone images flowing through AI classification within 1 week

---

## Competitive Landscape Summary

Before building, here's where this project fits in the market. There is **no existing product** that combines low-cost hardware + colored gemstone AI + multi-category sorting + melee size range + portability.

### What Exists

| Category | Examples | Why It Doesn't Solve Our Problem |
|---|---|---|
| **Industrial ore sorters** | Binder+Co CLARITY ($200K–$1M), TOMRA ($100K–$500K) | Too big, too expensive, designed for >10mm ore recovery |
| **Diamond grading systems** | Sarine ($50K–$200K), OGI MeleeScope ($5K–$15K) | Diamond-only, wrong stone type |
| **AI software only** | Gemtelligence (open source!), Porolis GemLUX (commercial) | No sorting hardware, require lab instruments |
| **Chinese mini color sorters** | GroTech, Skysorter, TAIHO ($3K–$8K) | Binary sorting only, color-based, no clarity assessment |
| **DIY/hobby sorters** | Arduino + servo projects, IronBeadSorter | Too slow, too basic for production use |

### The Gap We're Filling

Nobody makes a **portable, low-cost, multi-category sorting system for colored gemstone melee**. Every solution is either industrial-scale, diamond-only, software-only, or binary-sort-only. This is a real market opportunity.

### Key Learnings From Competitive Research

1. **Classical CV can beat CNNs** — Academic research (Chow & Reyes-Aldasoro, 2022) showed Random Forest with HSV/CIELAB color histograms outperformed ResNet for gemstone classification with limited data
2. **UV fluorescence is cheap and powerful** — Binder+Co uses UV + visible light sensor fusion for ruby/sapphire detection. 365nm UV LEDs cost <¥30
3. **Dark chamber is non-negotiable** — Every successful system uses enclosed, controlled lighting. Ambient light destroys accuracy.
4. **Transmitted light (backlight) is confirmed critical** — Multiple commercial systems validate our backlight approach for clarity assessment
5. **Chinese mini sorter platforms exist** — ¥3,000–8,000 machines with cameras, air jets, and chutes are available and potentially adaptable
6. **Open source gemstone code exists** — Gemtelligence (sapphire classification) and hybchow/gems (Random Forest approach) are on GitHub

---

## Philosophy

**NO mechanical sorting.** No vibratory feeder, no air jets, no Jetson. Human sorts by hand based on screen output. We are validating ONE thing: **can a camera + AI reliably classify rough sapphire melee by color and clarity?**

If the answer is yes → build the full system from the [architecture doc](./sorting-system-architecture.md).  
If the answer is no → we saved 13 weeks and ¥6,500.

```
MVP System:
┌──────────┐    ┌────────┐    ┌──────────┐    ┌─────────┐    ┌───────────┐
│  Stones  │───▶│ Camera │───▶│  Python  │───▶│ Screen  │───▶│  Human    │
│ on slide │    │+ Macro │    │ classify │    │ result  │    │  sorts    │
└──────────┘    └────────┘    └──────────┘    └─────────┘    └───────────┘
```

---

## Buy vs Build Decision

Before committing to a full custom build, consider adapting an existing Chinese mini color sorter.

### Option A: Fully Custom Build

**What:** Design and build everything from scratch per the [architecture doc](./sorting-system-architecture.md).

| Pros | Cons |
|---|---|
| Complete control over hardware + software | 14-week build timeline |
| Optimized for gemstone melee specifically | Higher risk — many unknowns |
| Multi-category sorting (4+ bins) | Requires mechanical engineering skills |
| Can integrate UV, dual-lighting, etc. | Camera, lighting, air jets all need calibration |
| Lower BOM cost (~¥8,500 / $1,200) | Software development from scratch |

### Option B: Buy Mini Color Sorter + Customize Software

**What:** Purchase a Chinese mini color sorter (¥3,000–8,000 / $400–$1,100) and replace/augment the software with custom AI.

| Pros | Cons |
|---|---|
| Proven mechanical platform (chute, air jets, feeder) | Binary sorting only (accept/reject) — need to modify for multi-bin |
| Much faster to get running (days, not months) | Camera may not have sufficient resolution for melee clarity |
| Tested at rice/grain scale (2-10mm, overlaps with melee) | Designed for uniform particles, not irregular rough gems |
| Spare parts readily available | Transparent/translucent stones may confuse standard optics |
| Chinese technical support available | Proprietary control boards — may need to reverse-engineer or replace |
| Compressed air system included | No transmitted light (backlight) — would need to add |

**Manufacturers to investigate:**
- **GroTech** (grotechcolorsorter.com) — KD32 mini sorter
- **Skysorter** (skysorter.com) — Mini RGB sorter
- **TAIHO** (chinacolorsort.com) — Zhiling Series
- **Wenyao** (wenyaocolorsorter.com) — explicitly lists gemstone sorting as an application

### Option C: Hybrid (Recommended Path)

**What:** Run the MVP (this guide) to validate the vision/AI approach first. In parallel, order a cheap mini color sorter to evaluate the mechanical platform. If the AI works, decide whether to:
- (a) gut the mini sorter and replace its brain with your custom AI, or
- (b) build fully custom using the mini sorter as a reference for mechanical design

**This guide focuses on validating the AI — which is required regardless of which hardware path you choose.**

### When to Buy vs Build

| Situation | Recommendation |
|---|---|
| Need sorting ASAP, accuracy less critical | Buy mini sorter, use as-is for basic color rejection |
| Need multi-category sorting (CUT/SELECT/FLIP/REJECT) | Custom build (mini sorters only do binary) |
| Budget < ¥5,000 | Buy mini sorter |
| Budget ¥8,000–15,000 | Hybrid — buy mini sorter + customize |
| Want portable system | Custom build (mini sorters are ~60×80cm) |
| Processing >1 tonne/hour | Buy industrial sorter |

---

## 1. MVP Hardware — Buy Tomorrow in Shenzhen

### Shopping List

| # | Item | Spec | Where to Buy | Search Terms | Price (¥) |
|---|---|---|---|---|---|
| 1 | USB microscope camera | 2MP+, manual focus, C-mount or built-in lens | 华强北 赛格电子市场 3F | `USB显微镜 工业相机 200万像素` | 150–300 |
| 2 | **OR** Hikvision USB industrial camera | MV-CS050-10UC 5MP (ideal but pricier) | 淘宝/1688 (2-3 day delivery) | `海康机器人 MV-CS050 工业相机 USB3` | 800–1,200 |
| 3 | C-mount macro lens (if using industrial cam) | 1:1 magnification, 50mm WD | 华强北 or 1688 | `工业微距镜头 C口 1倍 放大` | 300–500 |
| 4 | LED backlight panel | 50×50mm or larger, white, USB-powered OK | 华强北 赛格 | `LED背光板 白色 USB供电 50mm` | 50–100 |
| 5 | LED ring light | 40-60mm ID, white, dimmable, USB OK | 华强北 or 淘宝 | `LED环形灯 显微镜 可调亮度 USB` | 50–150 |
| 6 | **UV LED strip/module (365nm)** | UV-A, 365nm, for fluorescence detection | 淘宝 | `365nm UV LED灯珠 紫外线 模组` | 20–50 |
| 7 | Acrylic V-groove slide | Clear acrylic strip, ~200mm × 20mm × 5mm | 华强北 周边 亚克力加工店 | `透明亚克力条 定制 V槽` | 20–50 |
| 8 | **Black cardboard/foamcore for dark chamber** | Light-blocking enclosure around imaging area | Any stationery shop | `黑色卡纸 A3` or `黑色泡沫板` | 15–30 |
| 9 | Black cloth/card | Light-blocking background | Any stationery shop | `黑色绒布 遮光` | 10 |
| 10 | Small adjustable stand | Camera boom arm or lab stand | 华强北 or 淘宝 | `万向支架 相机 显微镜支架` | 50–150 |
| 11 | USB hub (powered) | For camera + lights | 华强北 | `USB集线器 带供电` | 30–50 |

### Two Budget Paths

**Path A: Ultra-cheap (¥400–700)** — USB microscope + USB ring light + USB backlight + UV LED + dark chamber + stand  
Good enough to validate the concept. Image quality may limit final accuracy but proves the workflow.

**Path B: Production-ready camera (¥1,500–2,000)** — Hikvision industrial camera + C-mount macro + proper lights + UV LED + dark chamber  
Same camera you'll use in the full system. Data collected now becomes training data for production.

**Recommendation: Path B** if the Hikvision camera can arrive within 2 days (order on 1688 today, many Shenzhen sellers have next-day delivery). Otherwise start Path A tomorrow and upgrade later.

### What You Already Have
- Laptop with NVIDIA GPU (for training later) — or any laptop for capture
- Rough sapphire melee stones (at least a few hundred across color/clarity grades)
- Sorting trays with compartments

### DIY Dark Chamber

**Critical learning from commercial systems:** Every successful sorting machine uses an enclosed, light-controlled chamber. Ambient light variation is the #1 failure mode for DIY vision projects.

**Quick build (30 minutes):**

```
Dark chamber (cardboard box or foamcore):

    ┌─────────────────────────────┐
    │  Black interior (all sides) │
    │                             │
    │      Camera hole (top)      │
    │         ┌───┐               │
    │         │ ◯ │ ← Camera      │
    │         └─┬─┘               │
    │           │                  │
    │    ╔═════╧═════╗            │
    │    ║ Ring Light ║           │
    │    ╚═════╤═════╝            │
    │          │                   │
    │    ──────┼──────── Acrylic  │
    │          ● stone             │
    │    ──────┼────────          │
    │    ┌─────┴─────┐            │
    │    │ Backlight  │           │
    │    └───────────┘            │
    │                             │
    │  ┌─ UV LEDs (side-mounted) │
    │  └──────────────────────── │
    │                             │
    │   Stone entry slot (front) │
    │   ═══════════════════════  │
    └─────────────────────────────┘
```

Materials: Black foamcore board (¥15-30), hot glue gun, box cutter. Cut a hole in top for camera, a slot in front for stone access. Paint or line interior with matte black. This alone will dramatically improve classification consistency.

### DIY V-Groove Slide

Don't overthink this. Two options:

**Option 1: Acrylic strip.** Buy a 200mm × 20mm × 5mm clear acrylic strip. Score a V-groove down the center with a rotary tool or have a 华强北 acrylic shop CNC it (¥20, 10 minutes). Stones sit in the groove, backlight shines through from below.

**Option 2: Skip the groove entirely.** Place stones one at a time on a clear acrylic sheet over the backlight. For MVP, speed doesn't matter — accuracy does.

### Total MVP Budget

| Path | Total |
|---|---|
| Path A (USB microscope) | ¥400–700 |
| Path B (industrial camera) | ¥1,500–2,000 |

---

## 2. MVP Software Stack — Dual Approach

### NEW: Why We Start with Classical CV, Not Just CNN

Academic research on gemstone classification (Chow & Reyes-Aldasoro, 2022, published in MDPI Minerals) found that **Random Forest with handcrafted color features outperformed ResNet-18 and ResNet-50** on a 68-category gemstone dataset. This is because:

1. Color histograms in perceptually uniform color spaces (HSV, CIELAB) directly capture the classification signal
2. With limited training data (<5000 images), classical ML avoids overfitting
3. Feature extraction is interpretable — you can see WHY a stone was classified a certain way
4. Runs on any hardware — no GPU needed

**Our approach: Phase 1 (Classical CV) → Phase 2 (CNN) → Phase 3 (Ensemble)**

### Open Source Resources to Leverage

**1. Gemtelligence** (github.com/TommasoBendinelli/Gemtelligence)
- From Gübelin Gem Lab + CSEM, published in Nature Communications Engineering (2024)
- Deep learning for blue sapphire classification (origin determination + treatment detection)
- Uses multi-modal data (spectral + tabular)
- **How to leverage:** Their neural network architecture (HoL-Net) handles heterogeneous data. While we won't have spectrometer data, their approach of combining image features with tabular data (stone size, transmittance ratio) is directly applicable. Study their `src/` directory for the attention-based architecture.
- **Note:** Their code is designed for spectral input, not camera images. Don't use it directly — use it as architectural inspiration for multi-modal fusion.

**2. hybchow/gems** (github.com/hybchow/gems)
- Published in MDPI Minerals, 2022
- Classical CV pipeline: RGB/HSV/CIELAB color histograms + LBP texture + Haralick features + GLCM → Random Forest
- **How to leverage:** This is our Phase 1 approach. Their feature extraction code can be adapted directly. Key insight: they tested multiple classifiers and Random Forest won. Copy their feature extraction pipeline.

**3. loremendez/Gemstones** (GitHub)
- Deep CNN with residual blocks for gemstone classification (TensorFlow)
- Good reference for CNN architecture, but we'll use PyTorch MobileNetV3 instead

### Dependencies

```bash
pip install opencv-python numpy scikit-learn pillow matplotlib
# Phase 1: Classical CV (no GPU needed!)

# Phase 2 (later, when you have 500+ images):
pip install torch torchvision onnxruntime
# For GUI:
pip install tkinter  # Usually included with Python
# OR for web UI:
pip install flask
```

If using Hikvision camera, also install the MVS SDK from [hikrobotics.com](https://www.hikrobotics.com/en/machinevision/service/download).

### Architecture: Phase 1 (Classical CV) + Phase 2 (CNN)

```
mvp_sorter/
├── capture.py              # Camera capture + stone detection
├── features.py             # NEW: Classical feature extraction (HSV/CIELAB histograms)
├── classify_classical.py   # NEW: Random Forest classifier (Phase 1)
├── classify_cnn.py         # CNN classifier (Phase 2, was classify.py)
├── train_classical.py      # NEW: Train Random Forest
├── train_cnn.py            # Train CNN (was train.py)
├── gui.py                  # Live classification GUI
├── collect_data.py         # Training data collection mode
├── config.py               # Settings
├── images/                 # Captured training images
│   ├── blue_transparent/
│   ├── blue_translucent/
│   ├── blue_opaque/
│   ├── light_transparent/
│   ├── light_translucent/
│   ├── light_opaque/
│   ├── inky_transparent/
│   ├── inky_translucent/
│   └── inky_opaque/
└── models/                 # Saved model files
    ├── rf_color.pkl        # Random Forest color model
    ├── rf_clarity.pkl      # Random Forest clarity model
    └── best_model.onnx     # CNN model (Phase 2)
```

### features.py — Classical Feature Extraction (NEW)

```python
"""
Feature extraction for classical CV gemstone classification.
Based on Chow & Reyes-Aldasoro (2022) approach that outperformed ResNet.

Extracts: HSV histograms, CIELAB histograms, basic texture features.
"""
import cv2
import numpy as np
from typing import Dict, List

def extract_color_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract color histogram features in multiple color spaces.
    
    Returns: 1D feature vector (~200 dimensions)
    """
    features = []
    
    # --- HSV color space ---
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Hue histogram (180 bins → downsample to 36)
    h_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180])
    h_hist = h_hist.flatten() / (h_hist.sum() + 1e-8)
    features.append(h_hist)
    
    # Saturation histogram (256 → 32 bins)
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    s_hist = s_hist.flatten() / (s_hist.sum() + 1e-8)
    features.append(s_hist)
    
    # Value histogram (256 → 32 bins)
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    v_hist = v_hist.flatten() / (v_hist.sum() + 1e-8)
    features.append(v_hist)
    
    # --- CIELAB color space ---
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    # L* histogram (lightness — key for clarity!)
    l_hist = cv2.calcHist([lab], [0], None, [32], [0, 256])
    l_hist = l_hist.flatten() / (l_hist.sum() + 1e-8)
    features.append(l_hist)
    
    # a* histogram (green-red axis)
    a_hist = cv2.calcHist([lab], [1], None, [32], [0, 256])
    a_hist = a_hist.flatten() / (a_hist.sum() + 1e-8)
    features.append(a_hist)
    
    # b* histogram (blue-yellow axis — key for sapphire color!)
    b_hist = cv2.calcHist([lab], [2], None, [32], [0, 256])
    b_hist = b_hist.flatten() / (b_hist.sum() + 1e-8)
    features.append(b_hist)
    
    # --- Statistical features ---
    for channel_img in [hsv, lab]:
        for c in range(3):
            ch = channel_img[:, :, c].astype(np.float32)
            features.append(np.array([
                ch.mean(),
                ch.std(),
                np.median(ch),
                ch.min(),
                ch.max(),
            ]))
    
    return np.concatenate(features)


def extract_clarity_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract features relevant to clarity (transparency/translucency/opacity).
    Best used with BACKLIT images.
    
    Returns: 1D feature vector (~50 dimensions)
    """
    features = []
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Transmittance statistics (how much light passes through)
    features.append(np.array([
        gray.mean() / 255.0,       # Mean transmittance
        gray.std() / 255.0,        # Transmittance uniformity
        np.median(gray) / 255.0,
        (gray > 128).sum() / gray.size,  # Fraction of bright pixels
        (gray > 200).sum() / gray.size,  # Fraction of very bright pixels
        (gray < 50).sum() / gray.size,   # Fraction of dark pixels
    ]))
    
    # Histogram of transmittance
    t_hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [32], [0, 256])
    t_hist = t_hist.flatten() / (t_hist.sum() + 1e-8)
    features.append(t_hist)
    
    # Edge density (transparent stones have fewer internal edges)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    features.append(np.array([
        edges.sum() / (edges.size * 255.0),  # Edge density
    ]))
    
    # Texture: variance in local windows
    kernel_size = 7
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    local_var = cv2.blur((gray - local_mean) ** 2, (kernel_size, kernel_size))
    features.append(np.array([
        local_var.mean(),
        local_var.std(),
        np.median(local_var),
    ]))
    
    return np.concatenate(features)


def extract_all_features(image_bgr: np.ndarray) -> np.ndarray:
    """Extract combined color + clarity features from a single image."""
    color_feat = extract_color_features(image_bgr)
    clarity_feat = extract_clarity_features(image_bgr)
    return np.concatenate([color_feat, clarity_feat])
```

### classify_classical.py — Random Forest Classifier (NEW — Phase 1)

```python
"""
Classical ML classification using Random Forest.
Phase 1 approach — works with as few as 100 training images.
Based on findings from Chow & Reyes-Aldasoro (2022).
"""
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from features import extract_color_features, extract_clarity_features

COLOR_CLASSES = ['blue', 'inky', 'light']
CLARITY_CLASSES = ['opaque', 'translucent', 'transparent']

DECISION_MAP = {
    ('blue', 'transparent'): 'CUT',
    ('blue', 'translucent'): 'FLIP',
    ('blue', 'opaque'): 'FLIP',
    ('light', 'transparent'): 'SELECT',
    ('light', 'translucent'): 'FLIP',
    ('light', 'opaque'): 'FLIP',
    ('inky', 'transparent'): 'SELECT',
    ('inky', 'translucent'): 'SELECT',
    ('inky', 'opaque'): 'REJECT',
}


class ClassicalClassifier:
    def __init__(self, color_model_path=None, clarity_model_path=None):
        if color_model_path:
            with open(color_model_path, 'rb') as f:
                self.color_model = pickle.load(f)
            with open(clarity_model_path, 'rb') as f:
                self.clarity_model = pickle.load(f)
        else:
            self.color_model = RandomForestClassifier(
                n_estimators=200, max_depth=20, 
                min_samples_split=5, random_state=42, n_jobs=-1
            )
            self.clarity_model = RandomForestClassifier(
                n_estimators=200, max_depth=20,
                min_samples_split=5, random_state=42, n_jobs=-1
            )
    
    def predict(self, image_bgr):
        """
        Classify a stone image.
        Returns: dict with color, clarity, confidences
        """
        color_feat = extract_color_features(image_bgr).reshape(1, -1)
        clarity_feat = extract_clarity_features(image_bgr).reshape(1, -1)
        
        color_probs = self.color_model.predict_proba(color_feat)[0]
        clarity_probs = self.clarity_model.predict_proba(clarity_feat)[0]
        
        color_idx = np.argmax(color_probs)
        clarity_idx = np.argmax(clarity_probs)
        
        return {
            "color": COLOR_CLASSES[color_idx],
            "color_conf": float(color_probs[color_idx]),
            "clarity": CLARITY_CLASSES[clarity_idx],
            "clarity_conf": float(clarity_probs[clarity_idx]),
            "color_probs": {c: float(p) for c, p in zip(COLOR_CLASSES, color_probs)},
            "clarity_probs": {c: float(p) for c, p in zip(CLARITY_CLASSES, clarity_probs)},
        }
    
    def save(self, color_path="models/rf_color.pkl", clarity_path="models/rf_clarity.pkl"):
        with open(color_path, 'wb') as f:
            pickle.dump(self.color_model, f)
        with open(clarity_path, 'wb') as f:
            pickle.dump(self.clarity_model, f)
```

### train_classical.py — Train Random Forest (NEW)

```python
"""
Train Random Forest classifiers for color and clarity.
Phase 1: Works with as few as 100 images. No GPU needed.
Training time: seconds, not minutes.
"""
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from features import extract_color_features, extract_clarity_features
from classify_classical import ClassicalClassifier, COLOR_CLASSES, CLARITY_CLASSES

def load_dataset(root_dir="images"):
    """Load images and extract features."""
    color_features = []
    clarity_features = []
    color_labels = []
    clarity_labels = []
    
    color_to_idx = {c: i for i, c in enumerate(COLOR_CLASSES)}
    clarity_to_idx = {c: i for i, c in enumerate(CLARITY_CLASSES)}
    
    for class_dir in os.listdir(root_dir):
        parts = class_dir.split('_')
        if len(parts) != 2:
            continue
        color, clarity = parts
        if color not in color_to_idx or clarity not in clarity_to_idx:
            continue
        
        class_path = os.path.join(root_dir, class_dir)
        for img_name in os.listdir(class_path):
            if img_name.endswith('_raw.jpg'):
                continue
            if not img_name.endswith(('.jpg', '.png')):
                continue
            
            img = cv2.imread(os.path.join(class_path, img_name))
            if img is None:
                continue
            
            # Resize to consistent size
            img = cv2.resize(img, (128, 128))
            
            color_features.append(extract_color_features(img))
            clarity_features.append(extract_clarity_features(img))
            color_labels.append(color_to_idx[color])
            clarity_labels.append(clarity_to_idx[clarity])
    
    return (np.array(color_features), np.array(clarity_features),
            np.array(color_labels), np.array(clarity_labels))


def train():
    print("Loading dataset and extracting features...")
    color_X, clarity_X, color_y, clarity_y = load_dataset()
    print(f"Loaded {len(color_y)} images")
    
    # Print class distribution
    for i, name in enumerate(COLOR_CLASSES):
        print(f"  Color {name}: {(color_y == i).sum()}")
    for i, name in enumerate(CLARITY_CLASSES):
        print(f"  Clarity {name}: {(clarity_y == i).sum()}")
    
    # Create classifier
    clf = ClassicalClassifier()
    
    # Cross-validation
    print("\nCross-validation (5-fold):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    color_scores = cross_val_score(clf.color_model, color_X, color_y, cv=skf, scoring='accuracy')
    print(f"  Color accuracy: {color_scores.mean():.1%} (±{color_scores.std():.1%})")
    
    clarity_scores = cross_val_score(clf.clarity_model, clarity_X, clarity_y, cv=skf, scoring='accuracy')
    print(f"  Clarity accuracy: {clarity_scores.mean():.1%} (±{clarity_scores.std():.1%})")
    
    # Train on full dataset
    print("\nTraining final models on full dataset...")
    clf.color_model.fit(color_X, color_y)
    clf.clarity_model.fit(clarity_X, clarity_y)
    
    # Feature importance
    print("\nTop 10 color features by importance:")
    color_imp = clf.color_model.feature_importances_
    top_idx = np.argsort(color_imp)[-10:][::-1]
    for idx in top_idx:
        print(f"  Feature {idx}: {color_imp[idx]:.4f}")
    
    # Save
    os.makedirs("models", exist_ok=True)
    clf.save()
    print("\nModels saved to models/rf_color.pkl and models/rf_clarity.pkl")
    print(f"Training time: basically instant (Random Forest on {len(color_y)} samples)")
    
    return clf


if __name__ == "__main__":
    train()
```

### capture.py — Camera Interface

```python
"""Camera capture module. Supports USB webcam or Hikvision industrial camera."""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

class CameraCapture:
    def __init__(self, camera_id=0, width=1920, height=1080):
        """
        camera_id: 0 for default USB camera, or device index
        For Hikvision, use HikvisionCapture class instead.
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Disable auto-exposure and auto-white-balance for consistency
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=manual on some cameras
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        
    def grab_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        return frame
    
    def release(self):
        self.cap.release()


def detect_stone(frame, bg_frame, min_area=500, max_area=500000):
    """
    Detect stone in frame by background subtraction.
    Returns: list of (x, y, w, h) bounding boxes, or empty list.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray, bg_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stones = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            margin = max(w, h) // 4
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)
            stones.append((x, y, w, h))
    
    return stones


def extract_roi(frame, bbox, output_size=128):
    """Extract and resize stone ROI to fixed size."""
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    size = max(w, h)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    dy = (size - h) // 2
    dx = (size - w) // 2
    square[dy:dy+h, dx:dx+w] = roi
    resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized
```

### collect_data.py — Training Data Collection

```python
"""
Training data collection tool.
Place stones one at a time. Press key to classify and save.

Controls:
  1-3: Color (1=Blue, 2=Light, 3=Inky)
  q/w/e: Clarity (q=Transparent, w=Translucent, e=Opaque)
  SPACE: Capture background frame
  ESC: Quit
"""
import cv2
import os
import json
from datetime import datetime
from capture import CameraCapture, detect_stone, extract_roi

COLOR_KEYS = {'1': 'blue', '2': 'light', '3': 'inky'}
CLARITY_KEYS = {'q': 'transparent', 'w': 'translucent', 'e': 'opaque'}
CLASSES = [f"{c}_{cl}" for c in ['blue', 'light', 'inky'] 
           for cl in ['transparent', 'translucent', 'opaque']]

def main():
    cam = CameraCapture(camera_id=0)
    base_dir = "images"
    
    for cls in CLASSES:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)
    
    print("Remove all stones from view, then press SPACE to capture background")
    bg_frame = None
    current_color = None
    current_clarity = None
    
    counts = {}
    for cls in CLASSES:
        counts[cls] = len(os.listdir(os.path.join(base_dir, cls)))
    
    metadata_log = []
    
    while True:
        frame = cam.grab_frame()
        display = frame.copy()
        
        color_str = current_color or "?"
        clarity_str = current_clarity or "?"
        cv2.putText(display, f"Color: {color_str} (1=Blue 2=Light 3=Inky)", 
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Clarity: {clarity_str} (q=Trans w=Transluc e=Opaque)", 
                     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_pos = 100
        for cls in CLASSES:
            cv2.putText(display, f"{cls}: {counts[cls]}", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 20
        
        if bg_frame is not None:
            stones = detect_stone(frame, bg_frame)
            for (x, y, w, h) in stones:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if current_color and current_clarity and len(stones) == 1:
                cv2.putText(display, "PRESS ENTER to save | R to reset selection", 
                           (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
        
        cv2.imshow("Data Collection", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        elif key == ord(' '):
            bg_frame = frame.copy()
            print("Background captured!")
        elif chr(key) in COLOR_KEYS if key < 128 else False:
            current_color = COLOR_KEYS[chr(key)]
            print(f"Color: {current_color}")
        elif chr(key) in CLARITY_KEYS if key < 128 else False:
            current_clarity = CLARITY_KEYS[chr(key)]
            print(f"Clarity: {current_clarity}")
        elif key == ord('r'):
            current_color = None
            current_clarity = None
        elif key == 13 and current_color and current_clarity:
            if bg_frame is not None and len(stones) == 1:
                cls_name = f"{current_color}_{current_clarity}"
                idx = counts[cls_name]
                
                roi = extract_roi(frame, stones[0], output_size=224)
                filename = f"{cls_name}_{idx:04d}.jpg"
                filepath = os.path.join(base_dir, cls_name, filename)
                cv2.imwrite(filepath, roi)
                
                x, y, w, h = stones[0]
                raw_crop = frame[y:y+h, x:x+w]
                raw_path = os.path.join(base_dir, cls_name, f"{cls_name}_{idx:04d}_raw.jpg")
                cv2.imwrite(raw_path, raw_crop)
                
                meta = {
                    "filename": filename,
                    "color": current_color,
                    "clarity": current_clarity,
                    "timestamp": datetime.now().isoformat(),
                    "bbox": stones[0],
                }
                metadata_log.append(meta)
                
                counts[cls_name] += 1
                print(f"Saved: {filepath} (total {cls_name}: {counts[cls_name]})")
    
    with open(os.path.join(base_dir, "metadata.json"), "w") as f:
        json.dump(metadata_log, f, indent=2, default=str)
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"\nTotal images collected: {sum(counts.values())}")
    for cls, count in counts.items():
        print(f"  {cls}: {count}")

if __name__ == "__main__":
    main()
```

### gui.py — Live Classification GUI

```python
"""
Live classification GUI — supports both Classical CV and CNN classifiers.
Shows camera feed, detects stones, classifies, displays result.
Human sorts by hand based on screen output.

Controls:
  SPACE: Capture/update background
  S: Toggle semi-auto mode (continuous classification)
  M: Toggle model (Classical ↔ CNN)
  U: Toggle UV light capture
  ESC: Quit
"""
import cv2
import numpy as np
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from capture import CameraCapture, detect_stone, extract_roi

# Try to load both classifiers
try:
    from classify_classical import ClassicalClassifier, DECISION_MAP
    classical_available = True
except ImportError:
    classical_available = False

try:
    from classify_cnn import StoneClassifier
    cnn_available = True
except ImportError:
    cnn_available = False

DECISION_COLORS = {
    "CUT": (0, 255, 0),
    "SELECT": (255, 255, 0),
    "FLIP": (0, 165, 255),
    "REJECT": (0, 0, 255),
}

def main():
    cam = CameraCapture(camera_id=0)
    
    # Load available classifiers
    classifier = None
    model_name = "none"
    
    if classical_available:
        try:
            classifier = ClassicalClassifier("models/rf_color.pkl", "models/rf_clarity.pkl")
            model_name = "Classical (RF)"
            print("Loaded Classical CV (Random Forest) classifier")
        except FileNotFoundError:
            print("No classical model found")
    
    if classifier is None and cnn_available:
        try:
            classifier = StoneClassifier("models/best_model.onnx")
            model_name = "CNN (MobileNetV3)"
            print("Loaded CNN classifier")
        except FileNotFoundError:
            print("No CNN model found")
    
    if classifier is None:
        print("ERROR: No trained model found. Run train_classical.py or train_cnn.py first.")
        return
    
    bg_frame = None
    semi_auto = False
    last_classify_time = 0
    classify_cooldown = 0.5
    
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"sort_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "model", "color", "color_conf", "clarity", 
                         "clarity_conf", "decision", "overall_conf"])
    
    results_history = []
    stones_per_min = 0
    minute_start = time.time()
    minute_count = 0
    
    while True:
        frame = cam.grab_frame()
        display = frame.copy()
        
        mode_str = "SEMI-AUTO" if semi_auto else "MANUAL (press S)"
        bg_str = "BG: ✓" if bg_frame is not None else "BG: ✗ (SPACE)"
        cv2.putText(display, f"{mode_str} | {bg_str} | Model: {model_name} | {stones_per_min:.0f}/min", 
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if bg_frame is not None:
            stones = detect_stone(frame, bg_frame)
            
            for (x, y, w, h) in stones:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            should_classify = False
            if semi_auto and len(stones) >= 1 and (time.time() - last_classify_time) > classify_cooldown:
                should_classify = True
            
            if should_classify or (not semi_auto and len(stones) == 1):
                for (x, y, w, h) in stones[:1]:
                    roi = extract_roi(frame, (x, y, w, h), output_size=128)
                    result = classifier.predict(roi)
                    
                    color = result["color"]
                    clarity = result["clarity"]
                    color_conf = result["color_conf"]
                    clarity_conf = result["clarity_conf"]
                    decision = DECISION_MAP.get((color, clarity), "SELECT")
                    overall_conf = min(color_conf, clarity_conf)
                    
                    dec_color = DECISION_COLORS.get(decision, (255, 255, 255))
                    
                    cv2.putText(display, decision, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, dec_color, 3)
                    cv2.putText(display, f"{color} / {clarity}", (x, y + h + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display, f"Conf: {overall_conf:.0%}", (x, y + h + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                               (0, 255, 0) if overall_conf > 0.75 else (0, 0, 255), 2)
                    
                    if semi_auto:
                        last_classify_time = time.time()
                        minute_count += 1
                        
                        with open(log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([datetime.now().isoformat(), model_name,
                                           color, f"{color_conf:.3f}",
                                           clarity, f"{clarity_conf:.3f}", 
                                           decision, f"{overall_conf:.3f}"])
                        
                        results_history.append({
                            "decision": decision, "color": color, "clarity": clarity,
                            "conf": overall_conf, "time": time.time()
                        })
                        if len(results_history) > 20:
                            results_history.pop(0)
        
        if time.time() - minute_start > 60:
            stones_per_min = minute_count
            minute_count = 0
            minute_start = time.time()
        
        sidebar_x = display.shape[1] - 250
        cv2.rectangle(display, (sidebar_x - 10, 0), (display.shape[1], display.shape[0]), (40, 40, 40), -1)
        cv2.putText(display, "Recent:", (sidebar_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for i, r in enumerate(reversed(results_history[-15:])):
            y_pos = 50 + i * 25
            dec_color = DECISION_COLORS.get(r["decision"], (255, 255, 255))
            cv2.putText(display, f"{r['decision']:6s} {r['color'][:3]}/{r['clarity'][:5]} {r['conf']:.0%}", 
                        (sidebar_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dec_color, 1)
        
        cv2.imshow("Sapphire Sorter MVP", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        elif key == ord(' '):
            bg_frame = frame.copy()
            print("Background updated")
        elif key == ord('s'):
            semi_auto = not semi_auto
            print(f"Semi-auto: {'ON' if semi_auto else 'OFF'}")
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### classify_cnn.py — CNN Model Inference (Phase 2)

```python
"""Stone classification using trained MobileNetV3-Small with dual heads.
Phase 2 classifier — use when you have 500+ training images.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

COLOR_CLASSES = ['blue', 'inky', 'light']
CLARITY_CLASSES = ['opaque', 'translucent', 'transparent']

DECISION_MAP = {
    ('blue', 'transparent'): 'CUT',
    ('blue', 'translucent'): 'FLIP',
    ('blue', 'opaque'): 'FLIP',
    ('light', 'transparent'): 'SELECT',
    ('light', 'translucent'): 'FLIP',
    ('light', 'opaque'): 'FLIP',
    ('inky', 'transparent'): 'SELECT',
    ('inky', 'translucent'): 'SELECT',
    ('inky', 'opaque'): 'REJECT',
}

class DualHeadMobileNet(nn.Module):
    """MobileNetV3-Small with two classification heads for color and clarity."""
    def __init__(self, num_color=3, num_clarity=3):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        self.shared_fc = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.color_head = nn.Linear(256, num_color)
        self.clarity_head = nn.Linear(256, num_clarity)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.shared_fc(x)
        color_logits = self.color_head(x)
        clarity_logits = self.clarity_head(x)
        return color_logits, clarity_logits


class StoneClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        if model_path.endswith('.onnx'):
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
            self.use_onnx = True
        else:
            self.model = DualHeadMobileNet()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            self.use_onnx = False
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def predict(self, image_bgr):
        """Classify a stone image."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb).unsqueeze(0)
        
        if self.use_onnx:
            inputs = {self.session.get_inputs()[0].name: tensor.numpy()}
            color_logits, clarity_logits = self.session.run(None, inputs)
            color_probs = self._softmax(color_logits[0])
            clarity_probs = self._softmax(clarity_logits[0])
        else:
            with torch.no_grad():
                tensor = tensor.to(self.device)
                color_logits, clarity_logits = self.model(tensor)
                color_probs = torch.softmax(color_logits, dim=1)[0].cpu().numpy()
                clarity_probs = torch.softmax(clarity_logits, dim=1)[0].cpu().numpy()
        
        color_idx = np.argmax(color_probs)
        clarity_idx = np.argmax(clarity_probs)
        
        return {
            "color": COLOR_CLASSES[color_idx],
            "color_conf": float(color_probs[color_idx]),
            "clarity": CLARITY_CLASSES[clarity_idx],
            "clarity_conf": float(clarity_probs[clarity_idx]),
            "color_probs": {c: float(p) for c, p in zip(COLOR_CLASSES, color_probs)},
            "clarity_probs": {c: float(p) for c, p in zip(CLARITY_CLASSES, clarity_probs)},
        }
    
    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()
```

---

## 3. Day-by-Day Build Plan (7 Days)

### Day 1: Buy Hardware + Build Dark Chamber

**Morning (华强北):**
1. Go to 赛格电子市场 (SEG Electronics Market), floors 2-4
2. Buy: USB camera or microscope, ring light, adjustable stand, USB hub
3. For backlight: look for "LED灯板" or "LED背光模组" — a small white LED panel. Even a cheap LED tracing pad (临摹台 / `LED拷贝台 A5`) works perfectly as a backlight (¥30-50 on 淘宝)
4. Buy black velvet cloth from a fabric stall nearby
5. **NEW:** Buy 365nm UV LED module (¥20-50, search `365nm UV LED模组`)
6. **NEW:** Buy black foamcore board or thick cardboard (A3 size × 2-3 sheets) for dark chamber

**Afternoon (acrylic shop + dark chamber build):**
1. Find an acrylic fabrication shop near 华强北 (there are many on 振华路)
2. Get a clear acrylic piece: 200mm × 100mm × 5mm
3. Optional: ask them to cut a shallow V-groove (2mm deep, 90° angle)
4. **NEW:** Build dark chamber (see diagram above). 30 minutes with box cutter + hot glue.

**Evening:**
- Order Hikvision camera on 1688/淘宝 if not available locally (arrives Day 2-3)
- Install Python, OpenCV, scikit-learn on laptop (no GPU needed for Phase 1!)
- Test USB camera works with OpenCV

```python
# Quick camera test
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
```

### Day 2: Assemble Camera Station Inside Dark Chamber

**Morning — Physical assembly:**

```
Assembly layout (side view) — INSIDE DARK CHAMBER:
                    
    Camera on stand arm, pointing DOWN
         ┃
    ╔════╋════╗  ← Ring light (around lens)
    ║    ┃    ║
    ║    ▼    ║     Working distance: 50-80mm
    ║         ║
    ║  [stone]║  ← Clear acrylic sheet on stand
    ╚═════════╝
         │
    ┌────┴────┐  ← Backlight (LED panel below acrylic)
    │ LED pad │     Gap: 5-10mm below acrylic
    └─────────┘
    
    UV LEDs mounted on chamber wall, aimed at stone
    (switched separately — optional capture mode)
```

Steps:
1. Mount camera on adjustable stand, lens pointing down
2. Place LED backlight panel flat on desk
3. Raise acrylic sheet 5-10mm above backlight (use small blocks/spacers)
4. Position ring light around or just below the camera lens
5. **NEW:** Mount UV LED module on the side wall of the dark chamber, aimed at the stone position
6. **NEW:** Close dark chamber around everything, leaving only the stone entry slot
7. Place a stone on the acrylic — can you see it clearly on screen?

**Afternoon — Optimize image quality:**

Run the interactive setup tool to tune camera settings:

```python
"""Interactive camera setup tool."""
import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

def nothing(x): pass

cv2.namedWindow("Setup")
cv2.createTrackbar("Exposure", "Setup", 50, 100, nothing)
cv2.createTrackbar("Brightness", "Setup", 50, 100, nothing)

while True:
    ret, frame = cam.read()
    if not ret: break
    
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    crop_size = min(h, w) // 3
    crop = frame[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
    for i in range(256):
        cv2.line(hist_img, (i, 200), (i, 200 - int(hist[i])), (255, 255, 255), 1)
    
    display = cv2.resize(crop, (600, 600))
    cv2.putText(display, f"Resolution: {w}x{h}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Setup", display)
    cv2.imshow("Histogram", hist_img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
```

**What you're looking for:**
- Stone is sharp (in focus). Adjust camera height/focus until stone edges are crisp.
- With backlight ON, ring light OFF: transparent stones glow bright white, opaque stones are dark silhouettes
- With ring light ON, backlight OFF: you can see the stone's COLOR clearly (blue vs light vs dark)
- **NEW:** With UV LED ON, all other lights OFF: check if different clarity grades fluoresce differently
- No harsh reflections or hotspots
- Consistent, even illumination
- **Dark chamber keeps ambient light out** — verify by checking that the image doesn't change when room lights are toggled

### Day 3: Capture 200+ Stone Images

**Goal: Minimum 25 images per class × 9 classes = 225 images.** More is better. Aim for 50+ per class if you have enough stones.

**Procedure:**
1. Run `collect_data.py`
2. Press SPACE to capture background (empty acrylic sheet, lights on)
3. Sort your stones roughly by eye into 9 piles first
4. Work through one pile at a time:
   - Press `1` for Blue, `2` for Light, `3` for Inky
   - Press `q` for Transparent, `w` for Translucent, `e` for Opaque
   - Place stone on acrylic
   - Wait for green bounding box to appear
   - Press ENTER to save
   - Remove stone, place next one
5. Re-capture background every ~50 stones (press SPACE with no stone)

**Tips for speed:**
- Pre-sort stones into class piles. This way you set the class keys once and just place/capture/remove.
- At 3-4 seconds per stone, 225 stones takes ~15 minutes.
- Aim for 500+ total if you have time — accuracy improves significantly.
- **Keep the dark chamber closed** while capturing — only open the stone entry slot.

**Two-shot protocol (recommended if time allows):**

For each stone, capture TWO images:
1. **Backlight ON, ring light OFF** → save as clarity image
2. **Ring light ON, backlight OFF** → save as color image

This gives the model the clearest signal for each classification axis. In production, you might use combined lighting, but for training data, separated lighting gives cleaner labels.

**Optional UV shot (Day 3 evening experiment):**
Capture a third image per stone with UV LED ON, all other lights OFF. Some sapphires fluoresce differently based on origin and treatment — this data may improve classification later. Not required for MVP but costs nothing to capture.

### Day 4: Train Classical Model (Phase 1) — Takes Minutes, Not Hours

**THIS IS THE KEY CHANGE FROM v1:** Instead of jumping straight to CNN training, start with Random Forest.

```bash
# Train classical model — takes ~30 seconds
python train_classical.py
```

**Expected output:**
```
Loading dataset and extracting features...
Loaded 450 images
  Color blue: 175
  Color inky: 125
  Color light: 150
  Clarity opaque: 130
  Clarity translucent: 180
  Clarity transparent: 140

Cross-validation (5-fold):
  Color accuracy: 82.3% (±3.1%)
  Clarity accuracy: 76.8% (±4.2%)

Training final models on full dataset...
Models saved to models/rf_color.pkl and models/rf_clarity.pkl
Training time: basically instant (Random Forest on 450 samples)
```

**Why this is better than CNN for Day 4:**
- No GPU needed — runs on any laptop
- Trains in seconds, not hours
- Works well with 200-500 images (CNN needs 1000+)
- Interpretable — you can see which features matter
- If accuracy is low, you know it's a data/lighting problem, not a model problem

**If accuracy > 75%:** Great start! Test it live on Day 5.  
**If accuracy < 60%:** Lighting problem — go back and fix your dark chamber and backlight setup.

### Day 5: Test Live + Collect More Data

1. Run `gui.py` with the classical model
2. Test on new stones not in the training set
3. Note which classes are confused
4. Collect targeted additional data for weak classes
5. Retrain (takes seconds) and test again

**Key insight:** With classical CV, you can iterate much faster. Train → test → collect more data → retrain → test again, all in one afternoon. With CNN, each training cycle takes 30-60 minutes.

### Day 6: Train CNN (Phase 2) + Compare

If you now have 500+ images from Days 3-5, try the CNN:

```bash
# Install PyTorch if not already done
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Train CNN
python train_cnn.py
```

**Compare Classical vs CNN:**

| Metric | Classical (RF) | CNN (MobileNetV3) |
|---|---|---|
| Training data needed | 200+ | 500+ |
| Training time | Seconds | 10-60 minutes |
| GPU required | No | Recommended |
| Color accuracy (200 images) | 70-80% | 55-70% |
| Color accuracy (500 images) | 78-85% | 75-85% |
| Color accuracy (1000+ images) | 80-88% | 85-93% |
| Clarity accuracy | Usually 5-10% lower than color | Similar |

**Key insight:** Classical CV wins with limited data. CNN wins with abundant data. The crossover point is typically around 500-1000 images per class.

### Day 7: Final Accuracy Test + Decision

Based on Days 5-6 results:

**If Classical accuracy > 80% or CNN accuracy > 85%:** 🎉 Success! Start planning the full mechanical build.

**If accuracy 65-80%:**
- Check confusion matrix — which classes are confused
- Collect 50+ more images for weak classes
- Retrain both models
- Try adjusting lighting — clarity classification improves dramatically with better backlight positioning
- Try UV capture if you haven't — it may help distinguish clarity grades

**If accuracy < 65%:**
- Likely a lighting or image quality problem
- Verify dark chamber is blocking all ambient light
- Experiment with backlight distance/intensity
- Try two-shot protocol (separate backlight and ring light captures)
- Higher camera resolution

**Common failure modes and fixes:**

| Problem | Symptom | Fix |
|---|---|---|
| Translucent vs transparent confusion | High clarity error rate | Increase backlight brightness; ensure backlight is directly below stone; try two-shot protocol |
| Light vs blue confusion | Color errors on medium-blue stones | Improve ring light CRI; ensure consistent white balance; add more borderline examples to training |
| Inconsistent background | Random errors | **Check dark chamber** — seal light leaks; re-capture background more frequently |
| Stones out of focus | Low accuracy across all classes | Reduce aperture (if adjustable); ensure camera height is fixed; mark the "sweet spot" position |
| Ambient light contamination | Accuracy varies by time of day | Fix dark chamber! This is the #1 issue. |

---

## 4. Quick-Start Camera Setup Guide

### Physical Positioning

```
Optimal geometry for 1-3.5mm stones (INSIDE DARK CHAMBER):

         Camera
           │
           │  50-80mm (working distance)
           │  Adjust until stone fills ~1/3 of frame
           ▼
    ═══════════════  ← Ring light (if separate from camera)
                        As close to lens as possible
                        Angled ~30° inward
    
    ───────────────  ← Clear acrylic sheet (stone sits here)
         ● stone
    ───────────────
           │
           │  5-10mm gap (air gap for diffusion)
           │
    ┌─────────────┐  ← Backlight (LED panel)
    └─────────────┘     Centered under camera

    UV LED ◉ ──────────── Mounted on chamber wall
                          Aimed at stone, 30-45° angle
```

### Camera Settings Cheat Sheet

| Setting | Value | Why |
|---|---|---|
| Resolution | Maximum available | More pixels per stone = better classification |
| Exposure | Manual, 2000-5000µs | Auto-exposure changes between stones → inconsistent |
| White balance | Manual, set once against white paper | Auto WB will shift colors |
| Focus | Manual, fixed | Autofocus hunts and causes blur |
| Gain/ISO | Low as possible while maintaining brightness | High gain = noise |

### How to Set Focus
1. Place a stone (or small printed text) on the acrylic
2. View live camera feed
3. Adjust focus ring (or camera height for fixed-focus) until edges are sharpest
4. Lock focus — tape the ring or mark the height
5. Don't touch it again

### How to Set White Balance
1. Place a white sheet of paper on the acrylic
2. Turn on ring light only (backlight off)
3. If camera supports manual WB: set to ~5500-6500K
4. If camera only has auto WB: let it auto-calibrate on the white paper, then switch to manual/lock
5. The paper should look neutral white on screen, not yellowish or bluish

### How to Set Exposure
1. Place a medium-blue stone on the acrylic
2. Ring light ON, backlight OFF
3. Adjust exposure until the stone is well-lit but not blown out (no pure white areas on the stone)
4. Check: darkest stones (inky/opaque) should still show some detail, not just black
5. Check: lightest stones (light/transparent) should not be blown out to white
6. Record the exposure value — use it for all captures

### Lighting Mode Tests

```python
"""Test lighting modes. Toggle backlight, ring light, and UV to verify separation."""
import cv2

cam = cv2.VideoCapture(0)

print("Manually toggle your lights and observe:")
print("1. Backlight ON, Ring OFF → transparent stones glow, opaque are dark")
print("2. Ring ON, Backlight OFF → see stone COLOR (blue vs light vs dark)")
print("3. Both ON → combined view")
print("4. UV ON, others OFF → check fluorescence differences")
print("Press ESC to quit")

while True:
    ret, frame = cam.read()
    cv2.imshow("Lighting Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
```

**What to verify:**
- ✅ Backlight only: transparent stones are clearly brighter than opaque
- ✅ Ring light only: blue stones look blue, light stones look lighter, inky stones look dark
- ✅ You can visually distinguish all 3 clarity levels under backlight
- ✅ You can visually distinguish all 3 color levels under ring light
- ✅ Dark chamber blocks all ambient light — image doesn't change when room lights toggle
- 🔬 UV only: note any fluorescence differences between clarity/color grades (bonus data)
- ❌ If you CAN'T distinguish visually, the AI won't either → fix lighting first

---

## 5. Image Capture Protocol

### Pre-Capture Checklist
- [ ] **Dark chamber sealed** — no ambient light leaks
- [ ] Camera mounted, focused, exposure locked
- [ ] White balance set and locked
- [ ] Backlight working, centered under camera
- [ ] Ring light working, even illumination
- [ ] Background frame captured (no stone, lights on)
- [ ] `collect_data.py` running and detecting stones correctly
- [ ] Test: place stone → green box appears → remove stone → box disappears

### Capture Procedure

1. **Pre-sort stones by eye** into 9 piles (3 color × 3 clarity). Don't agonize over borderline cases — put them in the closest class. You'll capture them and the model will learn the boundary.

2. **Batch capture by class:**
   - Set color key (1/2/3) and clarity key (q/w/e) for the current pile
   - Place stone → wait for detection → press ENTER → remove stone
   - Repeat for entire pile
   - Move to next class

3. **Capture borderline stones in BOTH classes** — if a stone is between "blue" and "light", capture it once labeled as each. This teaches the model the boundary.

### Image Specifications

| Parameter | Value |
|---|---|
| Saved resolution | 224 × 224 px (normalized), plus raw crop |
| Format | JPEG, quality 95 |
| Color space | BGR (OpenCV default) |
| Background | Should be consistent dark or backlit |
| Stone centering | Automatic via bounding box + padding |

### Minimum Dataset Targets

| Class | Minimum | Target | Notes |
|---|---|---|---|
| blue_transparent | 25 | 100+ | Most valuable — CUT stones |
| blue_translucent | 25 | 50+ | |
| blue_opaque | 25 | 50+ | |
| light_transparent | 25 | 50+ | Heat treatment candidates |
| light_translucent | 25 | 50+ | |
| light_opaque | 25 | 50+ | |
| inky_transparent | 25 | 50+ | |
| inky_translucent | 25 | 50+ | |
| inky_opaque | 25 | 50+ | |
| **TOTAL** | **225** | **500+** | |

### Single vs Two-Shot Lighting

**Single-shot (combined lighting) — recommended for MVP:**
- Both backlight and ring light ON simultaneously
- Faster capture (one image per stone)
- The model learns to extract color AND clarity from a single combined image
- May work well enough — test this first

**Two-shot (sequential lighting) — if single-shot accuracy is poor:**
- Shot 1: Backlight ON, ring light OFF → clarity features
- Shot 2: Ring light ON, backlight OFF → color features
- Classical CV approach naturally handles this — train separate RF models per lighting mode
- CNN approach: feed both images as 6-channel input (see training data protocol doc)

**Start with single-shot.** Only switch to two-shot if clarity classification accuracy is below 75%.

---

## 6. Model Training Quick-Start

### Phase 1: Classical CV (Random Forest) — START HERE

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

pip install opencv-python numpy scikit-learn pillow matplotlib
```

```bash
# Train — takes seconds
python train_classical.py
```

**Advantages of starting with Classical CV:**
- Trains in seconds, not minutes/hours
- No GPU needed
- Works with 200+ images
- Interpretable features
- Fast iteration cycle

### Phase 2: CNN (MobileNetV3) — When You Have 500+ Images

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime

python train_cnn.py
```

The CNN training script handles everything:
- Loads images from `images/` directory
- Splits 80/10/10 (train/val/test)
- Trains MobileNetV3-Small with dual heads for 50 epochs
- Saves best model to `models/best_model.pth`
- Exports to `models/best_model.onnx`
- Prints confusion matrices on test set

### Phase 3: Ensemble (Both) — Best Accuracy

When both models are trained, you can ensemble them:

```python
def ensemble_predict(classical_clf, cnn_clf, image_bgr, alpha=0.5):
    """Combine predictions from both classifiers."""
    r1 = classical_clf.predict(image_bgr)
    r2 = cnn_clf.predict(image_bgr)
    
    # Weighted average of probabilities
    combined_color = {}
    combined_clarity = {}
    for cls in r1["color_probs"]:
        combined_color[cls] = alpha * r1["color_probs"][cls] + (1-alpha) * r2["color_probs"][cls]
    for cls in r1["clarity_probs"]:
        combined_clarity[cls] = alpha * r1["clarity_probs"][cls] + (1-alpha) * r2["clarity_probs"][cls]
    
    color = max(combined_color, key=combined_color.get)
    clarity = max(combined_clarity, key=combined_clarity.get)
    
    return {
        "color": color,
        "color_conf": combined_color[color],
        "clarity": clarity,
        "clarity_conf": combined_clarity[clarity],
    }
```

### Expected Accuracy by Dataset Size and Method

| Images per class | Classical CV (RF) | CNN (MobileNetV3) | Ensemble |
|---|---|---|---|
| 25 (225 total) | 65–78% | 55–68% | 65–78% |
| 50 (450 total) | 72–83% | 65–78% | 74–85% |
| 100 (900 total) | 78–87% | 78–88% | 82–90% |
| 200+ (1800+ total) | 80–88% | 85–93% | 87–94% |

**The crossover:** Classical CV is better below ~100 images/class. CNN catches up around 100 and surpasses at 200+. Ensemble is always best.

### ONNX Export (for future Jetson deployment)

```python
from classify_cnn import DualHeadMobileNet
import torch

model = DualHeadMobileNet()
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

dummy = torch.randn(1, 3, 128, 128)
torch.onnx.export(model, dummy, "models/best_model.onnx",
                  input_names=["image"],
                  output_names=["color_logits", "clarity_logits"],
                  opset_version=13)
```

For future Jetson deployment, convert ONNX → TensorRT:
```bash
# On the Jetson device:
/usr/src/tensorrt/bin/trtexec --onnx=best_model.onnx --saveEngine=best_model.trt --fp16
```

---

## 7. Testing Protocol

### Accuracy Test Procedure

1. Set aside 50+ stones NOT used in training (ideally 5-6 per class)
2. Run gui.py — model predicts, you confirm or correct
3. Record results
4. **Test BOTH models** (classical and CNN) on the same stones for comparison

### Confusion Matrix Interpretation

```
Example color confusion matrix:
              Predicted:
              blue    inky    light
True: blue    [  42      1       2  ]   ← 93% recall for blue
      inky    [   0     18       3  ]   ← 86% recall for inky
      light   [   3      2      29  ]   ← 85% recall for light
```

**Most costly misclassifications (ranked by business impact):**

| Misclassification | Impact | Why |
|---|---|---|
| CUT-grade → FLIP/REJECT | **CRITICAL** — Lost revenue | Blue+transparent stones sorted wrong = money left on table |
| FLIP → CUT | **HIGH** — Wasted cutting cost | Paying to cut a stone that should've been flipped |
| SELECT → REJECT | **MEDIUM** — Lost opportunity | Threw away a usable stone |
| REJECT → SELECT | **LOW** — Minor waste | Extra human review of junk stones |

---

## 8. What Success Looks Like

### MVP Success Criteria

| Metric | Minimum | Target |
|---|---|---|
| Color classification accuracy | 80% | 90%+ |
| Clarity classification accuracy | 75% | 85%+ |
| Combined (both correct) | 70% | 80%+ |
| CUT-grade recall (don't miss CUT stones) | 85% | 95%+ |
| Processing speed (semi-auto mode) | 1 stone/sec | 2+ stones/sec |
| Total images captured for training | 225 | 1000+ |

### Decision Points

**If MVP succeeds (combined accuracy > 80%):**
→ Proceed to full system build from [architecture doc](./sorting-system-architecture.md)
→ Training data collected during MVP becomes the foundation for production model
→ Camera and lighting setup carries over directly
→ Decide: full custom build vs adapt mini color sorter (see Buy vs Build section)

**If MVP partially succeeds (60-80% accuracy):**
→ Likely fixable with more data + lighting improvements
→ Spend another week on data collection and lighting experiments
→ Consider two-shot protocol if clarity is the weak axis
→ Try UV fluorescence capture

**If MVP fails (< 60% accuracy):**
→ Fundamental problem with either lighting or class definitions
→ Re-examine: are the 9 classes actually visually distinguishable at this stone size?
→ Consider simplifying: 2 classes (CUT vs not-CUT) instead of full 3×3 matrix
→ Consider different imaging modality (UV fluorescence, spectrometer)

### What You'll Have After 7 Days

1. **Working camera station** (with dark chamber) — reusable for production system
2. **Training dataset** — 225-1000+ labeled stone images
3. **Two trained models** — Random Forest (classical) + MobileNetV3 (CNN)
4. **Accuracy data** — know exactly which classes work and which don't
5. **Clear next steps** — whether to proceed with full build or iterate on vision
6. **UV fluorescence data** (optional) — may reveal additional classification signal

---

## Quick Reference: All Chinese Search Terms

| Item | 淘宝/1688 Search | 华强北 Ask For |
|---|---|---|
| USB industrial camera | `USB工业相机 200万 微距` | "USB工业相机，拍微距的" |
| Hikvision camera | `海康机器人 MV-CS050 工业相机` | (order online) |
| C-mount macro lens | `工业微距镜头 C口 1倍` | "C口微距镜头" |
| LED backlight | `LED背光源 机器视觉 白色` or `LED拷贝台 A5` | "LED灯板，白光的" |
| LED ring light | `LED环形灯 显微镜 可调亮度` | "显微镜用的环形灯" |
| **UV LED module** | `365nm UV LED灯珠 模组 紫外线` | "紫外线LED灯，365纳米的" |
| Adjustable stand | `万向支架 相机支架 显微镜` | "万向支架，固定相机用" |
| Clear acrylic sheet | `透明亚克力板 定制` | "透明亚克力板，要厚一点的" |
| Black foamcore board | `黑色泡沫板 A3` or `黑色KT板` | "黑色泡沫板，遮光用的" |
| Black cloth | `黑色绒布 遮光布` | "黑色绒布" |
| USB hub | `USB集线器 带供电 HUB` | "USB HUB，要带供电的" |
| Mini color sorter | `小型色选机 矿石 迷你` | (order online from 1688) |

---

## Open Source Resources

| Repository | What | How to Use |
|---|---|---|
| **TommasoBendinelli/Gemtelligence** | Deep learning sapphire classification (Gübelin Gem Lab) | Study the HoL-Net architecture for multi-modal fusion. Their approach of combining spectral + tabular data inspires combining image + UV + size features. Code is PyTorch. |
| **hybchow/gems** | Classical CV gemstone classification (Random Forest) | **Directly applicable.** Their feature extraction (HSV/CIELAB histograms + LBP + Haralick) is the basis for our Phase 1 approach. Published result: RF outperformed ResNet. |
| **loremendez/Gemstones** | CNN gemstone classification (TensorFlow) | Reference for deep learning architecture. We use PyTorch instead but the concepts transfer. |
| **RobertGetzner/IronBeadSorter** | DIY camera + OpenCV sorting machine | Good reference for the mechanical + software integration pattern. Uses color histograms. |

---

*This is a 7-day sprint. Don't over-engineer. Start with Classical CV (it's faster and works with less data). Get stones in front of a camera, get images classified, measure accuracy. Build the dark chamber on Day 1 — it's the single most impactful thing you can do for accuracy. Everything else comes later.*
