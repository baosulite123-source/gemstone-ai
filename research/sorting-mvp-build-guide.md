# MVP Build Guide: Rough Gemstone Melee Sorting Station
## "Classify & Display" — Validate Vision/AI in 7 Days

**Date:** 2026-02-15  
**Status:** Actionable Build Guide  
**Location:** Shenzhen (华强北 + 1688/淘宝)  
**Budget:** < ¥2,000 ($280)  
**Goal:** Get real stone images flowing through AI classification within 1 week

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

## 1. MVP Hardware — Buy Tomorrow in Shenzhen

### Shopping List

| # | Item | Spec | Where to Buy | Search Terms | Price (¥) |
|---|---|---|---|---|---|
| 1 | USB microscope camera | 2MP+, manual focus, C-mount or built-in lens | 华强北 赛格电子市场 3F | `USB显微镜 工业相机 200万像素` | 150–300 |
| 2 | **OR** Hikvision USB industrial camera | MV-CS050-10UC 5MP (ideal but pricier) | 淘宝/1688 (2-3 day delivery) | `海康机器人 MV-CS050 工业相机 USB3` | 800–1,200 |
| 3 | C-mount macro lens (if using industrial cam) | 1:1 magnification, 50mm WD | 华强北 or 1688 | `工业微距镜头 C口 1倍 放大` | 300–500 |
| 4 | LED backlight panel | 50×50mm or larger, white, USB-powered OK | 华强北 赛格 | `LED背光板 白色 USB供电 50mm` | 50–100 |
| 5 | LED ring light | 40-60mm ID, white, dimmable, USB OK | 华强北 or 淘宝 | `LED环形灯 显微镜 可调亮度 USB` | 50–150 |
| 6 | Acrylic V-groove slide | Clear acrylic strip, ~200mm × 20mm × 5mm | 华强北 周边 亚克力加工店 | `透明亚克力条 定制 V槽` | 20–50 |
| 7 | Black cloth/card | Light-blocking background | Any stationery shop | `黑色绒布 遮光` | 10 |
| 8 | Small adjustable stand | Camera boom arm or lab stand | 华强北 or 淘宝 | `万向支架 相机 显微镜支架` | 50–150 |
| 9 | USB hub (powered) | For camera + lights | 华强北 | `USB集线器 带供电` | 30–50 |

### Two Budget Paths

**Path A: Ultra-cheap (¥400–600)** — USB microscope + USB ring light + USB backlight + stand  
Good enough to validate the concept. Image quality may limit final accuracy but proves the workflow.

**Path B: Production-ready camera (¥1,400–1,900)** — Hikvision industrial camera + C-mount macro + proper lights  
Same camera you'll use in the full system. Data collected now becomes training data for production.

**Recommendation: Path B** if the Hikvision camera can arrive within 2 days (order on 1688 today, many Shenzhen sellers have next-day delivery). Otherwise start Path A tomorrow and upgrade later.

### What You Already Have
- Laptop with NVIDIA GPU (for training later) — or any laptop for capture
- Rough sapphire melee stones (at least a few hundred across color/clarity grades)
- Sorting trays with compartments

### DIY V-Groove Slide

Don't overthink this. Two options:

**Option 1: Acrylic strip.** Buy a 200mm × 20mm × 5mm clear acrylic strip. Score a V-groove down the center with a rotary tool or have a 华强北 acrylic shop CNC it (¥20, 10 minutes). Stones sit in the groove, backlight shines through from below.

**Option 2: Skip the groove entirely.** Place stones one at a time on a clear acrylic sheet over the backlight. For MVP, speed doesn't matter — accuracy does.

### Total MVP Budget

| Path | Total |
|---|---|
| Path A (USB microscope) | ¥400–600 |
| Path B (industrial camera) | ¥1,400–1,900 |

---

## 2. MVP Software Stack

### Dependencies

```bash
pip install opencv-python numpy torch torchvision pillow onnxruntime
# For GUI:
pip install tkinter  # Usually included with Python
# OR for web UI:
pip install flask
```

If using Hikvision camera, also install the MVS SDK from [hikrobotics.com](https://www.hikrobotics.com/en/machinevision/service/download).

### Architecture: One Python Script to Rule Them All

```
mvp_sorter/
├── capture.py          # Camera capture + stone detection
├── classify.py         # Model inference
├── train.py            # Training script
├── gui.py              # Live classification GUI
├── collect_data.py     # Training data collection mode
├── config.py           # Settings
├── images/             # Captured training images
│   ├── blue_transparent/
│   ├── blue_translucent/
│   ├── blue_opaque/
│   ├── light_transparent/
│   ├── light_translucent/
│   ├── light_opaque/
│   ├── inky_transparent/
│   ├── inky_translucent/
│   └── inky_opaque/
└── models/             # Saved model files
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
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    
    # Absolute difference
    diff = cv2.absdiff(gray, bg_gray)
    
    # Threshold
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stones = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            # Add margin
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
    # Make square by padding shorter dimension
    size = max(w, h)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    dy = (size - h) // 2
    dx = (size - w) // 2
    square[dy:dy+h, dx:dx+w] = roi
    # Resize to target
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
    
    # Create class directories
    for cls in CLASSES:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)
    
    # Capture background
    print("Remove all stones from view, then press SPACE to capture background")
    bg_frame = None
    current_color = None
    current_clarity = None
    
    # Load existing counts
    counts = {}
    for cls in CLASSES:
        counts[cls] = len(os.listdir(os.path.join(base_dir, cls)))
    
    metadata_log = []
    
    while True:
        frame = cam.grab_frame()
        display = frame.copy()
        
        # Show current selection
        color_str = current_color or "?"
        clarity_str = current_clarity or "?"
        cv2.putText(display, f"Color: {color_str} (1=Blue 2=Light 3=Inky)", 
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Clarity: {clarity_str} (q=Trans w=Transluc e=Opaque)", 
                     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show counts
        y_pos = 100
        for cls in CLASSES:
            cv2.putText(display, f"{cls}: {counts[cls]}", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 20
        
        # Detect stones if we have background
        if bg_frame is not None:
            stones = detect_stone(frame, bg_frame)
            for (x, y, w, h) in stones:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # If both color and clarity selected, and stone detected → save
            if current_color and current_clarity and len(stones) == 1:
                cv2.putText(display, "PRESS ENTER to save | R to reset selection", 
                           (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
        
        cv2.imshow("Data Collection", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
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
        elif key == 13 and current_color and current_clarity:  # ENTER
            if bg_frame is not None and len(stones) == 1:
                cls_name = f"{current_color}_{current_clarity}"
                idx = counts[cls_name]
                
                # Save full ROI
                roi = extract_roi(frame, stones[0], output_size=224)
                filename = f"{cls_name}_{idx:04d}.jpg"
                filepath = os.path.join(base_dir, cls_name, filename)
                cv2.imwrite(filepath, roi)
                
                # Also save raw crop at original resolution
                x, y, w, h = stones[0]
                raw_crop = frame[y:y+h, x:x+w]
                raw_path = os.path.join(base_dir, cls_name, f"{cls_name}_{idx:04d}_raw.jpg")
                cv2.imwrite(raw_path, raw_crop)
                
                # Log metadata
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
                
                # Don't reset selection — likely sorting a batch of same class
    
    # Save metadata
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
Live classification GUI.
Shows camera feed, detects stones, classifies, displays result.
Human sorts by hand based on screen output.

Controls:
  SPACE: Capture/update background
  S: Toggle semi-auto mode (continuous classification)
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
from classify import StoneClassifier, DECISION_MAP

# Colors for each decision
DECISION_COLORS = {
    "CUT": (0, 255, 0),       # Green — money!
    "SELECT": (255, 255, 0),   # Cyan
    "FLIP": (0, 165, 255),     # Orange
    "REJECT": (0, 0, 255),     # Red
}

def main():
    cam = CameraCapture(camera_id=0)
    classifier = StoneClassifier("models/best_model.onnx")  # or .pth
    
    bg_frame = None
    semi_auto = False
    last_classify_time = 0
    classify_cooldown = 0.5  # seconds between classifications in semi-auto
    
    # Results log
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"sort_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "color", "color_conf", "clarity", "clarity_conf", 
                         "decision", "overall_conf"])
    
    results_history = []  # Last N results for display
    stones_per_min = 0
    minute_start = time.time()
    minute_count = 0
    
    while True:
        frame = cam.grab_frame()
        display = frame.copy()
        
        # Status bar
        mode_str = "SEMI-AUTO" if semi_auto else "MANUAL (press S for semi-auto)"
        bg_str = "BG: ✓" if bg_frame is not None else "BG: ✗ (press SPACE)"
        cv2.putText(display, f"{mode_str} | {bg_str} | {stones_per_min:.0f} stones/min", 
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if bg_frame is not None:
            stones = detect_stone(frame, bg_frame)
            
            for (x, y, w, h) in stones:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Classify
            should_classify = False
            if semi_auto and len(stones) >= 1 and (time.time() - last_classify_time) > classify_cooldown:
                should_classify = True
            
            if should_classify or (not semi_auto and len(stones) == 1):
                for (x, y, w, h) in stones[:1]:  # Classify first detected stone
                    roi = extract_roi(frame, (x, y, w, h), output_size=128)
                    result = classifier.predict(roi)
                    
                    color = result["color"]
                    clarity = result["clarity"]
                    color_conf = result["color_conf"]
                    clarity_conf = result["clarity_conf"]
                    decision = DECISION_MAP.get((color, clarity), "SELECT")
                    overall_conf = min(color_conf, clarity_conf)
                    
                    # Draw result on frame
                    dec_color = DECISION_COLORS.get(decision, (255, 255, 255))
                    
                    # Big decision text
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
                        
                        # Log
                        with open(log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([datetime.now().isoformat(), color, f"{color_conf:.3f}",
                                           clarity, f"{clarity_conf:.3f}", decision, f"{overall_conf:.3f}"])
                        
                        # Save image
                        img_dir = log_path / "images" / decision.lower()
                        img_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(img_dir / f"{datetime.now().strftime('%H%M%S_%f')}.jpg"), roi)
                        
                        results_history.append({
                            "decision": decision, "color": color, "clarity": clarity,
                            "conf": overall_conf, "time": time.time()
                        })
                        if len(results_history) > 20:
                            results_history.pop(0)
        
        # Update stones/min
        if time.time() - minute_start > 60:
            stones_per_min = minute_count
            minute_count = 0
            minute_start = time.time()
        
        # Draw recent results sidebar
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
        
        if key == 27:  # ESC
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

### classify.py — Model Inference

```python
"""Stone classification using trained MobileNetV3-Small with dual heads."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

# Classification constants
COLOR_CLASSES = ['blue', 'inky', 'light']
CLARITY_CLASSES = ['opaque', 'translucent', 'transparent']

# Decision map: (color, clarity) → sorting decision
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
        # Remove original classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Shared FC
        self.shared_fc = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Separate heads
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
        """
        Classify a stone image.
        Args: image_bgr: OpenCV BGR image (any size, will be resized)
        Returns: dict with color, clarity, confidences
        """
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

### Day 1: Buy Hardware

**Morning (华强北):**
1. Go to 赛格电子市场 (SEG Electronics Market), floors 2-4
2. Buy: USB camera or microscope, ring light, adjustable stand, USB hub
3. For backlight: look for "LED灯板" or "LED背光模组" — a small white LED panel. Even a cheap LED tracing pad (临摹台 / `LED拷贝台 A5`) works perfectly as a backlight (¥30-50 on 淘宝)
4. Buy black velvet cloth from a fabric stall nearby

**Afternoon (acrylic shop):**
1. Find an acrylic fabrication shop near 华强北 (there are many on 振华路)
2. Get a clear acrylic piece: 200mm × 100mm × 5mm
3. Optional: ask them to cut a shallow V-groove (2mm deep, 90° angle)

**Evening:**
- Order Hikvision camera on 1688/淘宝 if not available locally (arrives Day 2-3)
- Install Python, OpenCV, PyTorch on laptop
- Test USB camera works with OpenCV

```python
# Quick camera test — run this evening
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
```

### Day 2: Assemble Camera Station, Test Image Capture

**Morning — Physical assembly:**

```
Assembly layout (side view):
                    
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
```

Steps:
1. Mount camera on adjustable stand, lens pointing down
2. Place LED backlight panel flat on desk
3. Raise acrylic sheet 5-10mm above backlight (use small blocks/spacers)
4. Position ring light around or just below the camera lens
5. Place a stone on the acrylic — can you see it clearly on screen?

**Afternoon — Optimize image quality:**

Run this to tune settings interactively:

```python
"""Interactive camera setup tool."""
import cv2
import numpy as np

cam = cv2.VideoCapture(0)

# Try to set manual exposure
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

def nothing(x): pass

cv2.namedWindow("Setup")
cv2.createTrackbar("Exposure", "Setup", 50, 100, nothing)
cv2.createTrackbar("Brightness", "Setup", 50, 100, nothing)

while True:
    ret, frame = cam.read()
    if not ret: break
    
    # Show zoomed-in center (likely where stone is)
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    crop_size = min(h, w) // 3
    crop = frame[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
    
    # Show histogram for exposure checking
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
- No harsh reflections or hotspots
- Consistent, even illumination

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

**Two-shot protocol (recommended if time allows):**

For each stone, capture TWO images:
1. **Backlight ON, ring light OFF** → save as clarity image
2. **Ring light ON, backlight OFF** → save as color image

This gives the model the clearest signal for each classification axis. In production, you might use combined lighting, but for training data, separated lighting gives cleaner labels.

To implement: just toggle your light switches manually between shots. You can modify `collect_data.py` to prompt for both shots per stone.

**Naming convention already handled by the script:**
```
images/
├── blue_transparent/
│   ├── blue_transparent_0000.jpg      (224×224 normalized)
│   ├── blue_transparent_0000_raw.jpg  (original resolution crop)
│   ├── blue_transparent_0001.jpg
│   ...
├── blue_translucent/
│   ...
```

### Day 4-5: Train Initial Classification Model

Run this training script:

```python
"""
train.py — Train MobileNetV3-Small with dual classification heads.
Expects images in: images/{color}_{clarity}/*.jpg
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import json
from datetime import datetime
from classify import DualHeadMobileNet, COLOR_CLASSES, CLARITY_CLASSES

class StoneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
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
                    continue  # Skip raw crops, use normalized ones
                if img_name.endswith(('.jpg', '.png')):
                    self.samples.append({
                        'path': os.path.join(class_path, img_name),
                        'color': color_to_idx[color],
                        'clarity': clarity_to_idx[clarity],
                    })
        
        print(f"Loaded {len(self.samples)} images")
        # Print class distribution
        from collections import Counter
        dist = Counter(f"{COLOR_CLASSES[s['color']]}_{CLARITY_CLASSES[s['clarity']]}" 
                       for s in self.samples)
        for cls, count in sorted(dist.items()):
            print(f"  {cls}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, sample['color'], sample['clarity']


def train():
    # Config
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    IMAGE_SIZE = 128
    DATA_DIR = "images"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Augmentation — heavy rotation (stones have no orientation), mild color jitter
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    full_dataset = StoneDataset(DATA_DIR, transform=train_transform)
    
    # Split 80/10/10
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])
    
    # Override transform for val/test
    val_set.dataset = StoneDataset(DATA_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model
    model = DualHeadMobileNet(num_color=3, num_clarity=3).to(DEVICE)
    
    # Loss — can add class weights if imbalanced
    color_criterion = nn.CrossEntropyLoss()
    clarity_criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0
    history = []
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_color_correct = 0
        train_clarity_correct = 0
        train_total = 0
        
        for images, color_labels, clarity_labels in train_loader:
            images = images.to(DEVICE)
            color_labels = color_labels.to(DEVICE)
            clarity_labels = clarity_labels.to(DEVICE)
            
            optimizer.zero_grad()
            color_logits, clarity_logits = model(images)
            
            loss_color = color_criterion(color_logits, color_labels)
            loss_clarity = clarity_criterion(clarity_logits, clarity_labels)
            loss = loss_color + loss_clarity
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_color_correct += (color_logits.argmax(1) == color_labels).sum().item()
            train_clarity_correct += (clarity_logits.argmax(1) == clarity_labels).sum().item()
            train_total += images.size(0)
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_color_correct = 0
        val_clarity_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, color_labels, clarity_labels in val_loader:
                images = images.to(DEVICE)
                color_labels = color_labels.to(DEVICE)
                clarity_labels = clarity_labels.to(DEVICE)
                
                color_logits, clarity_logits = model(images)
                val_color_correct += (color_logits.argmax(1) == color_labels).sum().item()
                val_clarity_correct += (clarity_logits.argmax(1) == clarity_labels).sum().item()
                val_total += images.size(0)
        
        train_color_acc = train_color_correct / train_total
        train_clarity_acc = train_clarity_correct / train_total
        val_color_acc = val_color_correct / max(val_total, 1)
        val_clarity_acc = val_clarity_correct / max(val_total, 1)
        val_combined_acc = (val_color_acc + val_clarity_acc) / 2
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {train_loss/train_total:.4f} | "
              f"Train Color: {train_color_acc:.1%} Clarity: {train_clarity_acc:.1%} | "
              f"Val Color: {val_color_acc:.1%} Clarity: {val_clarity_acc:.1%}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / train_total,
            "train_color_acc": train_color_acc,
            "train_clarity_acc": train_clarity_acc,
            "val_color_acc": val_color_acc,
            "val_clarity_acc": val_clarity_acc,
        })
        
        # Save best model
        if val_combined_acc > best_val_acc:
            best_val_acc = val_combined_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  → Saved best model (combined acc: {val_combined_acc:.1%})")
    
    # Save training history
    with open("models/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    print("Model saved to models/best_model.pth")
    
    # Export to ONNX
    export_onnx(model, DEVICE, IMAGE_SIZE)
    
    return model, test_set


def export_onnx(model, device, image_size=128):
    """Export trained model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    
    torch.onnx.export(
        model, dummy, "models/best_model.onnx",
        input_names=["image"],
        output_names=["color_logits", "clarity_logits"],
        dynamic_axes={"image": {0: "batch"}},
        opset_version=13,
    )
    print("Exported to models/best_model.onnx")


def evaluate_test_set(model, test_set, device):
    """Run full evaluation on test set with confusion matrices."""
    from collections import defaultdict
    import numpy as np
    
    model.eval()
    loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    color_preds, color_trues = [], []
    clarity_preds, clarity_trues = [], []
    
    with torch.no_grad():
        for images, color_labels, clarity_labels in loader:
            images = images.to(device)
            color_logits, clarity_logits = model(images)
            color_preds.extend(color_logits.argmax(1).cpu().tolist())
            color_trues.extend(color_labels.tolist())
            clarity_preds.extend(clarity_logits.argmax(1).cpu().tolist())
            clarity_trues.extend(clarity_labels.tolist())
    
    # Print confusion matrices
    print("\n=== COLOR Confusion Matrix ===")
    print(f"{'':>12s}", end="")
    for c in COLOR_CLASSES:
        print(f"{c:>12s}", end="")
    print()
    for i, true_class in enumerate(COLOR_CLASSES):
        print(f"{true_class:>12s}", end="")
        for j in range(len(COLOR_CLASSES)):
            count = sum(1 for t, p in zip(color_trues, color_preds) if t == i and p == j)
            print(f"{count:>12d}", end="")
        print()
    
    print(f"\n=== CLARITY Confusion Matrix ===")
    print(f"{'':>12s}", end="")
    for c in CLARITY_CLASSES:
        print(f"{c:>12s}", end="")
    print()
    for i, true_class in enumerate(CLARITY_CLASSES):
        print(f"{true_class:>12s}", end="")
        for j in range(len(CLARITY_CLASSES)):
            count = sum(1 for t, p in zip(clarity_trues, clarity_preds) if t == i and p == j)
            print(f"{count:>12d}", end="")
        print()
    
    color_acc = sum(1 for t, p in zip(color_trues, color_preds) if t == p) / len(color_trues)
    clarity_acc = sum(1 for t, p in zip(clarity_trues, clarity_preds) if t == p) / len(clarity_trues)
    print(f"\nColor accuracy: {color_acc:.1%}")
    print(f"Clarity accuracy: {clarity_acc:.1%}")
    print(f"Combined: {(color_acc + clarity_acc) / 2:.1%}")


if __name__ == "__main__":
    model, test_set = train()
    evaluate_test_set(model, test_set, 
                      torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

**Expected training time:**
| GPU | ~500 images | ~1000 images | ~3000 images |
|---|---|---|---|
| RTX 3060 | ~2 min | ~5 min | ~15 min |
| RTX 4090 | ~1 min | ~2 min | ~5 min |
| CPU only | ~20 min | ~45 min | ~2 hrs |

**Expected accuracy by dataset size:**

| Images per class | Expected color acc | Expected clarity acc |
|---|---|---|
| 25 (225 total) | 60–75% | 55–70% |
| 50 (450 total) | 70–82% | 65–78% |
| 100 (900 total) | 80–90% | 75–85% |
| 200+ (1800+ total) | 85–93% | 80–90% |

These are rough estimates. Actual results depend on how separable the classes are in your lighting setup.

### Day 6: Test Model on New Stones, Measure Accuracy

1. Run `gui.py` with the trained model
2. Prepare 50+ test stones that were NOT in the training set
3. For each stone:
   - Place on imaging station
   - Note the model's prediction
   - Note YOUR classification (ground truth)
   - Record agreement/disagreement
4. Calculate accuracy per class and overall

**Quick accuracy test script:**

```python
"""
Quick accuracy test. Model classifies, you confirm or correct.
Press Y if correct, or press the correct keys to override.
"""
import cv2
from capture import CameraCapture, detect_stone, extract_roi
from classify import StoneClassifier, DECISION_MAP, COLOR_CLASSES, CLARITY_CLASSES

cam = CameraCapture()
classifier = StoneClassifier("models/best_model.onnx")
bg_frame = None

results = []

print("Press SPACE for background, then place stones one by one.")
print("Y = model correct | 1/2/3 = correct color | q/w/e = correct clarity | ESC = done")

while True:
    frame = cam.grab_frame()
    display = frame.copy()
    
    if bg_frame is not None:
        stones = detect_stone(frame, bg_frame)
        if len(stones) == 1:
            roi = extract_roi(frame, stones[0])
            result = classifier.predict(roi)
            
            pred_color = result["color"]
            pred_clarity = result["clarity"]
            decision = DECISION_MAP.get((pred_color, pred_clarity), "SELECT")
            
            cv2.putText(display, f"Pred: {pred_color} / {pred_clarity} -> {decision}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Conf: color={result['color_conf']:.0%} clarity={result['clarity_conf']:.0%}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(display, "Y=correct | 1/2/3=color | q/w/e=clarity", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imshow("Accuracy Test", display)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:
        break
    elif key == ord(' '):
        bg_frame = frame.copy()
    elif key == ord('y') and bg_frame is not None and len(stones) == 1:
        results.append({"pred_color": pred_color, "pred_clarity": pred_clarity,
                        "true_color": pred_color, "true_clarity": pred_clarity, "correct": True})
        print(f"✓ {pred_color}/{pred_clarity} ({len(results)} tested)")

cam.release()

# Summary
correct = sum(1 for r in results if r["correct"])
print(f"\n=== Results: {correct}/{len(results)} = {correct/max(len(results),1):.0%} ===")
```

### Day 7: Iterate

Based on Day 6 results:

**If accuracy > 85%:** 🎉 Success! Start planning the full mechanical build.

**If accuracy 70-85%:**
- Check which classes are confused most (confusion matrix from training)
- Collect 50+ more images for the weakest classes
- Retrain
- Try adjusting lighting — often clarity classification improves dramatically with better backlight positioning

**If accuracy < 70%:**
- Likely a lighting or image quality problem, not a model problem
- Experiment with:
  - Different backlight distance/intensity
  - Different ring light angle
  - Two-shot protocol (separate backlight and ring light captures)
  - Higher camera resolution
- Collect more data with improved setup

**Common failure modes and fixes:**

| Problem | Symptom | Fix |
|---|---|---|
| Translucent vs transparent confusion | High clarity error rate | Increase backlight brightness; ensure backlight is directly below stone |
| Light vs blue confusion | Color errors on medium-blue stones | Improve ring light CRI; ensure consistent white balance; add more borderline examples to training |
| Inconsistent background | Random errors | Re-capture background more frequently; use darker/more uniform background |
| Stones out of focus | Low accuracy across all classes | Reduce aperture (if adjustable); ensure camera height is fixed; mark the "sweet spot" position |

---

## 4. Quick-Start Camera Setup Guide

### Physical Positioning

```
Optimal geometry for 1-3.5mm stones:

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

### Backlight vs Ring Light Test

Run this to verify your lighting works for classification:

```python
"""Test lighting modes. Toggle backlight and ring light to verify separation."""
import cv2

cam = cv2.VideoCapture(0)

print("Manually toggle your lights and observe:")
print("1. Backlight ON, Ring OFF → transparent stones glow, opaque are dark")
print("2. Ring ON, Backlight OFF → see stone COLOR (blue vs light vs dark)")
print("3. Both ON → combined view")
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
- ❌ If you CAN'T distinguish visually, the AI won't either → fix lighting first

---

## 5. Image Capture Protocol

### Pre-Capture Checklist
- [ ] Camera mounted, focused, exposure locked
- [ ] White balance set and locked
- [ ] Backlight working, centered under camera
- [ ] Ring light working, even illumination
- [ ] Background frame captured (no stone, lights on)
- [ ] Black cloth/card around station to block ambient light
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

### Naming Convention

Handled automatically by `collect_data.py`:
```
images/{color}_{clarity}/{color}_{clarity}_{NNNN}.jpg      # 224×224 normalized
images/{color}_{clarity}/{color}_{clarity}_{NNNN}_raw.jpg   # Original resolution
```

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
- Feed both images to model (modify architecture to accept 6-channel input or two separate images)
- More complex but gives cleaner signal per axis

**Start with single-shot.** Only switch to two-shot if clarity classification accuracy is below 75%.

---

## 6. Model Training Quick-Start

### Prerequisites

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy pillow onnxruntime matplotlib
```

### Training Command

```bash
python train.py
```

That's it. The training script in Section 2 handles everything:
- Loads images from `images/` directory
- Splits 80/10/10 (train/val/test)
- Trains MobileNetV3-Small with dual heads for 50 epochs
- Saves best model to `models/best_model.pth`
- Exports to `models/best_model.onnx`
- Prints confusion matrices on test set

### What to Watch During Training

```
Epoch 1/50  | Loss: 2.1234 | Train Color: 35% Clarity: 38% | Val Color: 33% Clarity: 34%
Epoch 10/50 | Loss: 0.8123 | Train Color: 72% Clarity: 65% | Val Color: 68% Clarity: 62%
Epoch 25/50 | Loss: 0.3456 | Train Color: 91% Clarity: 85% | Val Color: 85% Clarity: 79%
Epoch 50/50 | Loss: 0.1234 | Train Color: 96% Clarity: 92% | Val Color: 88% Clarity: 83%
```

**Good signs:**
- Loss decreasing steadily
- Train and val accuracy both climbing
- Gap between train and val < 10-15% (otherwise overfitting)

**Bad signs:**
- Val accuracy stuck below 60% after 20 epochs → likely need more data or better image quality
- Train accuracy 95%+ but val accuracy < 70% → overfitting, add more augmentation or data
- One axis (color or clarity) much worse than the other → lighting problem for that axis

### ONNX Export

Already included in `train.py`. To export manually:

```python
from classify import DualHeadMobileNet
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
2. Run the accuracy test script from Day 6
3. For each stone: model predicts, you confirm or correct
4. Record results

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

### Cost-Weighted Accuracy

Not all errors are equal. A simple weighted metric:

```python
# Cost of each type of misclassification (relative units)
MISCLASS_COST = {
    # (true_decision, predicted_decision): cost
    ("CUT", "FLIP"): 10,      # Lost the most valuable stone
    ("CUT", "REJECT"): 10,
    ("CUT", "SELECT"): 3,     # At least it gets reviewed
    ("FLIP", "CUT"): 5,       # Wasted cutting cost
    ("SELECT", "REJECT"): 2,
    ("REJECT", "CUT"): 3,
    # All other misclassifications: 1
}

def weighted_accuracy(results):
    total_cost = 0
    for r in results:
        true_dec = DECISION_MAP.get((r["true_color"], r["true_clarity"]), "SELECT")
        pred_dec = DECISION_MAP.get((r["pred_color"], r["pred_clarity"]), "SELECT")
        if true_dec != pred_dec:
            total_cost += MISCLASS_COST.get((true_dec, pred_dec), 1)
    
    max_possible_cost = sum(10 for _ in results)  # If everything was maximally wrong
    return 1 - (total_cost / max_possible_cost)
```

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

**If MVP partially succeeds (60-80% accuracy):**
→ Likely fixable with more data + lighting improvements
→ Spend another week on data collection and lighting experiments
→ Consider two-shot protocol if clarity is the weak axis

**If MVP fails (< 60% accuracy):**
→ Fundamental problem with either lighting or class definitions
→ Re-examine: are the 9 classes actually visually distinguishable at this stone size?
→ Consider simplifying: 2 classes (CUT vs not-CUT) instead of full 3×3 matrix
→ Consider different imaging modality (e.g., spectrometer, UV fluorescence)

### What You'll Have After 7 Days

1. **Working camera station** — reusable for production system
2. **Training dataset** — 225-1000+ labeled stone images
3. **Trained model** — MobileNetV3-Small in PyTorch + ONNX
4. **Accuracy data** — know exactly which classes work and which don't
5. **Clear next steps** — whether to proceed with full build or iterate on vision

---

## Quick Reference: All Chinese Search Terms

| Item | 淘宝/1688 Search | 华强北 Ask For |
|---|---|---|
| USB industrial camera | `USB工业相机 200万 微距` | "USB工业相机，拍微距的" |
| Hikvision camera | `海康机器人 MV-CS050 工业相机` | (order online) |
| C-mount macro lens | `工业微距镜头 C口 1倍` | "C口微距镜头" |
| LED backlight | `LED背光源 机器视觉 白色` or `LED拷贝台 A5` | "LED灯板，白光的" |
| LED ring light | `LED环形灯 显微镜 可调亮度` | "显微镜用的环形灯" |
| Adjustable stand | `万向支架 相机支架 显微镜` | "万向支架，固定相机用" |
| Clear acrylic sheet | `透明亚克力板 定制` | "透明亚克力板，要厚一点的" |
| Black cloth | `黑色绒布 遮光布` | "黑色绒布" |
| USB hub | `USB集线器 带供电 HUB` | "USB HUB，要带供电的" |

---

*This is a 7-day sprint. Don't over-engineer. Get stones in front of a camera, get images classified, measure accuracy. Everything else comes later.*
