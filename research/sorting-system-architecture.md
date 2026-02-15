# AI-Powered Rough Gemstone Melee Sorting Machine
## System Architecture & Software Specification

**Document Version:** 1.0  
**Date:** 2026-02-15  
**Status:** Draft Engineering Specification  
**Target:** Rough blue sapphire melee (1–3.5 mm), Thai market lots

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Architecture](#2-hardware-architecture)
3. [Software Architecture](#3-software-architecture)
4. [Training Data Strategy](#4-training-data-strategy)
5. [Calibration & Quality Assurance](#5-calibration--quality-assurance)
6. [Phased Build Plan](#6-phased-build-plan)
7. [Bill of Materials](#7-bill-of-materials)
8. [Risk Assessment](#8-risk-assessment)

---

## 1. System Overview

### 1.1 Purpose

Automated optical sorting of rough blue sapphire melee stones (1–3.5 mm) into four output bins based on color (Blue / Light / Inky) and clarity (Transparent / Translucent / Opaque). Replaces manual sorting which is slow, inconsistent, and fatiguing.

### 1.2 Key Performance Targets

| Metric | Target | Stretch |
|---|---|---|
| Throughput | 3,000 stones/hr | 5,000 stones/hr |
| Classification accuracy | 90% | 95% |
| Per-stone latency (capture → decision) | < 50 ms | < 30 ms |
| False CUT rate (non-premium in CUT bin) | < 5% | < 2% |
| Uptime per batch | > 95% | > 98% |

### 1.3 Classification Matrix

| | Transparent | Translucent | Opaque |
|---|---|---|---|
| **Blue** | **CUT** (if ≥ size threshold) / **SELECT** (if small) | FLIP | FLIP |
| **Light** | **SELECT** (heat treatment candidate) | FLIP | FLIP |
| **Inky** | **SELECT** | **SELECT** | **REJECT** |

Additional REJECT triggers: damaged stones, non-gem material, unclassifiable (low confidence on both axes).

### 1.4 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM OVERVIEW                              │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ VIBRATORY│    │  LINEAR  │    │ IMAGING  │    │  AIR JET     │  │
│  │  BOWL    │───▶│  CHANNEL │───▶│ STATION  │───▶│  SORTING     │  │
│  │ FEEDER   │    │ (V-groove)│    │          │    │  (4 bins)    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│                                       │                  │          │
│                                       ▼                  ▼          │
│                                 ┌──────────┐    ┌──────────────┐   │
│                                 │ COMPUTE  │    │  OUTPUT BINS │   │
│                                 │ (Jetson/ │    │ ┌───┬───┬───┐│   │
│                                 │  PC+GPU) │    │ │CUT│SEL│FLP││   │
│                                 │          │    │ ├───┼───┼───┤│   │
│                                 │ Camera   │    │ │   │   │REJ││   │
│                                 │ Classify │    │ └───┴───┴───┘│   │
│                                 │ Decide   │    └──────────────┘   │
│                                 │ Control  │                       │
│                                 └──────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.5 Data Flow

```
Hopper (bulk stones)
    │
    ▼
Vibratory Bowl Feeder ── singulates stones into single-file stream
    │
    ▼
Linear V-Groove Channel ── gravity or vibration-assisted, stones slide single-file
    │
    ▼
Imaging Station
    ├── Backlight ON → capture frame (clarity analysis: transparent/translucent/opaque)
    ├── Ring light ON → capture frame (color analysis: blue/light/inky)
    │   (or simultaneous with optical geometry separation)
    │
    ▼
Compute Pipeline (< 50 ms total)
    ├── Stone detection & segmentation
    ├── ROI extraction & normalization
    ├── CNN inference → (color_class, color_conf, clarity_class, clarity_conf)
    ├── Size estimation (pixels → mm via calibration)
    ├── Decision engine → CUT / SELECT / FLIP / REJECT
    │
    ▼
Sorting Controller
    ├── Track stone position (velocity × time since imaging)
    ├── Fire appropriate air jet nozzle at correct moment
    │
    ▼
Output Bins (4 channels)
    ├── Bin 1: CUT
    ├── Bin 2: SELECT
    ├── Bin 3: FLIP
    └── Bin 4: REJECT (default/no-blast trajectory)

Every stone → Data Logger (image, classification, confidence, decision, timestamp)
```

---

## 2. Hardware Architecture

### 2.1 Component Specifications

#### 2.1.1 Vibratory Bowl Feeder (振动盘)

| Parameter | Specification |
|---|---|
| Type | Spiral track vibratory bowl feeder |
| Bowl diameter | 200 mm |
| Part size range | 1–3.5 mm |
| Feed rate | Adjustable, target 1–1.5 stones/sec |
| Voltage | 220V AC (with controller) |
| Track design | Single spiral, narrowing to single-file at exit |
| Material | Stainless steel bowl (non-contaminating) |
| Search terms | `振动盘 200mm 小零件` / `微型振动盘送料器` |
| Price | ¥500–1,500 ($70–210) |

**Notes:** Custom track tooling may be needed for round rough stones. Request sample stones be sent to the 振动盘 manufacturer for track tuning. Rough sapphire is harder than steel (Mohs 9) so wear is on the feeder, not the stones.

#### 2.1.2 Industrial Camera

| Parameter | Specification |
|---|---|
| Recommended | HIKROBOT MV-CS050-10GC (5MP, color, GigE) or MV-CS050-10UC (USB3) |
| Sensor | Sony IMX264 or IMX547, 2/3" format |
| Resolution | 2448 × 2048 (5 MP) |
| Shutter | Global shutter (essential — stones are moving) |
| Frame rate | 24 fps @ 5MP, 75+ fps with ROI crop |
| Interface | GigE Vision or USB3 Vision |
| Pixel size | ~3.45 µm |
| SDK | MVS (Machine Vision Software) — Windows/Linux, Python bindings available |
| Search terms | `海康机器人 工业相机 500万 彩色 全局快门` |
| Price | ¥800–2,000 ($110–280) |

**Why 5 MP:** At 2:1 magnification with 3.45 µm pixels, a 1 mm stone spans ~580 pixels. Even at 1:1, a 1 mm stone is ~290 pixels — more than sufficient for color/clarity classification. We can use ROI cropping to boost frame rate to 75+ fps.

**Alternative:** If budget allows, MV-CH050-10GC (5MP, 75fps full frame) for higher throughput headroom.

#### 2.1.3 Macro Lens

| Parameter | Specification |
|---|---|
| Mount | C-mount |
| Magnification | 1:1 to 2:1 (adjustable or fixed) |
| Working distance | 40–100 mm (must clear lighting geometry) |
| Distortion | < 0.1% |
| Resolution | ≥ 120 lp/mm center |
| Type | Telecentric preferred (eliminates perspective distortion for size measurement) |
| Search terms | `工业远心镜头 C口 1:1 放大` or `工业微距镜头 C口 2倍` |
| Price | ¥300–800 ($42–110) standard macro; ¥800–2,000 ($110–280) telecentric |

**Recommendation:** Start with a standard 1:1 C-mount macro lens (e.g., VST VS-MC1-65 or comparable). Upgrade to telecentric only if size measurement accuracy under ±0.1 mm is needed.

#### 2.1.4 Lighting System

**Backlight (for clarity classification):**

| Parameter | Specification |
|---|---|
| Type | LED flat panel backlight, diffused white |
| Size | 50 × 50 mm active area (covers channel + margin) |
| Color | White (5500–6500K) |
| Mounting | Below the transparent section of channel |
| Purpose | Silhouette imaging — transparent stones pass light, opaque block it |
| Controller | Strobe-capable LED driver (sync with camera trigger) |
| Search terms | `机器视觉背光源 LED 50mm 白色` |
| Price | ¥100–300 ($14–42) |

**Ring Light (for color classification):**

| Parameter | Specification |
|---|---|
| Type | LED ring light, diffused white |
| Diameter | 40–60 mm inner diameter |
| Color | White (5500–6500K), high CRI ≥ 90 |
| Mounting | Around lens, coaxial with camera |
| Purpose | Even front illumination for color assessment |
| Controller | Strobe-capable, intensity-adjustable |
| Search terms | `机器视觉环形光源 LED 白色 高显指` |
| Price | ¥100–300 ($14–42) |

**Lighting controller:**

| Parameter | Specification |
|---|---|
| Type | Dual-channel strobe controller |
| Triggering | Camera trigger output → strobe sync |
| Search terms | `机器视觉光源控制器 频闪 双通道` |
| Price | ¥200–500 ($28–70) |

#### 2.1.5 Air Jet Sorting Mechanism

| Parameter | Specification |
|---|---|
| Solenoid valves | 4× high-speed solenoid valves (response < 5 ms) |
| Nozzle diameter | 1–2 mm (matched to stone size) |
| Air pressure | 0.2–0.5 MPa (adjustable) |
| Air supply | Small compressor or shop air line |
| Valve type | 2/2-way normally closed, 24V DC |
| Control | GPIO from compute board, with MOSFET driver |
| Physical layout | 4 nozzles at 0°, 30°, 60°, 90° deflection angles |
| Search terms | `高速电磁阀 2mm 分拣 喷气` / `微型气动分拣阀` |
| Price | ¥300–800 ($42–110) for 4-valve assembly |

**Sorting geometry:** Stones fall off the end of the channel under gravity. Default trajectory (no air blast) → REJECT bin. Three air jets deflect stones into CUT, SELECT, or FLIP bins at different angles. The fourth nozzle is spare/safety.

#### 2.1.6 Compute Platform

**Development:**

| Parameter | Specification |
|---|---|
| Platform | Windows 10/11 PC |
| GPU | NVIDIA RTX 3060 or better (for training + fast inference dev) |
| RAM | 16+ GB |
| Storage | 1 TB SSD (training images accumulate fast) |
| Camera interface | GigE (via Ethernet) or USB3 |

**Production:**

| Parameter | Specification |
|---|---|
| Platform | NVIDIA Jetson Orin Nano 8GB |
| Performance | 40 TOPS INT8 |
| Inference | TensorRT-optimized ONNX model |
| GPIO | 40-pin header for solenoid valve control |
| Interface | USB3 for camera, GPIO for valves, Ethernet for dashboard |
| Power | 7–15W |
| Search terms | `NVIDIA Jetson Orin Nano 开发套件` |
| Price | ¥1,500–3,000 ($210–420) |

#### 2.1.7 Channel / Chute

| Parameter | Specification |
|---|---|
| Material | Clear acrylic (at imaging station) + aluminum V-groove elsewhere |
| V-groove angle | 60–90° |
| Groove width | 4 mm (stones self-center in groove) |
| Imaging section | 30 mm of clear/transparent acrylic for backlight pass-through |
| Length | ~300 mm total (feeder exit → sorting point) |
| Slope | 10–20° (gravity-fed) or flat with vibration assist |
| Fabrication | CNC-machined or 3D-printed prototype, then CNC for production |
| Price | ¥100–300 ($14–42) materials + machining |

### 2.2 Physical Layout

```
SIDE VIEW (not to scale)
═══════════════════════════════════════════════════════════════

                    Camera + Lens
                        │
                    Ring Light ◯
                        │
                        ▼
    ┌─────┐       ┌───────────┐        ╲   Air Jets
    │Bowl │       │  Imaging  │         ╲  ↗ ↗ ↗
    │Feeder├──────┤  Station  ├──────────╲────────────
    │     │  ~~~  │(clear     │   ~~~~    ╲  Drop zone
    └─────┘  V-   │ acrylic)  │   V-groove ╲
             groove│           │             ╲
                   └─────┬─────┘              ╲
                         │                    ┌┴──┬──┬──┬──┐
                    ┌────┴────┐               │CUT│SEL│FLP│REJ│
                    │Backlight│               └───┴──┴──┴──┘
                    │(LED flat│                  Output Bins
                    │ panel)  │
                    └─────────┘

TOP VIEW
═══════════════════════════════════════════════════════════════

    ┌─────────┐     V-groove      Imaging       Drop
    │  Bowl   │    channel        Station        ↓
    │ Feeder  ├════════════════╤═══════════╤═══════╗
    │  (200mm │                │  ◯ Ring   │       ║→ CUT
    │  spiral)│                │  ◯ Light  │       ║→ SELECT
    └─────────┘                │  ◯ +Lens  │       ║→ FLIP
                               ╧═══════════╧       ║→ REJECT
                               Backlight below      ╚═══════
```

### 2.3 Wiring / Connectivity Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    WIRING DIAGRAM                             │
│                                                              │
│  ┌──────────┐                                                │
│  │ 220V AC  ├──┬──▶ Vibratory Feeder Controller             │
│  │ Mains    │  │                                             │
│  └──────────┘  ├──▶ 24V DC PSU ──┬──▶ Lighting Controller   │
│                │                  ├──▶ Solenoid Valves (×4)  │
│                │                  └──▶ Jetson Orin Nano (via │
│                │                       barrel jack adapter)  │
│                └──▶ Air Compressor                           │
│                                                              │
│  ┌──────────────┐    USB3/GigE    ┌───────────────┐         │
│  │  HIKROBOT    ├────────────────▶│  Jetson Orin  │         │
│  │  Camera      │                 │  Nano         │         │
│  └──────┬───────┘                 │               │         │
│         │ Trigger OUT             │  GPIO Header: │         │
│         ▼                         │  Pin 7  → Valve 1 (CUT)│
│  ┌──────────────┐                 │  Pin 11 → Valve 2 (SEL)│
│  │  Lighting    │◀── Strobe IN ───│  Pin 13 → Valve 3 (FLP)│
│  │  Controller  │                 │  Pin 15 → Valve 4 (spare│
│  │  (2-ch)      │                 │               │         │
│  └──────────────┘                 │  Ethernet → LAN/WiFi   │
│     Ch1: Backlight                │  (Dashboard access)     │
│     Ch2: Ring Light               └───────────────┘         │
│                                                              │
│  Solenoid Driver Board:                                      │
│  ┌────────────────────────────┐                              │
│  │ GPIO Pin → MOSFET gate     │                              │
│  │ 24V rail → MOSFET drain    │                              │
│  │ Solenoid → drain + flyback │                              │
│  │ diode across each coil     │                              │
│  └────────────────────────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.4 Lighting Geometry

```
CROSS-SECTION AT IMAGING STATION
═════════════════════════════════

            Camera
              │
         ┌────┴────┐
         │  Lens   │
         └────┬────┘        Working distance: 50-80mm
              │
        ╔═════╧═════╗
        ║ Ring Light ║      Ring light: coaxial, 40-60mm ID
        ╚═════╤═════╝      Provides diffuse front illumination
              │              for COLOR assessment
              │
    ──────────┼──────────   Channel surface (clear acrylic)
              ●              Stone (1-3.5mm)
    ──────────┼──────────   Channel bottom (clear acrylic)
              │
        ┌─────┴─────┐
        │ Backlight  │      Backlight: 50×50mm LED panel
        │ (LED flat  │      Directly below channel
        │  panel)    │      Provides transmitted illumination
        └───────────┘       for CLARITY assessment

IMAGING MODES:
  Mode A (Clarity):  Backlight ON, Ring light OFF
    → Transparent stones glow bright, opaque are dark silhouettes
    → Translucent stones show partial light transmission

  Mode B (Color):    Backlight OFF, Ring light ON
    → Reflected color under standardized white illumination
    → Blue vs Light vs Inky discrimination

  Mode C (Combined): Both ON simultaneously (if sufficient contrast)
    → Single capture, software separates features
    → Faster but may reduce classification accuracy

  Recommended: Start with Mode C (single capture). If accuracy
  insufficient, switch to A+B sequential (two captures per stone,
  ~10ms apart with strobe sync). At 1 stone/sec this is fine.
```

### 2.5 Calibration Requirements

- **Spatial calibration:** Known-size reference object (e.g., 2.0 mm steel ball bearing) to establish pixels-per-mm ratio. Re-calibrate if lens or camera position changes.
- **Color calibration:** X-Rite ColorChecker Nano or similar. Capture reference under ring light at startup of each batch. Software applies white-balance correction matrix.
- **Lighting intensity:** Consistent backlight and ring light intensity. Monitor via reference gray patch at edge of FOV. Alert if drift > 5%.
- **Timing calibration:** Drop a test stone, measure transit time from imaging station to sorting point. Calibrate air jet firing delay.

---

## 3. Software Architecture

### 3.1 Full Stack Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     DASHBOARD (Browser)                      │
│  React/Vue SPA or simple Jinja2 templates                   │
│  Real-time stats │ Batch summary │ Manual override │ Config │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/WebSocket
┌──────────────────────────┴──────────────────────────────────┐
│                   WEB SERVER (FastAPI)                        │
│  REST API │ WebSocket stream │ Config endpoints              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                 APPLICATION CORE (Python)                     │
│                                                              │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │   Image    │  │  Preprocess  │  │  Classification   │   │
│  │ Acquisition│─▶│  Pipeline    │─▶│  Engine           │   │
│  │  Module    │  │              │  │  (CNN inference)   │   │
│  └────────────┘  └──────────────┘  └─────────┬─────────┘   │
│                                               │              │
│  ┌────────────┐  ┌──────────────┐  ┌─────────▼─────────┐   │
│  │  Sorting   │◀─│  Decision    │◀─│  Size Estimator   │   │
│  │ Controller │  │  Engine      │  │  (pixel→mm)       │   │
│  │ (GPIO/air) │  │  (rules)     │  └───────────────────┘   │
│  └────────────┘  └──────────────┘                           │
│                                                              │
│  ┌────────────┐  ┌──────────────┐                           │
│  │   Data     │  │  Calibration │                           │
│  │  Logger    │  │  Manager     │                           │
│  └────────────┘  └──────────────┘                           │
└──────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                    HARDWARE LAYER                             │
│  HIKROBOT MVS SDK │ GPIO (Jetson.GPIO / RPi.GPIO) │ SQLite │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Module Specifications

#### 3.2.a Image Acquisition Module

**Responsibility:** Interface with HIKROBOT camera, trigger captures, deliver frames to pipeline.

```python
# Key interfaces
class ImageAcquisition:
    def initialize(camera_id: str, exposure_us: int, gain: float) -> None
    def set_roi(x, y, width, height) -> None  # For higher frame rates
    def trigger_capture() -> Frame  # Software or hardware trigger
    def continuous_stream(callback: Callable[[Frame], None]) -> None
    def shutdown() -> None
```

| Parameter | Value |
|---|---|
| SDK | HIKROBOT MVS (MvCameraControl) |
| Trigger mode | Hardware trigger (from stone detection sensor) or continuous + software stone detection |
| Exposure | 500–5,000 µs (tuned to lighting) |
| Gain | Auto or fixed after calibration |
| Color space | BGR (OpenCV native) → convert to LAB for color analysis |
| Frame format | 8-bit color, 2448×2048 or ROI-cropped |
| Trigger latency | < 1 ms (hardware), < 5 ms (software) |

**Stone detection trigger options:**
1. **Optical sensor:** IR break-beam across channel, 5 mm upstream of imaging zone. Simplest, most reliable.
2. **Software detection:** Continuous capture at 75 fps, detect stone entering FOV via frame differencing. No extra hardware but uses more compute.
3. **Recommended:** Start with software detection (Phase 1–2), add optical trigger for production (Phase 3+).

#### 3.2.b Preprocessing Pipeline

**Responsibility:** From raw frame → isolated, normalized stone image ready for classification.

```
Raw Frame (2448×2048 or ROI)
    │
    ▼
Background Subtraction
    │  Compare to reference "empty channel" image
    │  Threshold to binary mask
    ▼
Stone Detection / Segmentation
    │  Find contours in mask
    │  Filter by area (min/max for 1-3.5mm stones)
    │  Reject overlapping/touching stones (flag for re-sort)
    ▼
ROI Extraction
    │  Bounding box + margin around each stone
    │  Resize to fixed input size (e.g., 128×128 or 224×224)
    ▼
Color Normalization
    │  White balance correction (from calibration matrix)
    │  Convert BGR → LAB color space
    │  Histogram normalization for lighting consistency
    ▼
Feature Extraction (optional, for classical ML fallback)
    │  Mean L, A, B values
    │  Transmittance ratio (backlight intensity through stone / reference)
    │  Texture metrics (variance, entropy)
    ▼
Normalized Stone Image → Classification Engine
```

**Key parameters:**
- Background frame: captured at startup and periodically refreshed
- Minimum contour area: ~100 px² (at 1:1 mag, 1mm stone ≈ 290px diameter ≈ 66,000 px² — so threshold is generous)
- Maximum contour area: ~500,000 px² (reject large debris)
- Stone aspect ratio filter: 0.5–2.0 (rough stones are irregular but not extremely elongated)
- Overlap detection: if two contours within 10px, flag as potential overlap → slow feeder or skip

#### 3.2.c Classification Engine

**Responsibility:** Given a normalized stone image, output color class + clarity class with confidence scores.

**Recommended Model Architecture: MobileNetV3-Small**

| Rationale | Detail |
|---|---|
| Why MobileNetV3-Small | Designed for edge/mobile deployment. ~2.5M params. Runs at 5+ ms on Jetson Orin Nano with TensorRT. Accuracy sufficient for 9-class problem with good training data. |
| Why not EfficientNet-Lite | Slightly heavier, marginal accuracy benefit for this simple classification task. MobileNetV3-Small is the sweet spot for speed + accuracy on Jetson. |
| Why not a simpler model | ResNet18 works too but is 11M params — overkill. Custom tiny CNN could work but MobileNetV3 has proven training recipes. |
| Fallback option | If CNN accuracy plateaus, try EfficientNet-Lite0 (5.3M params, ~8ms on Jetson). |

**Model specification:**

```
Input:  128 × 128 × 3 (RGB, normalized to [0,1])
Backbone: MobileNetV3-Small (pretrained on ImageNet, fine-tuned)
Head: Two separate classification heads (multi-task learning)

         MobileNetV3-Small Backbone
                   │
            Global Average Pool
                   │
              FC (576 → 256)
              ReLU + Dropout(0.3)
                   │
          ┌────────┴────────┐
          │                 │
     FC (256→3)        FC (256→3)
     Softmax            Softmax
          │                 │
     Color class       Clarity class
     [Blue,Light,      [Transparent,
      Inky]             Translucent,
                        Opaque]
```

**Multi-task training:** Single backbone, two heads. Joint loss = CrossEntropy(color) + CrossEntropy(clarity). This is more efficient and shares feature learning.

**Inference output:**
```python
{
    "color": {"blue": 0.85, "light": 0.10, "inky": 0.05},
    "clarity": {"transparent": 0.72, "translucent": 0.25, "opaque": 0.03},
    "size_mm": 2.1,  # from pixel measurement
    "confidence": 0.85  # min(max_color_prob, max_clarity_prob)
}
```

**Confidence thresholds:**
- High confidence: both heads > 0.75 → auto-sort
- Medium confidence: either head 0.50–0.75 → sort but flag for review
- Low confidence: either head < 0.50 → divert to SELECT bin for human review

**Inference pipeline (Jetson Orin Nano with TensorRT):**

| Stage | Time |
|---|---|
| Image preprocessing (resize, normalize) | 1–2 ms |
| CNN inference (MobileNetV3-Small, TensorRT FP16) | 3–5 ms |
| Decision engine | < 1 ms |
| **Total** | **5–8 ms** |

At 1 stone/second, this leaves 992+ ms of headroom. At 5,000/hr (1.4 stones/sec, ~710 ms/stone), still abundant margin.

**Export pipeline:** PyTorch → ONNX → TensorRT engine (.trt). Script provided for one-command conversion.

#### 3.2.d Decision Engine

**Responsibility:** Map classification output to sorting action. Fully configurable via rules file.

```yaml
# decision_rules.yaml
size_threshold_mm: 2.0  # Below this, Blue+Transparent → SELECT instead of CUT

confidence_threshold:
  auto_sort: 0.75      # Both heads above this → trust classification
  human_review: 0.50   # Below this → SELECT bin for review

rules:
  # Format: [color, clarity] → action
  # Size override for CUT is handled in code
  
  - color: blue
    clarity: transparent
    size: ">= threshold"
    action: CUT
  
  - color: blue
    clarity: transparent
    size: "< threshold"
    action: SELECT
    reason: "small_blue_transparent"
  
  - color: blue
    clarity: translucent
    action: FLIP
  
  - color: blue
    clarity: opaque
    action: FLIP
  
  - color: light
    clarity: transparent
    action: SELECT
    reason: "heat_treatment_candidate"
  
  - color: light
    clarity: translucent
    action: FLIP
  
  - color: light
    clarity: opaque
    action: FLIP
  
  - color: inky
    clarity: transparent
    action: SELECT
    reason: "inky_transparent_review"
  
  - color: inky
    clarity: translucent
    action: SELECT
    reason: "inky_translucent_review"
  
  - color: inky
    clarity: opaque
    action: REJECT

  # Fallback rules
  - low_confidence: true
    action: SELECT
    reason: "low_confidence"
  
  - unclassifiable: true
    action: REJECT
    reason: "unclassifiable"
```

**Implementation:** Simple rule engine, ~50 lines of Python. Rules loaded from YAML, hot-reloadable without restart.

#### 3.2.e Sorting Controller

**Responsibility:** Fire the correct air jet at the correct time to deflect stone into target bin.

```
TIMING DIAGRAM
══════════════

Stone passes imaging station → t₀
    │
    │ Processing time (5-8 ms)
    │ Decision ready → t₁ = t₀ + ~8ms
    │
    │ Stone travels from imaging station to sort point
    │ Distance: D mm, velocity: v mm/s
    │ Transit time: T = D/v
    │
    │ Fire air jet at t₂ = t₀ + T - (valve_response_time / 2)
    │ Valve response: ~3-5 ms
    │ Air blast duration: 5-15 ms (tunable)
    │
    ▼
Stone deflected into target bin
```

**Key parameters (to be calibrated):**

| Parameter | Estimated Value | Calibration Method |
|---|---|---|
| Imaging-to-sort distance | 50–100 mm | Physical measurement |
| Stone velocity | 100–500 mm/s | High-speed capture or optical gate pair |
| Transit time | 100–1000 ms | Calculated from above |
| Valve response time | 3–5 ms | Datasheet + verification |
| Air blast duration | 5–15 ms | Tuned per bin distance |
| Air pressure | 0.2–0.4 MPa | Tuned per stone size |

**GPIO control (Jetson):**

```python
import Jetson.GPIO as GPIO

VALVE_PINS = {
    "CUT": 7,
    "SELECT": 11,
    "FLIP": 13,
    "SPARE": 15
}

class SortingController:
    def fire_valve(self, bin_name: str, duration_ms: float):
        pin = VALVE_PINS[bin_name]
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration_ms / 1000)
        GPIO.output(pin, GPIO.LOW)
    
    def schedule_sort(self, bin_name: str, delay_ms: float, duration_ms: float):
        # Non-blocking timer-based firing
        timer = threading.Timer(delay_ms / 1000, self.fire_valve, args=[bin_name, duration_ms])
        timer.start()
```

**Note:** For production, use hardware timers or a real-time co-processor (e.g., Arduino/STM32) for sub-millisecond timing precision. The Jetson's Linux kernel is not real-time, so jitter of 1–5 ms is expected. Given stone transit times of 100+ ms, this jitter is acceptable.

**REJECT bin = default trajectory.** No air blast needed. If system fails or loses tracking, stones fall into REJECT. This is the safe failure mode.

#### 3.2.f Data Logger

**Responsibility:** Record every stone for traceability, training data collection, and QA.

**Storage per stone:**
- Thumbnail image: 128×128 JPEG ≈ 5 KB
- Full ROI image: ~50 KB (saved for low-confidence stones only)
- Metadata: ~200 bytes JSON

**At 5,000 stones/hour:** ~25 MB/hr thumbnails, ~250 MB/hr if saving all full ROIs. Manageable on any modern storage.

**Database: SQLite** (simple, no server, portable)

```sql
CREATE TABLE stones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601
    batch_id TEXT,                     -- User-assigned batch name
    image_path TEXT NOT NULL,          -- Path to thumbnail
    full_image_path TEXT,              -- Path to full ROI (if saved)
    color_class TEXT,                  -- blue/light/inky
    color_confidence REAL,
    clarity_class TEXT,                -- transparent/translucent/opaque
    clarity_confidence REAL,
    size_mm REAL,
    decision TEXT,                     -- CUT/SELECT/FLIP/REJECT
    decision_reason TEXT,              -- Rule that triggered
    overall_confidence REAL,
    flagged_for_review BOOLEAN DEFAULT FALSE,
    human_label_color TEXT,            -- Filled in during review
    human_label_clarity TEXT,          -- Filled in during review
    notes TEXT
);

CREATE INDEX idx_batch ON stones(batch_id);
CREATE INDEX idx_decision ON stones(decision);
CREATE INDEX idx_flagged ON stones(flagged_for_review);
```

#### 3.2.g Dashboard / UI

**Tech:** FastAPI backend + simple HTML/JS frontend (or Streamlit for rapid prototyping).

**Screens:**

1. **Live View**
   - Camera feed with stone detection overlay
   - Last N stones: thumbnail + classification + decision
   - Current throughput (stones/min)
   - Running accuracy estimate (if human spot-checking)

2. **Batch Summary**
   - Distribution pie charts: color classes, clarity classes, decisions
   - Total counts per bin
   - Confidence histogram
   - Low-confidence flagged stones for review

3. **Review Queue**
   - Grid of flagged stone images
   - Click to assign human label (color + clarity)
   - Labels saved to database → available for retraining

4. **Calibration Mode**
   - Live camera view with exposure/gain controls
   - Color checker capture and white balance
   - Backlight/ring light intensity adjustment
   - Test sort (fire individual valves)

5. **Configuration**
   - Decision rules editor (YAML)
   - Confidence thresholds
   - Feeder speed
   - Air jet timing parameters

**Access:** Web browser on local network. Jetson serves on port 8080.

### 3.3 Tech Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Camera SDK | HIKROBOT MVS (MvCameraControl Python wrapper) |
| Image processing | OpenCV 4.x, NumPy |
| Deep learning (training) | PyTorch 2.x |
| Deep learning (inference) | ONNX Runtime or TensorRT (Jetson) |
| Web framework | FastAPI |
| Frontend | Jinja2 templates + HTMX (simple) or Vue.js (richer) |
| Database | SQLite3 |
| GPIO | Jetson.GPIO |
| Configuration | YAML (PyYAML) |
| Logging | Python logging + structured JSON logs |
| Process management | systemd service on Jetson |

### 3.4 Inference Pipeline Timing Budget

```
Target: < 50 ms end-to-end per stone (from detection to valve fire command)

┌─────────────────────────────────────────────────┐
│ Stage                          │ Budget (ms)     │
├────────────────────────────────┼─────────────────┤
│ Stone detection (in frame)     │ 2               │
│ ROI extraction + resize        │ 1               │
│ Color normalization            │ 1               │
│ CNN inference (TensorRT FP16)  │ 5               │
│ Decision engine                │ 0.5             │
│ Valve scheduling               │ 0.5             │
├────────────────────────────────┼─────────────────┤
│ TOTAL PROCESSING               │ ~10 ms          │
│ Available margin               │ 40 ms           │
│ (used for stone transit time)  │                 │
└────────────────────────────────┴─────────────────┘

Stone transit time (imaging → sort point) ≈ 100-500 ms
→ Processing completes well before stone reaches sort point
→ System is NOT latency-constrained at target throughputs
```

---

## 4. Training Data Strategy

### 4.1 Data Collection

Use the same hardware (camera + lighting + channel) in **Phase 1** to capture training images. No sorting mechanism needed yet.

**Collection workflow:**
1. Set up imaging station with feeder running
2. Run continuous capture, saving every detected stone image
3. Run stones through multiple times if needed to build volume
4. Aim for diverse lighting conditions within calibrated range

**Capture both lighting modes per stone** (if using sequential mode):
- Backlight image → for clarity labeling
- Ring light image → for color labeling
- Both as a pair with shared ID

### 4.2 Annotation Workflow

**Labels per stone (multi-label):**
1. **Color:** Blue / Light / Inky
2. **Clarity:** Transparent / Translucent / Opaque

**Annotation tool:** Simple custom web UI (or use Label Studio, open source).

**Workflow:**
1. Display stone image (both lighting modes if available)
2. Annotator selects color class (3 buttons)
3. Annotator selects clarity class (3 buttons)
4. Optional: mark as "uncertain" or "damaged"
5. Next stone

**At ~2 seconds per stone, one person annotates ~1,800 stones/hour.** A kilo of melee contains ~3,000–10,000 stones depending on size distribution, so 2–6 hours to annotate a kilo.

**Annotator:** Duke himself for the first few hundred (establishes ground truth standard), then can train an assistant using Duke's reference examples.

### 4.3 Data Augmentation

| Augmentation | Rationale |
|---|---|
| Random rotation (0–360°) | Stones have no canonical orientation |
| Random flip (H, V) | Same reason |
| Brightness ±15% | Simulate lighting variation |
| Contrast ±10% | Simulate lighting variation |
| Gaussian noise (σ=5) | Simulate sensor noise |
| Random crop + resize | Simulate slight position variation |
| Color jitter (hue ±5°, saturation ±10%) | Small — don't want to change the class! |

**DO NOT** apply heavy color augmentation — color IS the classification signal. Keep color jitter subtle.

### 4.4 Minimum Dataset Size Estimates

| Class (Color × Clarity) | Min Samples | Target Samples | Notes |
|---|---|---|---|
| Blue + Transparent | 200 | 500+ | Most important class (CUT decisions) |
| Blue + Translucent | 150 | 400 | Common |
| Blue + Opaque | 100 | 300 | Less common |
| Light + Transparent | 150 | 400 | Heat treatment candidates |
| Light + Translucent | 150 | 400 | Common in Thai lots |
| Light + Opaque | 100 | 300 | |
| Inky + Transparent | 100 | 300 | Less common |
| Inky + Translucent | 100 | 300 | |
| Inky + Opaque | 100 | 300 | |
| **TOTAL** | **~1,150** | **~3,200** | |

**With augmentation** (10× per image), effective training set: 11,500–32,000 images. Sufficient for fine-tuning MobileNetV3-Small.

**Recommendation:** Collect at least 3,000 labeled stones before training. This is roughly 1 kilo of mixed melee — one afternoon of sorting plus one evening of labeling.

### 4.5 Active Learning Loop

```
                    ┌──────────────────┐
                    │ Production Sort  │
                    └────────┬─────────┘
                             │
                    Stone classified with
                    confidence < threshold
                             │
                             ▼
                    ┌──────────────────┐
                    │ Flagged for      │
                    │ Human Review     │
                    │ (SELECT bin or   │
                    │  dashboard queue)│
                    └────────┬─────────┘
                             │
                    Human provides
                    correct label
                             │
                             ▼
                    ┌──────────────────┐
                    │ Added to         │
                    │ Training Dataset │
                    └────────┬─────────┘
                             │
                    Periodic retraining
                    (weekly or after N
                    new samples)
                             │
                             ▼
                    ┌──────────────────┐
                    │ Updated Model    │
                    │ Deployed         │
                    └──────────────────┘
```

**Trigger for retraining:** Every 500 new human-labeled samples, or weekly, whichever comes first.

**Model versioning:** Keep all model versions. A/B test new model on 100 stones against previous before full deployment.

---

## 5. Calibration & Quality Assurance

### 5.1 Color Calibration

**Startup protocol (run at beginning of each batch):**
1. Place X-Rite ColorChecker Nano (or printed reference card) in imaging zone
2. Capture reference frame under ring light
3. Software computes white balance correction matrix by comparing known patch colors to captured values
4. Apply correction to all subsequent frames

**Ongoing monitoring:**
- Reserve a small area at edge of FOV for a fixed gray reference patch (glued to channel wall)
- Monitor patch color every 100 frames
- Alert if drift exceeds threshold (indicates bulb aging, dust on lens, etc.)

### 5.2 Lighting Consistency

- LED drivers should be constant-current (not voltage-regulated) for stable output
- Monitor backlight intensity via reference area in each frame
- Replace LEDs proactively (every 10,000 hours or at first sign of drift)
- Keep spare LED panels and ring lights on hand

### 5.3 Model Accuracy Monitoring

**Per-batch QA protocol:**
1. Before each production batch, run 50 known reference stones (pre-labeled by Duke)
2. Compare machine classifications to known labels
3. Require ≥ 90% agreement to proceed
4. If below threshold: re-calibrate lighting, check for mechanical issues, or retrain model

**Continuous monitoring:**
- Track confidence score distribution over time
- Alert if mean confidence drops (model encountering unfamiliar stones)
- Track per-class accuracy if human spot-checks are available

### 5.4 Drift Detection

| Indicator | Normal Range | Alert Threshold |
|---|---|---|
| Mean color confidence | > 0.80 | < 0.70 |
| Mean clarity confidence | > 0.80 | < 0.70 |
| % stones flagged for review | < 10% | > 20% |
| Backlight reference intensity | ±5% of baseline | ±10% |
| Ring light reference color | ΔE < 2.0 | ΔE > 5.0 |

### 5.5 Reference Stone Kit

Maintain a set of 50 reference stones:
- ~6 per class (9 classes × 6 = 54)
- Stored in labeled compartments
- Photographed and labeled by Duke
- Used for QA validation at start of each session
- Replace if damaged or lost

---

## 6. Phased Build Plan

### Phase 1: Image Capture Station (Weeks 1–4)

**Goal:** Assemble hardware, capture training data.

**Tasks:**
- [ ] Source and receive all components (feeder, camera, lens, lights, channel)
- [ ] Assemble mechanical structure (feeder → channel → imaging station)
- [ ] Install camera + lens, connect to dev PC
- [ ] Install and test HIKROBOT MVS SDK
- [ ] Set up lighting (backlight + ring light) and strobe controller
- [ ] Calibrate: focus, exposure, white balance, pixels-per-mm
- [ ] Write image acquisition script (continuous capture + stone detection)
- [ ] Run 1+ kilos of stones through, capturing images
- [ ] Build simple annotation tool
- [ ] Annotate 3,000+ stone images (Duke + assistant)

**Deliverable:** Annotated dataset of 3,000+ stones, calibrated imaging station.

**Timeline:** 4 weeks (2 weeks for hardware assembly + calibration, 2 weeks for data collection + annotation).

### Phase 2: Model Training (Weeks 5–7)

**Goal:** Train and validate classification model offline.

**Tasks:**
- [ ] Set up PyTorch training environment on dev PC
- [ ] Implement data loading, augmentation pipeline
- [ ] Train MobileNetV3-Small with dual classification heads
- [ ] Evaluate: per-class accuracy, confusion matrix, confidence calibration
- [ ] Iterate: adjust augmentation, class weights, architecture if needed
- [ ] Export to ONNX
- [ ] Test ONNX inference speed on dev PC
- [ ] If Jetson available: convert to TensorRT, benchmark

**Target:** ≥ 90% accuracy on held-out test set (10% of data).

**Deliverable:** Trained model (.onnx + .trt), training report with metrics.

**Timeline:** 3 weeks.

### Phase 3: Real-Time Sorting Integration (Weeks 8–11)

**Goal:** Stones go in one end, sorted stones come out the other.

**Tasks:**
- [ ] Build sorting mechanism (air jets, valves, bins, chute extension)
- [ ] Wire solenoid driver board to Jetson GPIO
- [ ] Implement real-time pipeline: capture → preprocess → classify → decide → sort
- [ ] Calibrate sorting timing (stone velocity, valve delay)
- [ ] Test with reference stones: measure actual sorting accuracy
- [ ] Iterate on timing, air pressure, nozzle position
- [ ] Stress test at target throughput (3,000 stones/hr)

**Deliverable:** Working sorting machine, demonstrated 90%+ accuracy on test batch.

**Timeline:** 4 weeks.

### Phase 4: Dashboard + Production Optimization (Weeks 12–14)

**Goal:** Production-ready system with logging, monitoring, and UI.

**Tasks:**
- [ ] Implement FastAPI server + dashboard
- [ ] Implement data logger (SQLite + image storage)
- [ ] Build review queue for flagged stones
- [ ] Implement batch management (start/stop, naming, summary reports)
- [ ] Set up active learning pipeline (review → retrain trigger)
- [ ] Deploy as systemd service on Jetson (auto-start on boot)
- [ ] Production testing with real batches
- [ ] Document operating procedures

**Deliverable:** Production-ready system with dashboard, logging, and operator manual.

**Timeline:** 3 weeks.

### Dependency Chart

```
Week:  1   2   3   4   5   6   7   8   9   10  11  12  13  14
       ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤

Phase 1: Hardware + Data Collection
       ████████████████
       [order] [build] [capture] [annotate]

Phase 2: Model Training
                       ████████████
                       [train] [eval] [export]

Phase 3: Sorting Integration
                                   ████████████████
                                   [mech] [wire] [calibrate] [test]

Phase 4: Dashboard + Production
                                                   ████████████
                                                   [UI] [log] [deploy]

Critical path: Phase 1 → Phase 2 → Phase 3
Phase 4 can partially overlap with Phase 3.
Jetson can be set up in parallel during Phase 2.
```

**Total estimated timeline: 14 weeks (3.5 months)**

---

## 7. Bill of Materials

### 7.1 Complete BOM

| # | Component | Specification | Search Terms (1688/淘宝) | Qty | Unit Price (¥) | Unit Price ($) | Total (¥) | Total ($) |
|---|---|---|---|---|---|---|---|---|
| 1 | Vibratory bowl feeder | 200mm, adjustable, 1-3.5mm parts | `振动盘 200mm 微型零件 送料器` | 1 | 800 | 112 | 800 | 112 |
| 2 | Feeder controller | Variable frequency, 220V | (usually included with feeder) | 1 | 0 | 0 | 0 | 0 |
| 3 | Industrial camera | HIKROBOT MV-CS050-10UC, 5MP color, USB3, global shutter | `海康机器人 MV-CS050 工业相机 500万 USB3` | 1 | 1,200 | 168 | 1,200 | 168 |
| 4 | Macro lens | C-mount, 1:1 magnification, 50mm WD | `工业微距镜头 C口 1倍 放大镜头` | 1 | 500 | 70 | 500 | 70 |
| 5 | LED backlight | 50×50mm, white, machine vision | `机器视觉 背光源 50mm LED 白色` | 1 | 180 | 25 | 180 | 25 |
| 6 | LED ring light | 50mm ID, white, high CRI, diffused | `机器视觉 环形光源 50mm LED 白色 高显指` | 1 | 200 | 28 | 200 | 28 |
| 7 | Lighting controller | 2-channel, strobe-capable, 24V | `机器视觉 光源控制器 2路 频闪` | 1 | 350 | 49 | 350 | 49 |
| 8 | Solenoid valves | High-speed 2/2-way, 24V DC, < 5ms response | `高速电磁阀 24V 微型 2mm 常闭` | 4 | 80 | 11 | 320 | 45 |
| 9 | Air nozzles | 1-2mm precision nozzles, brass | `精密气嘴 1mm 黄铜` | 4 | 15 | 2 | 60 | 8 |
| 10 | Pneumatic fittings | Push-in fittings, 4mm tubing | `气动快插接头 4mm` | 1 set | 50 | 7 | 50 | 7 |
| 11 | Air tubing | 4mm OD PU tubing, 5m | `PU气管 4mm 5米` | 1 | 20 | 3 | 20 | 3 |
| 12 | Air pressure regulator | With gauge, 0-0.6 MPa | `气源处理器 调压阀 带表` | 1 | 60 | 8 | 60 | 8 |
| 13 | Small air compressor | Oil-free, quiet, 0.5 MPa | `静音无油空压机 小型` (or use shop air) | 1 | 500 | 70 | 500 | 70 |
| 14 | NVIDIA Jetson Orin Nano | 8GB developer kit | `NVIDIA Jetson Orin Nano 开发套件 8GB` | 1 | 2,000 | 280 | 2,000 | 280 |
| 15 | MOSFET driver board | 4-channel, 24V, opto-isolated | `4路 MOS管 驱动模块 24V 光耦隔离` | 1 | 30 | 4 | 30 | 4 |
| 16 | 24V DC power supply | 120W, switching | `24V 5A 开关电源` | 1 | 40 | 6 | 40 | 6 |
| 17 | V-groove channel material | Clear acrylic sheet + aluminum bar stock | `透明亚克力板 3mm` + `铝合金型材` | 1 set | 100 | 14 | 100 | 14 |
| 18 | Mounting hardware | 80/20 aluminum extrusion or optical breadboard, brackets | `2020铝型材 套件` or `光学面包板` | 1 set | 300 | 42 | 300 | 42 |
| 19 | Sorting bins | 4× small collection containers | `零件盒 小号` | 4 | 10 | 1 | 40 | 6 |
| 20 | USB3 cable | 3m, active, for camera | `USB3.0 数据线 3米 带信号放大` | 1 | 40 | 6 | 40 | 6 |
| 21 | Ethernet cable | Cat6, 2m (Jetson to LAN) | `六类网线 2米` | 1 | 15 | 2 | 15 | 2 |
| 22 | Color reference card | X-Rite ColorChecker Nano or printed substitute | `色卡 标准色板` (or order X-Rite online) | 1 | 150 | 21 | 150 | 21 |
| 23 | Misc wiring | Jumper wires, connectors, wire terminals | `杜邦线 面包板线 端子` | 1 set | 50 | 7 | 50 | 7 |
| 24 | Enclosure/dust cover | Acrylic or sheet metal, light-tight around imaging zone | `亚克力板 黑色 遮光` | 1 set | 100 | 14 | 100 | 14 |

### 7.2 Budget Summary

| Category | Cost (¥) | Cost ($) |
|---|---|---|
| Feeding system (#1-2) | 800 | 112 |
| Imaging system (#3-7) | 2,430 | 340 |
| Sorting mechanism (#8-13) | 1,010 | 141 |
| Compute (#14) | 2,000 | 280 |
| Electrical (#15-16, #23) | 120 | 17 |
| Mechanical (#17-19, #24) | 540 | 76 |
| Cables & accessories (#20-22) | 205 | 29 |
| **SUBTOTAL** | **7,105** | **995** |
| Contingency (20%) | 1,421 | 199 |
| **TOTAL** | **~8,500** | **~1,200** |

**Note:** Does not include the Windows dev PC (assumed existing) or CNC machining costs for custom channel parts (estimate ¥200-500 / $28-70).

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Stone jamming in channel** | High | Medium | V-groove geometry wider than max stone size. Add vibration assist to channel. Jam detection via camera (no stone movement for N seconds → alert). |
| **Overlapping/touching stones** | High | Medium | Feeder track design should singulate. Software detection of multi-stone blobs → skip and recirculate. Lower feed rate if overlap rate > 5%. |
| **Dust and debris** | High | Low | Enclosed imaging zone. Periodic compressed air cleaning. Pre-wash stones if extremely dusty lots. Background subtraction handles minor debris. |
| **Inconsistent stone orientation** | Medium | Medium | Rough stones are irregular — train model on all orientations. Heavy rotation augmentation. Some stones may look different from different angles → accept ~5% inconsistency as inherent. |
| **Wet stones (post-wash)** | Medium | Medium | Water changes optical properties — translucent may appear transparent. Either sort dry only, or train separate wet/dry models. Recommend: air-dry stones before sorting. |
| **Model accuracy below 90%** | Medium | High | Collect more training data, especially for underperforming classes. Try larger model (EfficientNet-Lite0). Add classical features (transmittance ratio) as auxiliary input. Worst case: run two passes. |
| **Color classes not cleanly separable** | Medium | High | Blue/Light/Inky boundaries are subjective. Use Duke's ground truth as the standard. Accept that boundary cases exist → these naturally fall to SELECT bin via low confidence. |
| **Backlight not distinguishing translucent from transparent** | Medium | Medium | Increase backlight intensity. Quantify transmittance as continuous value, not just visual. May need to combine transmitted + reflected features. Experiment in Phase 1. |
| **Air jet not reliably deflecting smallest stones (1mm)** | Medium | High | 1mm stones are very light (~0.005g). Even gentle air can over-deflect. Tune pressure per size class. Use smaller nozzle (0.5mm). May need separate calibration for < 1.5mm stones. |
| **Jetson GPIO timing jitter** | Low | Low | Linux is not real-time, expect 1-5ms jitter. Stone transit time is 100+ ms, so this is < 5% error. If problematic, add Arduino co-processor for valve timing. |
| **Camera frame rate bottleneck** | Low | Medium | At 5,000 stones/hr = 1.4/sec, we need < 2 fps. Even at 5MP full frame, camera does 24 fps. Not a bottleneck. ROI mode gives 75+ fps. |
| **Model drift over time** | Low | Medium | Active learning loop catches this. Monthly QA with reference stones. Track confidence metrics. |

### 8.2 Melee-Specific Challenges

**Why melee stones (1–3.5 mm) are harder than larger stones:**

1. **Size relative to dust/debris:** A 1mm stone is only 5× larger than a large dust particle. Rigorous cleaning needed.
2. **Difficult to handle individually:** Standard tweezers/vacuum pens don't work well. Vibratory feeding is the right approach.
3. **Less visual information per stone:** Fewer pixels to classify from, even with macro lens. This is why 5MP + 1:1 magnification is important (290+ pixels per mm).
4. **Electrostatic attraction:** Tiny stones can stick to channel walls, each other, or accumulate static charge. Use anti-static treatment on acrylic channel. Ground metal components.
5. **Air jet precision:** Small mass means high sensitivity to air pressure. Careful calibration needed. Stones may tumble unpredictably in air.
6. **High volume, low per-unit value:** At $0.01–0.28/stone for cutting cost, the machine must be cheap and reliable. The ¥8,500 ($1,200) BOM is appropriate — pays for itself after sorting ~5,000 CUT-grade stones that would otherwise be missed or mis-sorted.

### 8.3 Business Risks

| Risk | Mitigation |
|---|---|
| Sorting quality worse than manual | Run parallel: machine sort + human verification of CUT bin for first month. SELECT bin acts as safety net. |
| Machine too slow to be worthwhile | Even at 3,000/hr it beats manual sorting by 5-10×. Throughput can be improved later with faster feeder + dual-lane. |
| Training data bias (one lot doesn't represent all) | Collect data from multiple lots over time. Active learning adds diversity naturally. |
| Component sourcing delays (China domestic) | Order all components in Week 1. Most items ship in 3-7 days on 1688/淘宝. Camera may take 1-2 weeks. |

---

## Appendix A: Software Repository Structure

```
sapphire-sorter/
├── README.md
├── requirements.txt
├── config/
│   ├── camera.yaml          # Camera settings
│   ├── calibration.yaml     # Current calibration values
│   ├── decision_rules.yaml  # Sorting rules
│   └── system.yaml          # General settings
├── src/
│   ├── acquisition/
│   │   ├── camera.py        # HIKROBOT SDK wrapper
│   │   └── trigger.py       # Stone detection trigger
│   ├── preprocessing/
│   │   ├── background.py    # Background subtraction
│   │   ├── segmentation.py  # Stone detection + ROI
│   │   └── normalize.py     # Color normalization
│   ├── classification/
│   │   ├── model.py         # Model loading + inference
│   │   ├── train.py         # Training script
│   │   └── export.py        # PyTorch → ONNX → TensorRT
│   ├── decision/
│   │   └── engine.py        # Rule-based decision engine
│   ├── sorting/
│   │   └── controller.py    # GPIO + valve control
│   ├── logging/
│   │   └── datalogger.py    # SQLite + image storage
│   ├── calibration/
│   │   └── calibrate.py     # Color + spatial calibration
│   ├── dashboard/
│   │   ├── app.py           # FastAPI server
│   │   ├── templates/       # HTML templates
│   │   └── static/          # CSS/JS
│   └── pipeline.py          # Main pipeline orchestrator
├── tools/
│   ├── annotate.py          # Simple annotation UI
│   ├── collect_data.py      # Data collection mode
│   └── test_sorting.py      # Valve test utility
├── models/                  # Trained model files
├── data/                    # Training data
│   ├── images/
│   └── labels.csv
└── logs/                    # Runtime logs + stone database
```

## Appendix B: Key Equations

**Pixels per mm (at sensor):**
```
px_per_mm = magnification × (1000 / pixel_size_µm)
At 1:1 mag, 3.45µm pixel: 1000/3.45 = 290 px/mm
At 2:1 mag: 580 px/mm
```

**Stone size estimation:**
```
size_mm = contour_diameter_px / px_per_mm
```

**Throughput calculation:**
```
stones_per_hour = 3600 / inter_stone_interval_sec
For 3,000/hr: 1 stone every 1.2 sec
For 5,000/hr: 1 stone every 0.72 sec
```

**Valve firing time:**
```
t_fire = t_capture + (distance_mm / velocity_mm_per_sec) - (valve_response_ms / 1000 / 2)
```

---

*End of specification. This document should be updated as the system is built and calibrated.*
