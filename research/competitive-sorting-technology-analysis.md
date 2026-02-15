# Competitive Analysis: Automated Gemstone/Mineral Sorting Technology

**Date:** 2026-02-15
**Purpose:** Inform the design of a low-cost, portable colored sapphire melee sorting system for Duke

---

## Table of Contents
1. [Commercial Gemstone Sorting Machines](#1-commercial-gemstone-sorting-machines)
2. [Diamond-Specific Grading/Sorting Systems](#2-diamond-specific-gradingsorting-systems)
3. [AI/Software-Only Gemstone Classification](#3-aisoftware-only-gemstone-classification)
4. [Adjacent Technology (Food/Agricultural Sorters)](#4-adjacent-technology-foodagricultural-sorters)
5. [DIY/Open-Source Projects](#5-diyopen-source-projects)
6. [Academic Research](#6-academic-research)
7. [Key Technologies Summary](#7-key-technologies-summary)
8. [Technology Comparison Matrix](#8-technology-comparison-matrix)
9. [Gap Analysis](#9-gap-analysis)
10. [Key Learnings](#10-key-learnings)
11. [What to Avoid](#11-what-to-avoid)
12. [Recommendations](#12-recommendations)

---

## 1. Commercial Gemstone Sorting Machines

### 1.1 Binder+Co CLARITY Minerals (formerly MINEXX)
- **Company:** Binder+Co AG (Austria)
- **Target market:** Large-scale mining operations (Gemfields, etc.)
- **How it works:**
  - **Sensor fusion:** Combines VIS (visible color cameras), UV fluorescence, and NIR sensors in a single machine
  - Data from all sensors is linked per-object for sorting decisions
  - High-speed pneumatic air jet ejection valves
  - Vibration feeder distributes material across sorting width
  - Conveyor belt or chute-based feed system
- **Specs:**
  - Sorting widths: 1000mm and 1400mm
  - Throughput: 5–25 tonnes/hour (bulk ore, not individual stones)
  - Minimum particle size: ~10mm (they recommend classification/screening for <10mm fractions)
  - Claims >2/3 of world's rubies are now found by CLARITY Minerals machines
  - Used by Gemfields (Mozambique rubies, Zambian emeralds)
- **Price:** Not published; estimated $200,000–$1,000,000+ for industrial installations
- **Pros:**
  - Proven at scale for colored gemstone recovery (rubies, emeralds)
  - Sensor fusion (VIS+UV+NIR) is the gold standard approach
  - Handles color variations and intergrowths that pure color sorting misses
  - Automated, hands-off operation (security benefit)
- **Cons:**
  - Industrial scale only — minimum ~10mm particle size
  - Far too large and expensive for melee sorting
  - Designed for ore recovery (gem vs waste), not color/quality grading of already-identified gems
  - Not portable
- **Relevance:** The sensor fusion concept (VIS+UV+NIR) is directly applicable. UV fluorescence is particularly useful for distinguishing treated vs natural sapphires. Their approach of combining multiple data modalities per stone is the right architecture.

### 1.2 TOMRA Mining (COM Series)
- **Company:** TOMRA Sorting Solutions (Norway)
- **Target market:** Mining — diamonds, gemstones, industrial minerals, coal
- **How it works:**
  - Multiple sensor technologies: XRT (X-ray transmission), NIR, color cameras, LASER
  - COM Tertiary XRT: specialized for smaller particle sizes with high-resolution ejection
  - COM XRT 2.0: higher belt speed, larger particles
  - LASER Dual: 20–120mm particles, pre-concentration
  - Air jet ejection with new TS100C module (4x faster for small particles)
- **Specs:**
  - Gemstone sorting product line exists (diamonds, emeralds, tanzanite, rubies)
  - Minimum particle size: appears to be ~5-8mm for smallest models
  - Industrial throughput (tonnes/hour)
  - XRT particularly good for diamonds (carbon density detection)
- **Price:** $100,000–$500,000+ range (industrial equipment)
- **Pros:**
  - World leader in sensor-based sorting
  - Multiple sensor options adaptable to material type
  - Dedicated gemstone sorting product line
  - New high-speed ejectors for small particles
- **Cons:**
  - Industrial scale, not suitable for melee
  - XRT technology (their strength) less relevant for colored stone color grading
  - Extremely expensive
  - Designed for mine-site recovery, not gem trading/grading
- **Relevance:** Their high-speed air ejection valve technology and approach to small particle handling is informative. The concept of combining color cameras with other sensors is validated.

### 1.3 REDWAVE ROX
- **Company:** REDWAVE (BT-Wolfgang Binder GmbH, Austria)
- **Target market:** Mining, mineral processing, precious stones
- **How it works:**
  - Color-based optical sorting
  - Detection of colors, brightness, and transparency
  - Single or double-side detection
  - Combined systems (color + material recognition)
  - Working width up to 2,000mm
  - Vibration feeder + air jet ejection
- **Specs:**
  - Sorts wet and dry materials
  - Industrial throughput
  - Can sort gemstones, limestone, quartz, ores
- **Price:** Industrial pricing, $100,000+
- **Pros:**
  - Can detect transparency (useful for gem quality)
  - Handles wet materials
- **Cons:**
  - Industrial scale
  - Not specialized for melee-sized gemstones
- **Relevance:** Transparency detection is relevant for clarity assessment.

### 1.4 AMD Color Sorter (Zhongke Optic-electronic, China)
- **Company:** AMD® / Zhongke (Hefei, China)
- **Target market:** Ore and mineral sorting
- **How it works:**
  - CCD color cameras (RGB)
  - AI-powered sorting algorithms
  - Some models include XRT
  - Belt-type and chute-type sorters
  - Air jet ejection
- **Key model:** LKZ1440-2DB — handles 1–30mm particle size range
- **Specs:**
  - Particle size: 1–30mm (the 1mm lower bound is notable!)
  - CCD cameras, claimed 99.9% accuracy for color sorting
  - Multiple chute configurations
- **Price:** Estimated $10,000–$50,000 (Chinese industrial equipment, significantly cheaper than European)
- **Pros:**
  - 1mm minimum particle size — closest to melee range
  - Much cheaper than Binder+Co/TOMRA
  - AI-powered sorting
  - Chinese supply chain — easier to source components
- **Cons:**
  - Designed for bulk ore, not individual gemstone grading
  - Color-only (no UV, no advanced clarity assessment)
  - Still industrial sized equipment
  - Quality/reliability concerns vs European brands
- **Relevance:** HIGH — proves that machine color sorting at 1mm particle size is commercially viable. The chute-type feeding mechanism for small particles is worth studying. Price point shows Chinese manufacturing can dramatically reduce costs.

---

## 2. Diamond-Specific Grading/Sorting Systems

### 2.1 OGI Systems Melee Sorter (DiamScope)
- **Company:** OGI Systems Ltd. (Israel)
- **Target market:** Diamond manufacturing, melee diamond sorting
- **How it works:**
  - HD digital camera with fixed lens
  - Computerized measurement gauge
  - Measures length, width, corner angles, diamond edges
  - 1 second per stone scanning
  - Size sorting (not color/clarity)
- **Specs:**
  - Measures stones from 0.01 to 18 points (~1mm to ~4mm)
  - 1 second per stone
  - Weight: 1.75kg
  - Dimensions: 100x110x290mm (very compact!)
  - Voltage: 12V, 110-220V auto-switching
  - Measures all shapes: round, princess, baguette, square
- **Price:** Estimated $5,000–$15,000
- **Pros:**
  - Purpose-built for melee size stones
  - Extremely compact and portable
  - Fast (1 stone/second)
  - Handles the exact size range we need
- **Cons:**
  - Diamond-only (not colored stones)
  - Measures dimensions/shape, NOT color or clarity
  - Manual stone placement (one at a time)
  - No automated sorting mechanism (measurement only)
- **Relevance:** HIGH — proves that high-speed imaging of melee-sized stones is feasible in a compact form factor. The 1-second-per-stone benchmark is achievable. Their optical approach to tiny stones is informative.

### 2.2 Sarine Technologies
- **Company:** Sarine Technologies Ltd. (Israel, publicly traded)
- **Target market:** Diamond industry — manufacturing, grading, trading
- **How it works:**
  - AI-powered automated 4C grading (Cut, Color, Clarity, Carat)
  - **Sarine Color™:** Spectroscopic analysis + AI for color grading, also detects fluorescence, tinge, milkiness
  - **Sarine Clarity™ / Clarity-II™:** AI-based inclusion mapping, grades VVS to I3
  - Trained on massive diamond dataset from GIA-grade standards
  - First fully automated AI-powered diamond grading lab
- **Specs:**
  - Lab-grade accuracy for color and clarity
  - Automated sorting into sub-categories
  - Handles polished diamonds
  - Can detect: milkiness, tinge, fluorescence, Black in Table (BIT), Black in Crown (BIC)
- **Price:** Not published; likely $50,000–$200,000+ per system, or service-based pricing
- **Pros:**
  - Most advanced automated grading technology in existence
  - AI achieves lab-grade accuracy
  - Can sort by multiple quality parameters simultaneously
  - Proven at industrial scale in Indian diamond manufacturing
- **Cons:**
  - Diamond-specific (no colored stone products announced)
  - Extremely expensive
  - Proprietary, closed system
  - Not portable
- **Relevance:** Their AI architecture (spectroscopy + deep learning) is the target to emulate. The fact that they can detect subtle quality factors (milkiness, tinge) through automated imaging proves the concept for our colored stone application.

### 2.3 GIA Automated Grading
- **Company:** Gemological Institute of America
- **Target market:** Diamond grading labs
- **How it works:**
  - Partnership with IBM Research for AI development
  - Custom hardware captures high-resolution diamond images
  - AI trained to identify inclusion types (feathers, crystals, needles)
  - Assigns clarity grades based on inclusion patterns
  - Automated color grading via spectroscopic analysis
- **Specs:**
  - Lab-grade accuracy (designed to match/exceed human graders)
  - Focuses on diamonds >0.15ct currently
  - Continuous learning — keeps ingesting new data
- **Price:** Not commercially available (internal GIA equipment)
- **Pros:**
  - Gold standard for accuracy
  - Demonstrates AI can match expert human graders
  - Extensive training data (decades of GIA records)
- **Cons:**
  - Not commercially available
  - Diamond-only
  - Requires massive training datasets
- **Relevance:** Validates the concept that AI can grade gemstones as well as humans. Their approach of using specialized lighting/imaging hardware paired with trained AI is the right model.

---

## 3. AI/Software-Only Gemstone Classification

### 3.1 Gemtelligence (Gübelin Gem Lab + CSEM)
- **Company:** Gübelin Gem Lab (Switzerland), with CSEM
- **Target market:** Gem laboratories, origin determination, treatment detection
- **How it works:**
  - Deep learning neural network
  - Processes spectral data (UV-Vis-NIR spectroscopy) AND tabular data (refractive index, specific gravity, etc.)
  - Multi-modal: combines different analytical data types
  - Trained on ~5,500 samples from Gübelin's reference collection (30,000+ stones)
  - Focus on origin determination and treatment detection
  - Tested extensively on blue sapphires
- **Specs:**
  - Accuracy: >95% for some classification problems
  - Handles ruby, sapphire, emerald origin determination
  - Published in Nature Communications Engineering (2024)
  - Open source code: github.com/TommasoBendinelli/Gemtelligence
- **Price:** Not a commercial product (research/internal tool)
- **Pros:**
  - PURPOSE-BUILT for colored gemstones including sapphires
  - Open source code available!
  - Multi-modal data fusion (spectral + tabular)
  - >95% accuracy on origin determination
  - Published, peer-reviewed methodology
- **Cons:**
  - Requires expensive analytical instruments (spectrometers, etc.) as inputs
  - Origin determination, not color/clarity sorting
  - Not a sorting machine — software only
  - Needs extensive training data
- **Relevance:** VERY HIGH — open source code for gemstone classification using deep learning. The multi-modal architecture (combining image data with measurement data) is exactly what we should consider. Blue sapphire is their focus species.

### 3.2 Porolis GemLUX® / KROMA™
- **Company:** Porolis (Sri Lanka, part of BP de Silva Group)
- **Target market:** Gem traders, labs, cutters, miners
- **How it works:**
  - **KROMA™:** Physical scanning device for gemstone imaging (photomicrography)
  - **GemLUX® AIaaS:** Cloud-based AI platform
  - AI-driven color grading, identification, authentication
  - Treatment detection, provenance tracking
  - Digital signature tagging of gemstones
  - Partnership with Fcode Labs for software development
- **Specs:**
  - Cloud-based (requires internet)
  - Handles multiple gemstone types
  - Soft-launched at FACETS 2025 trade show in Colombo
  - Backed by Gemmological Institute of Colombo (GIC)
- **Price:** Likely subscription-based AIaaS model; hardware device price unknown
- **Pros:**
  - Built specifically for colored gemstone grading
  - AI color grading is directly relevant
  - Sri Lankan gem industry connections
  - Cloud-based = potentially accessible
  - Handles identification + grading in one system
- **Cons:**
  - Very new (soft-launched 2025), may not be mature
  - Cloud-dependent (not offline capable)
  - Not a sorting machine — grading/assessment tool
  - Pricing unclear
  - Manual, one-stone-at-a-time workflow
- **Relevance:** HIGH — closest commercial product to what we need for the AI/grading component. Their KROMA scanner's imaging approach for colored gems is directly relevant. Worth monitoring.

---

## 4. Adjacent Technology (Food/Agricultural Sorters)

### 4.1 Chinese Mini Color Sorters (Multiple Manufacturers)
- **Companies:** GroTech, Skysorter, Topsort, TAIHO, Wenyao, and dozens more (mostly Hefei, Anhui Province, China)
- **Target market:** Small-scale rice/grain processing
- **How it works:**
  - CCD or CMOS RGB cameras (some with NIR)
  - Chute-type gravity feed (stones slide down inclined chutes)
  - Solenoid-driven air jet ejection (compressed air nozzles)
  - Background illumination + front illumination
  - Real-time image processing on FPGA or embedded DSP
  - Typically sort by color difference from reference
- **Key models:**
  - GroTech KD32: 32-channel mini sorter
  - Skysorter Mini: smallest RGB color sorter, 32 channels
  - TAIHO Zhiling Series: mini rice color sorter
  - Various "1-chute" mini models
- **Specs:**
  - Particle size: typically 2-10mm (rice/grain sized)
  - Throughput: 0.3–1 tonne/hour for mini models
  - Claimed accuracy: 99.9% (for simple color defect rejection)
  - Channels: 32–64 for mini models
  - Power: 220V, ~1-2kW
  - Footprint: ~60x80x150cm for mini models
- **Price:**
  - Mini (1-chute, 32-channel): $3,000–$8,000 USD on Alibaba
  - Small (2-chute, 64-channel): $5,000–$15,000 USD
  - Full-size (7-14 chute): $15,000–$50,000 USD
- **Pros:**
  - **Incredibly cheap** for proven, high-speed optical sorting
  - Proven technology — millions deployed globally
  - Handles small particles (rice ~5-7mm, similar to melee gems)
  - High throughput even at mini scale
  - Air jet ejection is proven for small particles
  - Chinese supply chain — easy to source
  - Some explicitly mention mineral/ore sorting capability
- **Cons:**
  - Binary sorting only (accept/reject) — not multi-category
  - Color-based only — no clarity, no fluorescence
  - Designed for uniform particles (rice), not irregular gems
  - Chute feeding assumes consistent particle shape
  - Transparent/translucent materials may confuse standard algorithms
  - No individual stone tracking/measurement
- **Relevance:** EXTREMELY HIGH — this is potentially the most relevant adjacent technology:
  1. The mechanical platform (chute + air jet) could be adapted
  2. The camera + FPGA architecture is proven at low cost
  3. A modified version with better cameras and custom AI could sort gems
  4. The $3,000-$8,000 price point proves low-cost sorting hardware is achievable
  5. Component suppliers (cameras, air valves, FPGAs) are all in Shenzhen/Hefei

### 4.2 Key Insight: Wenyao Color Sorter
- Wenyao explicitly lists gemstone sorting as an application on their website
- This suggests that some Chinese color sorter manufacturers are already aware of the gemstone market and may offer customization

---

## 5. DIY/Open-Source Projects

### 5.1 Gemstone Classification CNN (loremendez/Gemstones)
- **Platform:** GitHub
- **What:** Deep CNN with residual blocks for gemstone classification
- **Tech:** TensorFlow 2.5, Python
- **Relevance:** Demonstrates feasibility of CNN-based gemstone classification

### 5.2 Automatic Gemstone Classification (hybchow/gems)
- **Platform:** GitHub (published in MDPI Minerals, 2022)
- **What:** Computer vision pipeline for classifying 68 categories of gemstones
- **Tech:** Multiple approaches tested — Random Forest + handcrafted features outperformed ResNet-18/50
- **Key finding:** Traditional ML (Random Forest with color histograms in RGB/HSV/CIELAB, LBP, Haralick texture, GLCM) achieved BETTER accuracy than deep learning (ResNet) on gemstones
- **Relevance:** VERY HIGH — suggests that for our application, simpler feature extraction + classical ML may work better than deep learning, especially with limited training data

### 5.3 Gemtelligence Open Source (TommasoBendinelli/Gemtelligence)
- **Platform:** GitHub
- **What:** Code from the Gübelin Gem Lab paper on gemstone origin determination
- **Tech:** Custom neural network architecture (HoL-Net), PyTorch
- **Includes:** Training pipeline, evaluation, visualization, web app
- **Relevance:** HIGH — production-quality gemstone classification code, specifically tested on sapphires

### 5.4 Arduino/Raspberry Pi Color Sorters
- Multiple DIY projects exist:
  - Arduino color sorter with TCS3200 color sensor + servo motors
  - IronBeadSorter (GitHub: RobertGetzner/IronBeadSorter) — USB camera + OpenCV + Arduino + stepper motors, sorts by color histogram matching
  - M&M/Coin sorting machines with RPi + camera + conveyor
- **Typical components:**
  - TCS3200 color sensor OR USB camera
  - Arduino/RPi for control
  - Servo motors for mechanical sorting
  - 3D-printed housings
- **Throughput:** Very low (1-5 items/minute for servo-based)
- **Relevance:** MEDIUM — proves the basic concept works at hobby level. The camera+OpenCV+histogram approach from IronBeadSorter is closest to what we need. Servo-based sorting is too slow; need air jet.

### 5.5 Roboflow Automated Sorting Guide
- Roboflow (2025) published a guide on building automated sorting with computer vision
- Uses YOLO or similar object detection + classification
- Can be deployed on edge devices (Jetson, RPi)
- **Relevance:** Good reference architecture for the CV pipeline

---

## 6. Academic Research

### 6.1 "Automatic Gemstone Classification Using Computer Vision" (Chow & Reyes-Aldasoro, 2022)
- **Published:** Minerals MDPI, 2022
- **Dataset:** 2,042 training + 284 test images, 68 gemstone categories
- **Methods tested:** Logistic Regression, LDA, KNN, Decision Tree, Random Forest, Naive Bayes, SVM, ResNet-18, ResNet-50
- **Features:** Color histograms (RGB, HSV, CIELAB), LBP, Haralick texture, GLCM
- **Key result:** Random Forest with handcrafted features outperformed deep learning
- **Implication:** For gemstone classification from images, classical CV features may be sufficient

### 6.2 "Gemo: AI-Powered Approach to Color, Clarity, Cut Prediction and Valuation" (2023)
- Hybrid CNN-KNN model for gemstone grading
- Predicts color, clarity, cut quality, and price
- Directly relevant to our sorting criteria

### 6.3 "Gemstone Classification Using Deep Convolutional Neural Network" (2024)
- ResearchGate, recent work on deep CNN for gemstone classification
- Builds on prior work with improved architectures

### 6.4 GIA Machine Learning for Gem Classification (2024)
- **Published:** Gems & Gemology (Fall 2024)
- GIA's own research on ML for gemstone classification
- Uses physical features, spectroscopic characteristics, elemental compositions
- Achieves >95% accuracy for some classification tasks
- Notes challenges with geologically similar deposits producing near-identical stones

### 6.5 GEMTELLIGENCE Paper (Nature Communications Engineering, 2024)
- Most rigorous academic work on AI gemstone classification
- Focus on blue sapphires specifically
- Multi-modal deep learning (spectral + tabular data)
- Demonstrates that combined data modalities outperform single-source

---

## 7. Key Technologies Summary

### Sensors Used Across All Systems
| Sensor Type | Used By | Best For | Cost |
|---|---|---|---|
| RGB Camera (CCD/CMOS) | Everyone | Color sorting | $10–$500 |
| NIR Spectroscopy | Binder+Co, TOMRA | Material identification, treatment detection | $1,000–$50,000 |
| UV Fluorescence | Binder+Co | Treatment detection, material ID | $50–$500 (UV LEDs) |
| XRT (X-ray) | TOMRA | Diamond recovery, density sorting | $50,000+ |
| Hyperspectral | Research | Detailed mineral ID | $10,000–$100,000 |
| Portable NIR (MicroNIR) | Viavi | Field identification | $5,000–$15,000 |

### AI/ML Approaches
| Approach | Used By | Accuracy | Data Needed | Edge-Deployable? |
|---|---|---|---|---|
| Random Forest + handcrafted features | Academic | Good (best for small datasets) | Low (100s) | Yes |
| CNN (ResNet, etc.) | Gemtelligence, Gemo | Very good | High (1000s+) | Yes (TFLite) |
| Multi-modal deep learning | Gemtelligence | Excellent (>95%) | Very high | Possible |
| Classical color thresholding | Chinese sorters | Good for binary | Minimal | Yes (FPGA) |
| Spectroscopy + ML | Sarine, GIA | Lab-grade | Massive | No |

### Mechanical Sorting Methods
| Method | Used By | Speed | Precision | Cost | Best For |
|---|---|---|---|---|---|
| Pneumatic air jet | All industrial | Very fast | Good | $$ | Small particles, high throughput |
| Servo/robotic arm | DIY, some lab | Slow | Very precise | $ | Low volume, individual handling |
| Conveyor diverter | Some industrial | Fast | Moderate | $$ | Larger items |
| Gravity chute + air | Chinese sorters | Very fast | Good | $ | Small uniform particles |
| Vibrating tray + pick | Lab equipment | Medium | Precise | $$ | Batch processing |

---

## 8. Technology Comparison Matrix

| Product | Type | Stone Size | Throughput | Sensors | AI | Sorting | Price Est. | Colored Gems? |
|---|---|---|---|---|---|---|---|---|
| Binder+Co CLARITY | Industrial | >10mm | 5-25 t/hr | VIS+UV+NIR | Custom | Air jet | $200K-$1M | ✅ Ruby, Emerald |
| TOMRA COM | Industrial | >5mm | Tonnes/hr | XRT+Color+NIR | Custom | Air jet | $100K-$500K | ✅ Diamonds mainly |
| REDWAVE ROX | Industrial | >5mm | Tonnes/hr | Color+transparency | Custom | Air jet | $100K+ | ✅ Some |
| AMD LKZ1440 | Industrial | 1-30mm | Tonnes/hr | CCD Color | AI | Air jet | $10K-$50K | ❓ Possible |
| OGI Melee Sorter | Lab gauge | 1-4mm | 1 stone/sec | HD Camera | None | Manual | $5K-$15K | ❌ Diamonds only |
| Sarine | Lab system | >1mm | Batch | Spectroscopy+Camera | Deep learning | Manual | $50K-$200K | ❌ Diamonds only |
| GIA Automated | Lab system | >1mm | Batch | Custom imaging | AI (IBM) | Manual | N/A | ❌ Diamonds only |
| Gemtelligence | Software | Any | N/A | Spectral input | Deep learning | None | N/A (open source) | ✅ Sapphires! |
| Porolis GemLUX | Scanner+Cloud | Any | Manual | Photomicrography | Cloud AI | None | Unknown | ✅ All colored |
| Chinese Mini Sorter | Machine | 2-10mm | 0.3-1 t/hr | CCD RGB | FPGA | Air jet | $3K-$8K | ❓ Adaptable |

---

## 9. Gap Analysis

### What Duke Needs:
1. **Sort rough colored sapphire melee** (1-5mm) by color category (blue shades, pink, yellow, green, etc.)
2. **Assess basic clarity** (clean vs heavily included)
3. **Low cost** (ideally <$5,000 for a working prototype)
4. **Portable** (bring to gem markets, mines)
5. **Moderate throughput** (100s-1000s of stones per hour, not tonnes)
6. **Multi-category sorting** (not binary accept/reject)

### What Exists vs What's Missing:

| Requirement | Available? | Gap |
|---|---|---|
| Color sorting of 1-5mm particles | ✅ AMD sorter does 1mm; Chinese mini sorters do 2mm+ | Need to adapt for translucent/transparent gems |
| Multi-category color sorting | ❌ Most do binary (good/bad) | Need custom AI for multi-bin classification |
| Clarity assessment | ❌ Only Sarine/GIA do this, for diamonds | Need to develop from scratch using transmitted light |
| Low cost (<$5K) | ✅ Chinese mini sorters are $3-8K | Need to strip down to essentials |
| Portable | ❌ Smallest systems are ~60x80cm | Need custom compact design |
| Colored gemstone AI | ⚠️ Gemtelligence (open source!) + Porolis (commercial) | Need to train on Duke's specific sapphire categories |
| Automated feeding of melee | ⚠️ Vibrating feeders exist for this size | Need custom feeder for irregular gem shapes |
| Air jet ejection at 1-5mm | ✅ Proven technology in Chinese sorters | Standard components available |

### The Critical Gap:
**No existing product combines low-cost hardware + colored gemstone AI + multi-category sorting + melee size range + portability.** Every existing solution is either:
- Industrial scale (Binder+Co, TOMRA) — too big, too expensive
- Diamond-only (Sarine, OGI, GIA) — wrong stone type
- Software-only (Gemtelligence, Porolis) — no sorting hardware
- Binary sorting (Chinese color sorters) — can't do multi-category
- Too basic (DIY projects) — can't handle the throughput or accuracy needed

**This gap represents a real market opportunity.**

---

## 10. Key Learnings

### From the best systems, we should copy:

1. **Sensor fusion (Binder+Co):** Don't rely on a single sensor. Combining RGB camera + UV LEDs (for fluorescence) dramatically improves discrimination. UV is cheap to add.

2. **Chute-based gravity feed + air jet ejection (Chinese sorters):** This is the proven, low-cost, high-speed mechanical architecture for small particles. Don't reinvent it.

3. **Controlled lighting environment (all systems):** Every successful system uses a dark enclosure with calibrated lighting. This is non-negotiable for consistent color measurement.

4. **FPGA/dedicated image processing (Chinese sorters):** Real-time sorting requires fast image processing. Even cheap Chinese sorters use FPGAs, not general-purpose CPUs.

5. **Classical CV features may beat deep learning (academic research):** With limited training data (<5000 images), Random Forest with HSV/CIELAB color histograms + texture features may outperform CNNs. Start simple.

6. **Multi-modal data (Gemtelligence):** Combining image data with measurement data (size, weight estimate from dimensions) improves classification.

7. **Transmitted light for clarity (Sarine concept):** Assessing clarity requires light passing THROUGH the stone, not just reflected light. Need both reflected (color) and transmitted (clarity) illumination.

8. **Individual stone tracking (OGI/Sarine):** For quality sorting (not bulk ore), each stone needs to be individually imaged and classified.

### Specific technical recommendations from research:

- **Color space:** Use CIELAB color space, not RGB. CIELAB is perceptually uniform and better matches human color perception. HSV is also good.
- **Camera:** Industrial machine vision camera with controlled white balance, not a consumer camera
- **Lighting:** Use diffuse, uniform illumination to minimize specular reflections from faceted/rough surfaces
- **UV:** 365nm UV-A LEDs are cheap (<$5) and can reveal treatment/origin information
- **Training data:** Photograph stones from multiple angles; augment with rotation/brightness variation

---

## 11. What to Avoid

1. **Don't build an industrial ore sorter.** Duke doesn't need tonnes/hour. A simpler, lower-throughput system that handles individual stones is better for quality sorting.

2. **Don't use XRT/X-ray.** Overkill for color sorting, expensive, regulatory issues with radiation.

3. **Don't rely on deep learning with insufficient data.** Start with classical CV (color histograms + Random Forest). Train a CNN later when you have thousands of labeled images.

4. **Don't use servo motors for sorting.** They're too slow. Air jets or a rotating bin mechanism is faster and more reliable.

5. **Don't skip the enclosure/lighting.** Ambient light variation will destroy classification accuracy. This is the #1 failure mode for DIY vision sorting projects.

6. **Don't try to match GIA/Sarine accuracy.** For rough melee sorting by color bucket, 85-90% accuracy is perfectly adequate. Pursuing lab-grade accuracy will 10x the cost.

7. **Don't overengineer the feeding mechanism.** For moderate throughput (500-1000 stones/hr), a simple vibrating tray that presents one stone at a time is sufficient. No need for the complex chute systems used in high-speed industrial sorters.

8. **Don't ignore the Chinese supply chain.** Air solenoid valves, miniature vibrating feeders, industrial cameras, and FPGA boards are all available on Taobao/1688 for a fraction of Western prices.

---

## 12. Recommendations

### Recommended Architecture:

```
[Vibrating Feeder] → [Imaging Station] → [AI Classification] → [Multi-bin Air Jet Sorter]
       ↓                    ↓                    ↓                      ↓
  Singulates stones    Dark enclosure      Edge compute           3-6 output bins
  on track/channel     RGB + UV LEDs       (Jetson/RPi5)          pneumatic diverter
                       Machine vision       Color hist + RF
                       camera (top+bottom)  or YOLO classifier
```

### Specific Component Recommendations:

1. **Camera:** Industrial machine vision camera, global shutter, ≥5MP, USB3 or CSI
   - e.g., Arducam IMX477 ($50) or Hikvision MV-CS050 ($150)
   - Consider TWO cameras: top (reflected) + bottom (transmitted) light

2. **Lighting:**
   - Ring of diffuse white LEDs for color (reflected)
   - White LED beneath for clarity (transmitted light through stone)
   - 365nm UV-A LED strip for fluorescence (switched separately)
   - All in enclosed dark chamber

3. **Compute:**
   - NVIDIA Jetson Orin Nano ($200) for ML inference
   - OR Raspberry Pi 5 ($80) if using classical CV (Random Forest)
   - Pre-process on device, no cloud dependency

4. **Sorting mechanism:**
   - 4-6 output bins
   - Miniature pneumatic solenoid valves + compressed air nozzles
   - OR rotating bin/carousel beneath drop point (simpler, lower speed)
   - Small aquarium air compressor sufficient for 1-5mm stones

5. **Feeding:**
   - Miniature electromagnetic vibrating feeder
   - V-channel to singulate stones into single file
   - Gravity drop past camera station

6. **AI/ML approach (phased):**
   - **Phase 1:** HSV/CIELAB color histogram + Random Forest classifier. Simple, fast, needs ~200-500 labeled images per category
   - **Phase 2:** Add texture features (GLCM, LBP) for clarity assessment
   - **Phase 3:** Fine-tune a lightweight CNN (MobileNet/EfficientNet) when dataset is large enough

### Estimated BOM Cost:
| Component | Estimated Cost |
|---|---|
| Machine vision camera (x2) | $100-$300 |
| LED lighting (white + UV) | $30-$50 |
| Jetson Orin Nano or RPi 5 | $80-$200 |
| Vibrating feeder | $50-$150 |
| Air solenoid valves (x4-6) | $40-$80 |
| Small air compressor | $30-$60 |
| 3D printed enclosure + bins | $50-$100 |
| Power supply, wiring, misc | $50-$100 |
| **TOTAL** | **$430-$1,040** |

### Key Risk Factors:
1. **Translucent/transparent stones** — reflected light color will vary with background. Need careful lighting design.
2. **Wet vs dry stones** — wet sapphires look very different. Need consistent stone preparation or adaptive algorithm.
3. **Size variation** — 1mm and 5mm stones need different imaging parameters. May need adjustable focus or multiple stations.
4. **Training data** — need Duke to manually sort 500-1000 stones into categories as training set.

---

## Appendix: Useful Links

### Open Source Code
- Gemtelligence: https://github.com/TommasoBendinelli/Gemtelligence
- Gemstone Classification CV: https://github.com/hybchow/gems
- Gemstone CNN: https://github.com/loremendez/Gemstones
- IronBead Sorter (OpenCV + Arduino): https://github.com/RobertGetzner/IronBeadSorter

### Chinese Sorter Manufacturers (for components/reference)
- AMD/Zhongke: https://www.amdcolorsorter.com
- GroTech: https://www.grotechcolorsorter.com
- Skysorter: https://skysorter.com
- Topsort: https://www.topsortcolorsorter.com
- TAIHO: https://www.chinacolorsort.com
- Wenyao: https://www.wenyaocolorsorter.com

### Key Papers
- Chow & Reyes-Aldasoro (2022). "Automatic Gemstone Classification Using Computer Vision." Minerals, 12(1), 60.
- Bendinelli et al. (2024). "GEMTELLIGENCE: Accelerating gemstone classification with deep learning." Nature Comms Engineering.
- GIA (2024). "Classification of Gem Materials Using Machine Learning." Gems & Gemology, Fall 2024.

### Industry Equipment
- Binder+Co CLARITY Minerals: https://www.binder-co.com/en/products/clarity-sorting-machines/mineral-sorting/
- TOMRA Mining: https://www.tomra.com/mining
- REDWAVE ROX: https://redwave.com/en/products/redwave-rox
- OGI Systems: https://www.ogisystems.com
- Sarine Technologies: https://sarine.com
- Porolis GemLUX: https://porolis.com
