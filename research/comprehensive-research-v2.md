# Comprehensive Research Report: AI Gemstone Grading & Market Intelligence
**Date:** 2026-02-14 | **Version:** 2.0

---

## 1. AI Gemstone Grading — Companies & Competitors (2024–2026)

### Tier 1: Major Established Players

#### Gübelin Gem Lab + CSEM → **Gemtelligence**
- **What:** Deep learning platform for colored gemstone origin determination and treatment detection
- **Tech:** Multi-modal deep learning (spectroscopy + imaging data); published in *Nature Communications Engineering* (Aug 2024)
- **Data:** Tens of thousands of client gemstones tested since 1970s + Gübelin Reference Stone Collection (27,000+ gems)
- **Status:** Operational and deployed in Gübelin's lab workflow; open-source code on GitHub
- **Key paper:** Bendinelli et al., "GEMTELLIGENCE: Accelerating gemstone classification with deep learning," *Commun. Eng.* 3, 110 (2024)
- **URL:** https://gubelingemlab.com/gemtelligence/

#### GIA (Gemological Institute of America) + IBM Research
- **What:** AI-based diamond clarity grading; expanding to colored gemstone classification
- **Tech:** Computer vision + IBM Watson-era ML for clarity grading; achieved automated clarity at scale
- **Status:** Deployed in GIA labs for diamond clarity since ~2020; Fall 2024 *Gems & Gemology* published ML-based gem classification paper (alexandrite provenance, saltwater pearl classification)
- **Key paper:** Hardman et al., "Classification of Gem Materials Using Machine Learning," *G&G* Fall 2024
- **URL:** https://discover.gia.edu/gia-ibm-diamond-clarity-ai.html

#### Sarine Technologies (Israel, public: SARN)
- **What:** End-to-end AI diamond pipeline — rough scanning, cut optimization, automated 4Cs grading
- **Tech:** DiaMension® for cut measurement; AI-based color and clarity grading (since 2016-2018)
- **Status:** World's first lab with fully automated 4Cs grading; GCAL by Sarine expanding into gemstone ID
- **Forbes (June 2025):** "We've been using AI-based color and clarity grading technologies since 2018"
- **URL:** https://sarine.com/egrading/

### Tier 2: Emerging Startups & Specialized Players

#### Porolis — Digital Gemmology 3.0
- **What:** AI-powered gemstone assessment SaaS platform
- **Products:**
  - **GemLUX®** — hardware/SaaS imaging system powered by KROMA™ AI engine
  - **GemPASSPORT™** — immutable digital certificate with images + digital fingerprint
- **Tech:** Object-detection AI models trained on FGA-qualified gemmologist-labelled datasets; detects heat-treatment inclusions, performs AI color grading
- **Target users:** Gem labs, cutters, miners, jewelry designers, e-marketplaces
- **Status:** Active (website updated Oct 2025); appears to be Singapore/Asia-based
- **URL:** https://porolis.com/

#### GemSorter.ai
- **What:** AI-based physical gemstone sorting hardware
- **Tech:** Computer vision sorting by color, clarity, cut, and shape
- **Target:** Gemstone businesses needing automated batch sorting
- **Status:** Active product offering
- **URL:** https://gemsorter.ai/

#### Gembridge
- **What:** Digital B2B colored gemstone trading platform (not AI grading per se, but digital infrastructure)
- **Status:** Launched 2021, raised $500K; appointed China ambassador (Chen Shen, Shanghai)
- **Relevance:** Represents the digital transformation infrastructure that AI grading feeds into
- **URL:** https://www.gembridge.com/

### Tier 3: Adjacent / Diamond-Focused

| Company | Focus | Notes |
|---------|-------|-------|
| **Sarine** | Diamond AI grading + rough planning | Market leader; expanding to gemstone ID via GCAL |
| **TOMRA (OBTAIN™)** | AI ore sorting | Deep-learning single-particle mineral sorting |
| **Rapaport** | Diamond pricing + AI coverage | Reports on AI adoption across industry |

### Key Observation
The colored gemstone AI grading space is **far less mature** than diamonds. Gübelin/Gemtelligence and GIA are the only entities with published, peer-reviewed ML approaches for colored stones. Porolis is the most prominent startup specifically targeting colored gemstone AI assessment as a product.

---

## 2. Latest Research Papers (2024–2026)

### Flagship Papers

| Year | Title | Authors/Source | Key Contribution |
|------|-------|---------------|-----------------|
| 2024 | **GEMTELLIGENCE: Accelerating gemstone classification with deep learning** | Bendinelli et al., *Nature Commun. Eng.* 3, 110 | Multi-modal DL for origin determination & treatment detection of rubies/sapphires; performance comparable to LA-ICP-MS |
| 2024 | **Classification of Gem Materials Using Machine Learning** | Hardman et al., *GIA Gems & Gemology* Fall 2024 | ML models for alexandrite provenance and natural saltwater pearl classification |
| 2024 | **Gemstone Classification Using Deep CNN** | Yamsani et al., *J. Inst. Eng. (India) Series B* | Deep CNN on Kaggle gemstone image dataset; open-source approach |
| 2024 | **Jewelry rock discrimination using LIBS and convolutional LSTM** | *Scientific Reports* (Nature) | Laser-induced breakdown spectroscopy + deep learning for gem rock identification |
| 2024 | **ML-Based Gemstone Classification through Absorption Band Spectrum + Refractive Index** | IEEE Conference | Computer vision method using spectral images |
| 2024 | **High Dynamic Range Edge Filtered LeNet based Gemstone Classification** | Shyamala Devi et al. (Conference) | HDR image processing for gem classification |
| 2025 | **Machine Learning Approaches for Real-Time Mineral Classification** | MDPI Applied Sciences | Real-time multi-detection system combining classical CV + deep learning |
| 2025 | **Mycelial_Net: Bio-Inspired DL for Mineral Classification in Thin Section** | MDPI Minerals | Novel bio-inspired architecture for mineral thin section analysis |
| 2024 | **Two-stage DL pipeline for mineral classification in Raman spectra** | CLEO Conference | Denoising autoencoders + CNNs for planetary mineral Raman spectra |

### Related: Cutting Yield Optimization
- **Sarine** leads in AI-driven rough diamond planning (scan → map inclusions → generate optimal cutting plans)
- **GemCad** — traditional faceting CAD software (not AI-driven)
- No published AI papers specifically on *colored gemstone* cutting yield optimization found — this is a **gap in the literature**

---

## 3. Open Source Tools & Datasets

### GitHub Repositories

| Repository | Description | Stars/Activity |
|-----------|-------------|----------------|
| **[TommasoBendinelli/Gemtelligence](https://github.com/TommasoBendinelli/Gemtelligence)** | Code for the Nature paper; multi-modal DL for gemstone classification | Key reference implementation |
| **[hybchow/gems](https://github.com/hybchow/gems)** | Computer vision gemstone classification (68 categories, 2326 images) | Accompanies MDPI Minerals 2022 paper |
| **[carlm451/Gemstone_Images_Classification_Fine_Tuning](https://github.com/carlm451/Gemstone_Images_Classification_Fine_Tuning)** | Fine-tuning multiclass image classification on Kaggle dataset | Tutorial/educational |
| **[loliverhennigh/Crystal-Gems](https://github.com/loliverhennigh/Crystal-Gems)** | ~4,000 mineral images with labels scraped from minerals.net | Dataset repo |

### Kaggle Datasets

| Dataset | Description |
|---------|-------------|
| **[lsind18/gemstones-images](https://www.kaggle.com/datasets/lsind18/gemstones-images)** | Gemstone images for multiclass classification (widely cited) |
| **[gauravkamath02/precious-gemstone-identification](https://www.kaggle.com/datasets/gauravkamath02/precious-gemstone-identification)** | Precious gemstone identification dataset |
| **[colearninglounge/gemstone-price-prediction](https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction)** | Tabular gemstone price prediction |
| **Playground Series S3E8** | Regression with tabular gemstone price dataset (Kaggle competition) |

### Notable Gap
There is **no large-scale, professionally annotated open dataset** of rough gemstone images with quality grades, origin labels, or treatment status. The Gübelin/Gemtelligence dataset is proprietary (spectroscopy data from their lab). This represents a significant opportunity.

---

## 4. Market Prices — Rough Ruby & Sapphire

### Ruby Prices (2024–2025, Faceted/Retail Reference)

| Origin | Quality | Price Range ($/carat) | Notes |
|--------|---------|----------------------|-------|
| **Myanmar (Burma)** | Top unheated, "pigeon blood" | $15,000–$100,000+ | Rarest; auction records exceed $1M/ct |
| **Myanmar** | Good heated | ~$12,000 | 2ct reference |
| **Mozambique** | Top quality | $5,000–$15,000 | Now major source; approaching Burma prices |
| **Madagascar** | Good quality | $1,000–$5,000 | Growing source |
| **Thailand/Cambodia** | Commercial | $500–$3,000 | Darker, often heated |
| **India/Tajikistan** | Variable | $20–$2,000 | Wide quality range |
| **Rough (good quality)** | — | Up to $10,000/ct | "Hardly unheard of" per Natural Ruby Co. |
| **Commercial rough** | Low grade | $1–$50/ct | Bulk material |

### Sapphire Prices (2024–2025, Faceted/Retail Reference)

| Origin | Quality | Price Range ($/carat) | Notes |
|--------|---------|----------------------|-------|
| **Kashmir** | Top unheated "cornflower" | $50,000–$200,000+ | Extremely rare; mine exhausted |
| **Myanmar** | Top "Royal Blue" | $5,000–$20,000 | |
| **Sri Lanka (Ceylon)** | Top unheated | $3,000–$10,000+ | Classic source |
| **Madagascar** | Good quality | $1,000–$5,000 | Major modern source |
| **Commercial heated** | Decent quality | $200–$800/ct | Standard trade goods |
| **Low grade** | Dark/included | $20–$100/ct | Cabochon material |
| **AAAA grade, 1ct** | Top tier | $4,500–$8,000 | Per Gemdaia pricing guide |

### Key Price Trends
- Colored gemstone prices **rose 30-50% across categories in China** during H1 2023 (per China Gems and Jade Association)
- Mozambique rubies continue to **close the gap** with Burmese stones
- Unheated premium: 2-5x over heated equivalents for top-quality stones
- Large stones (>3ct) command exponential premiums

---

## 5. Chinese Gemstone Market

### Market Position
- **China led Asia-Pacific in gemstone market share in 2024** (Data Bridge Market Research)
- Global gemstone market: **$36B in 2025**, projected $68B by 2035 (Future Market Insights)
- Chinese colored gemstone prices surged **30-50% in H1 2023** across all categories

### Consumer Preferences
- **Red stones (rubies)** — cultural association with joy and prosperity
- **Green stones (jadeite, emeralds)** — traditional significance
- Popular sizes: 1ct rubies, 2ct sapphires, 3-4ct emeralds
- Growing demand for **certified, origin-documented stones**

### Key Trading Hubs
| Hub | Role |
|-----|------|
| **Guangzhou** | Largest physical gemstone & jewelry market in China |
| **Wuzhou** | World capital of synthetic gemstones; major cutting center |
| **Hong Kong** | International trading hub; HKTDC jewelry shows |
| **Shanghai/Beijing** | High-end retail and luxury markets |

### Digital Platforms & E-Commerce
| Platform | Type | Notes |
|----------|------|-------|
| **Douyin (TikTok China)** | Livestream selling | Massive gemstone livestream commerce; "blind box" gem sales trending |
| **Taobao/Tmall** | E-commerce | Major gemstone retail |
| **Made-in-China.com** | B2B wholesale | Crystal, gemstone wholesale platform |
| **Gembridge** | B2B digital trading | Appointed China ambassador (Chen Shen, Shanghai) |
| **GemtoChina.com** | B2B connector | Connects foreign miners/dealers to Chinese importers |
| **JewelleryNet.com** | B2B platform | HKTDC-backed industry platform |

### Livestream Commerce Revolution
- China pioneered **gemstone livestream selling** on Douyin and Taobao Live
- Licensed pearl-selling livestreamers received vocational skill certificates from National Gemstone Testing Center (2023)
- "Blind box" gem sales are a trending format — viewers pay for chance to receive gems of varying quality
- This creates **massive demand for rapid, reliable grading** to build buyer trust

### 2024-2025 Market Challenges
- Chinese economic slowdown impacted luxury spending
- September 2024 Hong Kong show saw fewer Chinese pearl buyers
- Colored gemstone market growth was **slow in early 2024** but expected to recover
- Despite slowdown, China remains the **primary driver of top-end colored gemstone prices globally**

---

## 6. Strategic Insights & Gaps

### Underserved Areas (Opportunities)
1. **No AI grading system specifically for rough colored gemstones** — all current solutions focus on polished/faceted or use spectroscopy
2. **No open dataset** of rough gemstone images with professional quality grades
3. **Colored gemstone cutting yield optimization** has no published AI solution (vs. diamond space where Sarine dominates)
4. **Chinese market needs** fast, trustworthy digital grading for livestream commerce
5. **Treatment detection via imaging only** (without expensive spectroscopy) is an open research problem

### Technology Landscape Summary
```
Maturity Spectrum:

Diamond AI Grading    ████████████████████ (Mature - Sarine, GIA deployed)
Colored Gem Origin/ID ██████████████       (Research → Early Product - Gübelin, GIA)
Colored Gem AI Grade  ████████             (Early Stage - Porolis)
Rough Gem AI Grading  ███                  (Virtually Nonexistent)
Gem Cutting AI (Color)██                   (No published work)
```

### Key Competitors to Watch
1. **Porolis** — Most directly relevant colored gem AI startup
2. **Gübelin/Gemtelligence** — Gold standard for published research
3. **GIA** — Expanding ML to colored stones
4. **Sarine/GCAL** — Diamond expertise potentially extending to colored stones
5. **GemSorter.ai** — Physical sorting hardware play

---

*Report compiled from 20+ web searches across industry sources, academic databases, GitHub, and market reports. February 2026.*
