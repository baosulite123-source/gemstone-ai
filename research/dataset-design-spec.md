# Rough Gemstone Grading: Dataset & Annotation System Design Spec

> **Scope:** Ruby (Corundum – red) and Blue Sapphire (Corundum – blue) in rough/pre-cut form.
> **Goal:** Build a labeled image dataset and annotation pipeline to train ML models for automated color grading, clarity assessment, and value estimation of rough stones.

---

## 1. Image Capture Protocol

### 1.1 Camera Hardware

| Parameter | Minimum Viable | Ideal |
|---|---|---|
| Sensor | APS-C, ≥24 MP (e.g., Sony α6400) | Full-frame, ≥45 MP (e.g., Nikon Z7 II) |
| Lens | 60 mm macro, 1:1 reproduction | 105 mm macro, 1:1, ED glass (e.g., Nikkor Z MC 105 mm) |
| Bit depth | 12-bit RAW | 14-bit RAW |
| Color profile | sRGB JPEG + RAW | ProPhoto RGB 16-bit TIFF (from RAW) |
| White balance | Custom (measured off grey card under each lighting) | Per-lighting-condition profile via X-Rite i1 Studio |
| Tethering | Optional (card-based) | Mandatory – live capture to workstation via USB-C |

**Stabilisation:** Camera on copy-stand or macro rail with geared head. No handheld shots.

### 1.2 Lighting Conditions

Each stone is photographed under **five** lighting setups. All lights should be rated CRI ≥ 95.

#### L1 – D65 Diffused (Daylight Equivalent)
- **Source:** LED panel calibrated to 6500 K (e.g., Nanlite PavoSlim 60B verified with spectrometer).
- **Modifier:** Shoot through 120 cm octabox or integrating-sphere-style dome to eliminate directional shadows.
- **Purpose:** Primary color grading reference. Matches GIA/Gübelin daylight observation conditions.

#### L2 – Transmitted Light
- **Source:** D65 LED lightbox beneath a frosted acrylic diffuser plate (3 mm opal).
- **Camera:** Directly above, stone resting on diffuser.
- **Purpose:** Reveals internal clarity, color zoning, silk, and fractures. Critical for transparency assessment.

#### L3 – Cross-Polarised
- **Source:** D65 diffused (same as L1) with linear polariser on light source.
- **Camera:** Second linear polariser on lens, rotated 90° to extinction.
- **Purpose:** Eliminates surface reflections; isolates body color and subsurface inclusions. Distinguishes surface scratches from internal features.

#### L4 – Fibre-Optic / Spot (Darkfield)
- **Source:** Fibre-optic illuminator from the side, stone on darkfield well (black background, lit from below at oblique angle).
- **Purpose:** Highlights inclusions, silk needles, fingerprints, crystals against dark background. High contrast for annotation.

#### L5 – UV Fluorescence
- **Source:** Long-wave (365 nm) and short-wave (254 nm) UV lamps in darkroom conditions.
- **Camera:** Exposure 2-8 s, ISO 800-1600, lens UV/IR-cut filter.
- **Purpose:** Fluorescence response (strong/medium/weak/inert). Important for detecting treatments (lead-glass filling fluoresces differently) and origin estimation.

### 1.3 Angles and Positions

#### Minimum Viable Set (6 images × 5 lightings = 30 images per stone)

| ID | Description | Camera Position |
|----|-------------|-----------------|
| A1 | Top-down (table/crystal face up) | 90° overhead |
| A2 | Bottom-up (pavilion/base up) | 90° overhead, stone flipped |
| A3 | Side view 0° | Horizontal, 0° azimuth |
| A4 | Side view 90° | Horizontal, 90° azimuth |
| A5 | Side view 180° | Horizontal, 180° azimuth |
| A6 | Side view 270° | Horizontal, 270° azimuth |

#### Ideal Set (18 images × 5 lightings = 90 images per stone)

Add:
- **A7–A12:** 45° oblique views at 0°, 60°, 120°, 180°, 240°, 300° azimuth.
- **A13–A18:** Additional 6 macro crops of notable features (largest inclusion, color zone boundary, crystal face, fracture, surface texture, growth feature).

#### Stone Orientation

- Mark a reference point on the stone holder (notch or dot) so all angles are repeatable.
- Use a custom 3D-printed cradle or putty on a rotating stage with click-stops every 60° (ideal) or 90° (MVP).

### 1.4 Background and Calibration

#### Background
- **Primary:** Neutral grey (18% reflectance, Munsell N5) — does not shift color perception.
- **For transmitted light:** White opal acrylic diffuser (built into lightbox).
- **For darkfield:** Matte black felt or anodised aluminium well.

#### Calibration Targets (in every capture session, ≥ once per batch of 20 stones)

| Target | Purpose |
|--------|---------|
| X-Rite ColorChecker Classic (24 patch) | Color calibration & white balance verification |
| X-Rite ColorChecker SG (140 patch) | Extended gamut profiling (ideal) |
| Grey card (18%) | Exposure reference |
| Millimetre scale bar (certified, steel) | Spatial calibration – place beside stone in A1 shot |
| 1 ct / 5 ct reference weight | Optional – for scale in metadata |

#### Calibration Workflow
1. Photograph ColorChecker under each lighting condition at start of session.
2. Generate per-condition ICC profile using `dcamprof` or X-Rite i1Profiler.
3. Apply profile during RAW conversion (batch process).
4. Verify ΔE*₀₀ < 2.0 for all colour patches; if exceeded, re-calibrate.

### 1.5 File Format and Naming

#### Formats

| Stage | Format | Notes |
|-------|--------|-------|
| Capture | Camera RAW (.NEF / .ARW) | Archival master |
| Working | 16-bit TIFF, ProPhoto RGB | ICC-profiled, used for annotation |
| Training | 8-bit PNG, sRGB | Converted from TIFF; lossless, web-compatible |
| Thumbnail | JPEG, sRGB, quality 92 | For UI previews |

#### Naming Convention

```
{lot_id}_{stone_id}_{lighting}_{angle}_{date}.{ext}

Examples:
LOT2026-001_S0042_L1-D65_A1-TOP_20260214.tif
LOT2026-001_S0042_L2-TRANS_A3-SIDE0_20260214.tif
LOT2026-001_S0042_L5-UV365_A1-TOP_20260214.tif
LOT2026-001_CALIB_L1-D65_COLORCHECKER_20260214.tif
```

| Token | Values |
|-------|--------|
| `lot_id` | `LOT{YYYY}-{NNN}` |
| `stone_id` | `S{NNNN}` or `CALIB` |
| `lighting` | `L1-D65`, `L2-TRANS`, `L3-XPOL`, `L4-DARK`, `L5-UV365`, `L5-UV254` |
| `angle` | `A1-TOP`, `A2-BOT`, `A3-SIDE0`, ..., `A18-MACRO6`, `COLORCHECKER` |
| `date` | `YYYYMMDD` |

---

## 2. Grading Schema (JSON)

### 2.1 Overview

Each stone gets a single JSON grading record that links to its image set. The schema is designed for:
- Multi-grader consensus (array of `assessments`)
- Machine-readable enumerated fields (for classification tasks)
- Continuous numeric fields (for regression tasks)
- Validation via JSON Schema Draft 2020-12

### 2.2 Full JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://gemgrade.internal/schemas/rough-stone-v1.json",
  "title": "Rough Gemstone Grading Record",
  "type": "object",
  "required": ["schema_version", "stone_id", "lot_id", "metadata", "images", "assessments"],
  "additionalProperties": false,
  "properties": {

    "schema_version": {
      "const": "1.0.0"
    },

    "stone_id": {
      "type": "string",
      "pattern": "^S\\d{4,6}$",
      "description": "Unique stone identifier"
    },

    "lot_id": {
      "type": "string",
      "pattern": "^LOT\\d{4}-\\d{3,5}$"
    },

    "metadata": {
      "type": "object",
      "required": ["species", "variety", "weight_ct", "dimensions_mm", "shape"],
      "additionalProperties": false,
      "properties": {

        "species": {
          "const": "corundum"
        },

        "variety": {
          "type": "string",
          "enum": ["ruby", "blue_sapphire"]
        },

        "weight_ct": {
          "type": "number",
          "minimum": 0.01,
          "maximum": 500,
          "description": "Carat weight, measured to 0.01 ct"
        },

        "dimensions_mm": {
          "type": "object",
          "required": ["length", "width", "depth"],
          "properties": {
            "length": { "type": "number", "minimum": 0.1, "maximum": 100 },
            "width":  { "type": "number", "minimum": 0.1, "maximum": 100 },
            "depth":  { "type": "number", "minimum": 0.1, "maximum": 100 }
          },
          "description": "Maximum extents in mm, measured with digital caliper"
        },

        "shape": {
          "type": "string",
          "enum": [
            "crystal_hexagonal", "crystal_tabular", "crystal_barrel",
            "crystal_bipyramid", "fragment", "waterworn", "tumbled",
            "slab", "preform", "other"
          ]
        },

        "origin_declared": {
          "type": ["string", "null"],
          "enum": [
            "myanmar", "mozambique", "madagascar", "sri_lanka",
            "kashmir", "thailand", "cambodia", "tanzania",
            "vietnam", "afghanistan", "kenya", "other", "unknown", null
          ]
        },

        "origin_confidence": {
          "type": "string",
          "enum": ["lab_certified", "dealer_declared", "estimated", "unknown"]
        },

        "treatment_declared": {
          "type": "string",
          "enum": ["none", "heated", "lead_glass_filled", "beryllium_diffused",
                   "fracture_filled", "oiled", "coated", "unknown"]
        },

        "acquisition_date": {
          "type": "string",
          "format": "date"
        },

        "notes": {
          "type": "string",
          "maxLength": 2000
        }
      }
    },

    "images": {
      "type": "array",
      "minItems": 6,
      "items": {
        "type": "object",
        "required": ["filename", "lighting", "angle", "capture_date"],
        "properties": {
          "filename": { "type": "string" },
          "lighting": {
            "type": "string",
            "enum": ["L1-D65", "L2-TRANS", "L3-XPOL", "L4-DARK", "L5-UV365", "L5-UV254"]
          },
          "angle": {
            "type": "string",
            "pattern": "^A\\d{1,2}-[A-Z0-9]+$"
          },
          "capture_date": { "type": "string", "format": "date" },
          "pixel_scale_mm_per_px": {
            "type": ["number", "null"],
            "minimum": 0.001,
            "description": "Derived from scale bar calibration"
          },
          "annotations": {
            "type": "array",
            "items": { "$ref": "#/$defs/image_annotation" }
          }
        }
      }
    },

    "assessments": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/assessment" },
      "description": "One entry per grader; consensus derived in review stage"
    },

    "consensus": {
      "$ref": "#/$defs/assessment",
      "description": "Final agreed grade after review. Null until review complete."
    },

    "qa": {
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "enum": ["pending", "in_review", "approved", "disputed", "rejected"]
        },
        "reviewer_id": { "type": "string" },
        "review_date": { "type": "string", "format": "date-time" },
        "inter_rater_kappa": {
          "type": ["number", "null"],
          "minimum": -1,
          "maximum": 1,
          "description": "Cohen's Kappa (2 graders) or Fleiss' Kappa (3+)"
        },
        "disagreement_fields": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    }
  },

  "$defs": {

    "assessment": {
      "type": "object",
      "required": ["grader_id", "timestamp", "color", "clarity"],
      "properties": {

        "grader_id": { "type": "string" },
        "timestamp": { "type": "string", "format": "date-time" },

        "color": {
          "type": "object",
          "required": ["hue", "saturation", "tone"],
          "properties": {

            "hue": {
              "type": "string",
              "enum": [
                "red", "slightly_purplish_red", "purplish_red",
                "strongly_purplish_red", "red_orange", "pinkish_red",
                "blue", "violetish_blue", "slightly_greenish_blue",
                "greenish_blue", "strongly_greenish_blue",
                "very_slightly_violetish_blue"
              ],
              "description": "Primary hue following GIA hue wheel for corundum"
            },

            "hue_angle_deg": {
              "type": ["number", "null"],
              "minimum": 0,
              "maximum": 360,
              "description": "Numeric hue angle on GIA/Munsell wheel (optional, for regression)"
            },

            "saturation": {
              "type": "integer",
              "minimum": 1,
              "maximum": 6,
              "description": "1=greyish/brown → 6=vivid (GIA-style 6-step scale)"
            },

            "tone": {
              "type": "integer",
              "minimum": 0,
              "maximum": 10,
              "description": "0=colorless → 10=black (GIA-style 11-step scale)"
            },

            "color_grade_overall": {
              "type": "string",
              "enum": [
                "pigeon_blood", "vivid", "intense", "medium", "light",
                "very_light", "dark", "very_dark",
                "royal_blue", "cornflower_blue", "velvet_blue"
              ],
              "description": "Trade-name or summary grade"
            },

            "zoning": {
              "type": "object",
              "properties": {
                "present": { "type": "boolean" },
                "severity": {
                  "type": "string",
                  "enum": ["none", "faint", "moderate", "strong", "extreme"]
                },
                "pattern": {
                  "type": "string",
                  "enum": ["straight", "angular", "hexagonal", "patchy", "concentric", "irregular"]
                },
                "description": { "type": "string", "maxLength": 500 }
              }
            },

            "color_shift": {
              "type": "object",
              "properties": {
                "present": { "type": "boolean" },
                "d65_to_incandescent": {
                  "type": "string",
                  "enum": ["none", "slight", "moderate", "strong"]
                },
                "notes": { "type": "string", "maxLength": 500 }
              },
              "description": "Alexandrite effect or fluorescence-induced shift"
            },

            "special_designation": {
              "type": ["string", "null"],
              "enum": [
                "pigeon_blood", "royal_blue", "cornflower", "padparadscha",
                "star_ruby", "star_sapphire", "trapiche", null
              ]
            }
          }
        },

        "clarity": {
          "type": "object",
          "required": ["transparency", "overall_grade"],
          "properties": {

            "transparency": {
              "type": "string",
              "enum": ["transparent", "semi_transparent", "translucent",
                       "semi_translucent", "opaque"]
            },

            "overall_grade": {
              "type": "string",
              "enum": ["loupe_clean", "eye_clean", "slightly_included",
                       "moderately_included", "heavily_included", "opaque_included"],
              "description": "Rough-stone clarity grade (adapted from GIA Type II/III)"
            },

            "inclusions": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                  "type": {
                    "type": "string",
                    "enum": [
                      "silk_needles", "rutile_needles", "crystal_inclusion",
                      "negative_crystal", "fingerprint", "cloud", "feather",
                      "fracture", "cavity", "twinning_plane", "color_zone_boundary",
                      "growth_tube", "halo_inclusion", "mineral_crystal",
                      "iron_staining", "surface_reaching_fracture", "lead_glass_fill",
                      "flux_residue", "other"
                    ]
                  },
                  "mineral_id": {
                    "type": ["string", "null"],
                    "description": "If crystal inclusion: e.g., 'rutile', 'zircon', 'calcite', 'apatite'"
                  },
                  "density": {
                    "type": "string",
                    "enum": ["isolated", "few", "moderate", "dense", "saturated"]
                  },
                  "location": {
                    "type": "string",
                    "enum": ["surface", "near_surface", "mid_depth", "deep", "throughout"]
                  },
                  "size_mm": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Estimated largest dimension of inclusion"
                  },
                  "impact_on_cut": {
                    "type": "string",
                    "enum": ["none", "minor_avoidable", "limits_shape",
                             "significant_weight_loss", "uncuttable"]
                  },
                  "notes": { "type": "string", "maxLength": 500 }
                }
              }
            },

            "fracture_risk": {
              "type": "object",
              "properties": {
                "level": {
                  "type": "string",
                  "enum": ["low", "moderate", "high", "critical"]
                },
                "surface_reaching_fractures": { "type": "integer", "minimum": 0 },
                "cleavage_plane_visible": { "type": "boolean" },
                "notes": { "type": "string", "maxLength": 500 }
              }
            }
          }
        },

        "estimated_yield": {
          "type": "object",
          "description": "Cutter's estimate of finished stone potential",
          "properties": {
            "weight_retention_pct": {
              "type": ["number", "null"],
              "minimum": 0,
              "maximum": 100
            },
            "best_shape": {
              "type": "string",
              "enum": ["oval", "cushion", "round", "emerald_cut", "pear",
                       "cabochon", "sugar_loaf", "bead", "carving", "multiple_stones"]
            },
            "notes": { "type": "string", "maxLength": 500 }
          }
        },

        "confidence": {
          "type": "integer",
          "minimum": 1,
          "maximum": 5,
          "description": "Grader's self-reported confidence (1=uncertain, 5=certain)"
        }
      }
    },

    "image_annotation": {
      "type": "object",
      "required": ["type", "label"],
      "properties": {
        "type": {
          "type": "string",
          "enum": ["polygon", "point", "bounding_box", "polyline"]
        },
        "label": {
          "type": "string",
          "description": "What this annotation marks, e.g., 'color_zone_red', 'inclusion_silk', 'fracture'"
        },
        "coordinates": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "x": { "type": "number" },
              "y": { "type": "number" }
            },
            "required": ["x", "y"]
          },
          "description": "Pixel coordinates. Polygon: ordered vertices. Point: single item. BBox: [top-left, bottom-right]."
        },
        "attributes": {
          "type": "object",
          "description": "Free-form attributes (e.g., color_zone hue, inclusion subtype)"
        },
        "grader_id": { "type": "string" },
        "timestamp": { "type": "string", "format": "date-time" }
      }
    }
  }
}
```

### 2.3 Validation Rules Summary

| Rule | Implementation |
|------|---------------|
| Ruby must have red-family hue | If `variety == "ruby"`, `hue` must be in `[red, slightly_purplish_red, purplish_red, strongly_purplish_red, red_orange, pinkish_red]` |
| Blue sapphire must have blue-family hue | If `variety == "blue_sapphire"`, `hue` must be in `[blue, violetish_blue, slightly_greenish_blue, greenish_blue, strongly_greenish_blue, very_slightly_violetish_blue]` |
| Pigeon blood only for ruby | `special_designation == "pigeon_blood"` requires `variety == "ruby"` |
| Royal/cornflower only for sapphire | `special_designation in [royal_blue, cornflower]` requires `variety == "blue_sapphire"` |
| Weight–dimension sanity | `weight_ct` should correlate with `dimensions_mm` volume (±40% of corundum density 4.0 g/cm³) |
| At least 2 assessments before consensus | `consensus` requires `len(assessments) >= 2` |
| Image coverage | At least one image per required lighting/angle combo |

> **Note:** Cross-field validation rules beyond JSON Schema's capability should be enforced in application code (e.g., Python `jsonschema` + custom validators or AJV with custom keywords).

---

## 3. Annotation Tool Requirements

### 3.1 Architecture Overview

Web-based SPA (React/Vue) + REST/GraphQL API + PostgreSQL + S3-compatible object store.

```
┌─────────────┐     ┌─────────────┐     ┌────────────┐
│  Grader UI  │────▶│   API       │────▶│ PostgreSQL │
│  (Browser)  │◀────│  (FastAPI)  │◀────│ + S3 Store │
└─────────────┘     └─────────────┘     └────────────┘
       │                   │
       │  WebSocket        │  Background workers
       │  (live sync)      │  (export, metrics)
       ▼                   ▼
  Collaborative        ML Training
  grading session      Pipeline (§4)
```

### 3.2 Core UI Features

#### Multi-Image Viewer
- **Grid view:** All images for one stone displayed simultaneously (thumbnail grid, click to enlarge).
- **Comparison mode:** Side-by-side panes (e.g., D65 vs. transmitted light). Synchronized zoom and pan.
- **Lighting toggle:** Quick-switch between lightings for the same angle.
- **Zoom:** Continuous zoom from fit-to-view to native pixel 1:1. Keyboard shortcuts (`+`/`-`/`0`).
- **EXIF/metadata overlay:** Pixel scale, lighting condition, capture date on hover.

#### Color Zone Polygon Annotation
- Draw polygons on any image to delineate color zones.
- Each polygon tagged with: hue label, saturation estimate, boundary sharpness (sharp/diffuse).
- Polygon operations: add vertex, delete vertex, move vertex, split, merge.
- Semi-transparent fill with hue-representative color.
- Copy annotations from one image to another (with manual adjustment).

#### Inclusion Marking Tool
- **Point marker:** Click on inclusion → select type from dropdown → set attributes.
- **Polygon/freeform:** Outline larger features (clouds, fingerprints, fractures).
- **3D estimation aid:** Mark same inclusion on multiple angle views; system estimates depth.
- Auto-generated inclusion summary table per stone.

#### Grading Form
- Structured form matching the JSON schema (§2).
- Dropdowns with visual aids (hue wheel, saturation/tone matrix thumbnail).
- Inline validation with real-time feedback.
- "Quick grade" templates for common profiles (e.g., "typical Mozambique heated ruby").
- Side panel showing the stone's images while filling the form.

### 3.3 Multi-Grader Workflow

```
           ┌──────────┐
           │  Assign   │  Pool of ungraded stones
           │  Stone    │  auto-assigned to 2-3 graders (round-robin, skill-matched)
           └────┬─────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   ┌─────────┐    ┌─────────┐
   │ Grader A │    │ Grader B │   (blind – cannot see each other's grades)
   └────┬─────┘    └────┬─────┘
        │               │
        └───────┬───────┘
                ▼
        ┌──────────────┐
        │  Auto-Compare │  Field-by-field diff
        └──────┬───────┘
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
   AGREE    MINOR Δ   MAJOR Δ
   (auto     (flag     (escalate to
   approve)  for       senior reviewer)
             review)
```

**Agreement Thresholds:**

| Field | Auto-agree if |
|-------|---------------|
| `hue` | Exact match |
| `saturation` | ±1 step |
| `tone` | ±1 step |
| `transparency` | Exact match |
| `overall_grade` (clarity) | ±1 step |
| `zoning.severity` | ±1 step |
| `special_designation` | Exact match |
| `fracture_risk.level` | Exact match |

Minor disagreement: 1–2 fields outside threshold → flag for quick review.
Major disagreement: 3+ fields or any `special_designation`/`fracture_risk` mismatch → escalate.

### 3.4 Inter-Rater Reliability Metrics

Calculated continuously and displayed on admin dashboard:

| Metric | When | Computation |
|--------|------|-------------|
| **Cohen's Kappa (κ)** | 2 graders | Per ordinal field, weighted (quadratic weights for tone/saturation) |
| **Fleiss' Kappa** | 3+ graders | For nominal fields (hue, transparency) |
| **Weighted % Agreement** | Always | Per field, aggregated |
| **ICC (Intraclass Correlation)** | 2+ graders | For continuous fields (hue_angle_deg, weight_retention_pct) |
| **Confusion Matrix** | Per pair | Heatmap of grader A vs B per field value |

**Targets:**
- κ ≥ 0.80 (substantial agreement) for production data
- κ ≥ 0.60 (moderate) triggers grader re-training
- κ < 0.40 (fair/poor) triggers workflow review and recalibration session

**Grader Calibration:**
- Monthly calibration set: 20 pre-graded reference stones, grades hidden.
- Each grader's deviation from reference tracked over time.
- Systematic biases detected (e.g., "Grader C consistently rates saturation 1 step higher") and flagged.

### 3.5 Export to Training-Ready Format

#### Classification Export (per stone)

```
output/
├── classification/
│   ├── labels.csv          # stone_id, variety, hue, saturation, tone, ...
│   ├── splits.csv          # stone_id, split (train/val/test)
│   └── images/
│       ├── S0042_L1-D65_A1-TOP.png
│       ├── S0042_L2-TRANS_A1-TOP.png
│       └── ...
```

`labels.csv` columns: `stone_id, variety, hue, hue_angle, saturation, tone, color_grade, transparency, clarity_grade, zoning_severity, special_designation, fracture_risk, treatment, origin, weight_ct`

#### Segmentation Export (per image)

```
output/
├── segmentation/
│   ├── images/
│   │   └── S0042_L1-D65_A1-TOP.png
│   ├── masks/
│   │   └── S0042_L1-D65_A1-TOP.png    # color-coded mask (each zone = unique ID)
│   ├── annotations/
│   │   └── S0042_L1-D65_A1-TOP.json   # COCO-format polygon annotations
│   └── categories.json
```

#### Detection Export (inclusions)

```
output/
├── detection/
│   ├── images/
│   ├── labels/                          # YOLO-format .txt or COCO JSON
│   └── classes.txt                      # inclusion type → class ID mapping
```

#### Export Formats Supported
- **COCO JSON** (segmentation + detection)
- **YOLO v8 txt** (detection)
- **CSV** (classification)
- **HuggingFace Datasets** (Arrow/Parquet with image bytes)
- **TFRecord** (optional)

---

## 4. Data Pipeline Architecture

### 4.1 Pipeline Stages

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ CAPTURE  │──▶│ INGEST  │──▶│  LABEL  │──▶│ REVIEW  │──▶│ EXPORT  │──▶│  TRAIN  │
│          │   │         │   │         │   │         │   │         │   │         │
│ Camera → │   │ Validate│   │ Graders │   │ QA +    │   │ Format  │   │ Model   │
│ Station  │   │ + Store │   │ annotate│   │ Resolve │   │ Convert │   │ Training│
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

#### Stage 1: Capture
- Operator captures images per protocol (§1).
- Tethered capture writes RAW files to local workstation.
- Operator enters stone metadata (weight, dimensions) on capture tablet app.
- **Output:** RAW files + metadata JSON stub on local disk.

#### Stage 2: Ingest
- Automated watcher (`watchdog` or `inotify`) detects new files.
- RAW → 16-bit TIFF conversion with ICC profile applied (`dcraw` / `rawpy` / Lightroom CLI).
- TIFF → 8-bit sRGB PNG for annotation UI.
- Validate: correct number of images per stone, filename format, EXIF sanity.
- Upload to S3 (RAW → `s3://gemdata/raw/`, TIFF → `s3://gemdata/tiff/`, PNG → `s3://gemdata/png/`).
- Create database record: stone ID, lot, image manifest, status = `ingested`.
- **Quality gates:** reject if < 6 images, if any image is blurred (Laplacian variance < threshold), if ColorChecker ΔE > 2.0.

#### Stage 3: Label
- Stone enters grading queue.
- Auto-assigned to N graders (default N=2, configurable).
- Graders work in annotation tool (§3).
- Each grader submits independently (blind).
- **Output:** N assessment JSON records per stone.

#### Stage 4: Review
- Auto-comparison engine diffs assessments.
- Agreement → auto-approve, consensus = average/mode of assessments.
- Disagreement → routed to senior reviewer with side-by-side diff view.
- Reviewer makes final call → consensus record written.
- Kappa and agreement metrics updated.
- Status → `approved`.

#### Stage 5: Export
- Triggered manually or on schedule (e.g., weekly).
- Query: all stones with `status == approved` and `schema_version == current`.
- Generate train/val/test splits (stratified by variety, origin, grade).
  - Default split: 70/15/15.
  - Ensure no lot-level leakage (all stones from one lot in same split).
- Write to export formats (§3.5).
- Version the export with semantic versioning and git tag.

#### Stage 6: Train
- Downstream (out of scope for this spec) but:
  - Export directory is a versioned artifact (DVC or MLflow Artifacts).
  - Training scripts read from export directory.
  - Model metrics traced back to dataset version.

### 4.2 Storage Structure

```
s3://gemdata/
├── raw/                          # Camera RAW files (archival, write-once)
│   └── LOT2026-001/
│       ├── S0042/
│       │   ├── S0042_L1-D65_A1-TOP_20260214.ARW
│       │   └── ...
│       └── CALIB/
│           └── CALIB_L1-D65_COLORCHECKER_20260214.ARW
├── tiff/                         # 16-bit profiled TIFF (working copies)
│   └── LOT2026-001/S0042/...
├── png/                          # 8-bit sRGB PNG (annotation UI)
│   └── LOT2026-001/S0042/...
├── exports/                      # Versioned training exports
│   ├── v1.0.0/
│   │   ├── classification/
│   │   ├── segmentation/
│   │   └── detection/
│   └── v1.1.0/
│       └── ...
└── calibration/                  # ICC profiles, ColorChecker images
    └── 2026-02-14/
        ├── profile_L1-D65.icc
        └── ...
```

**Database (PostgreSQL):**

```
stones          → id, lot_id, metadata (JSONB), status, created_at
images          → id, stone_id, filename, s3_key, lighting, angle, pixel_scale
assessments     → id, stone_id, grader_id, data (JSONB), created_at
consensus       → id, stone_id, data (JSONB), reviewer_id, created_at
annotations     → id, image_id, grader_id, type, label, coordinates (JSONB)
graders         → id, name, role, calibration_score
export_runs     → id, version, created_at, split_seed, stone_count, config (JSONB)
```

### 4.3 Version Control for Labels

| Layer | Tool | Notes |
|-------|------|-------|
| **Schema** | Git (this repo) | Schema JSON + migration scripts versioned in `schemas/` |
| **Label data** | PostgreSQL + audit log | Every assessment INSERT is immutable; edits create new version rows. `assessments` table has `version` column. |
| **Dataset exports** | DVC (Data Version Control) | `.dvc` files in git track S3 export paths. `dvc diff` shows label changes between versions. |
| **Model ↔ Data** | MLflow | Each training run logs `dataset_version` parameter. Full lineage: model → export version → label versions → images. |

**Audit trail:**
- All label writes are append-only (no UPDATE/DELETE on assessment records).
- Each record has `grader_id`, `timestamp`, `schema_version`.
- Soft-delete via `status = 'superseded'` with pointer to replacement record.
- Nightly backup of PostgreSQL to S3 with 90-day retention.

---

## Appendix A: Equipment Budget Estimate

| Item | MVP Cost (USD) | Ideal Cost (USD) |
|------|---------------|------------------|
| Camera body | $1,000 | $2,500 |
| Macro lens | $500 | $1,000 |
| D65 LED panel + diffuser | $400 | $800 |
| Transmitted light box | $200 | $500 |
| Polariser set (2×) | $100 | $200 |
| Fibre-optic illuminator | $300 | $800 |
| UV lamps (LW + SW) | $200 | $500 |
| Copy stand + rotating stage | $300 | $1,200 |
| X-Rite ColorChecker + profiler | $500 | $1,500 |
| Tethering setup | $50 | $200 |
| **Total** | **$3,550** | **$9,200** |

## Appendix B: Minimum Dataset Size Targets

| Task | Min Stones | Min Images | Rationale |
|------|-----------|------------|-----------|
| Binary (ruby vs sapphire) | 500 | 15,000 | Simple binary, high accuracy achievable |
| Color grade (6-class saturation) | 2,000 | 60,000 | Ordinal classification, need per-class coverage |
| Clarity grade (6-class) | 2,000 | 60,000 | Same reasoning |
| Special designation (pigeon blood, etc.) | 3,000+ | 90,000+ | Rare classes need oversampling; target ≥200 per rare class |
| Color zone segmentation | 500 | 2,500 | Polygon annotations expensive; 5 annotated images per stone |
| Inclusion detection | 1,000 | 10,000 | Dense annotations; moderate dataset sufficient with augmentation |

## Appendix C: Recommended Tech Stack

| Component | Recommendation |
|-----------|----------------|
| API | FastAPI (Python) |
| Database | PostgreSQL 16 + JSONB |
| Object Store | MinIO (self-hosted) or AWS S3 |
| Frontend | React + OpenLayers (for image annotation canvas) |
| Image Processing | `rawpy`, `Pillow`, `colour-science` (Python) |
| Annotation Canvas | OpenLayers or Leaflet (tile-based, handles large images) |
| Auth | Keycloak or Auth0 |
| Export Pipeline | Python scripts + DVC |
| ML Training | PyTorch + timm (vision models) |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana (pipeline health, grading throughput) |
