# GemGrade — Rough Stone Annotation Tool

A web-based grading and annotation tool for rough gemstones (ruby & blue sapphire), following GIA-style color grading adapted for rough stones.

## Quick Start

```bash
cd tools/annotation-tool
python3 -m http.server 8080
```

Then open **http://localhost:8080** in your browser.

## Features

- **Stone Browser** — Filter by variety (ruby/sapphire), search by ID or origin
- **Multi-Image Viewer** — Grid view of all lighting conditions (D65, transmitted, cross-polarized, darkfield, UV) with lightbox zoom
- **Color Grading** — Hue, saturation (1-6), tone (1-7), distribution, special designations (pigeon blood, royal blue, cornflower)
- **Clarity Grading** — Transparency, inclusion types (multi-select), density, location, fracture risk
- **Annotations** — Draw polygon color zones on images (toggle with ✏️ button)
- **Import/Export** — Load stones from JSON, export all grades as JSON
- **Keyboard Shortcuts** — Arrow keys to navigate, Enter to save, S to skip, Esc to close

## Data Format

Import a JSON file as an array of stone objects:

```json
[
  {
    "id": "RB-2024-001",
    "lotId": "LOT-A1",
    "variety": "ruby",
    "weight": 3.42,
    "dimensions": "8.1 x 6.3 x 5.2",
    "shape": "rough_crystal",
    "origin": "Mogok, Myanmar",
    "treatment": "none",
    "images": [
      { "url": "path/to/image.jpg", "lighting": "D65", "angle": "top" }
    ]
  }
]
```

Supported lighting types: `D65`, `transmitted`, `cross_polarized`, `darkfield`, `UV`

## Grades Storage

Grades are saved to `localStorage` for the MVP. Use **Export** to download all grades as a JSON file for permanent storage.

## Annotation Mode

1. Click the **✏️ Annotate** button in the header
2. Click on an image to place polygon points
3. Double-click to close the polygon and add a label
4. Press **Esc** to cancel current polygon

## Tech Stack

Vanilla HTML/CSS/JS — no build step, no dependencies. Just a static file server.
