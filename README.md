# Crimson Skies HD Texture Upscaling & Vector Reconstruction

A comprehensive high-definition texture upgrade, AI redraw pipeline, and vector reconstruction system for the 2000 PC flight combat game **Crimson Skies** and its modern open-source engine reimplementation, [CSVM](https://github.com/Laeresh/CSVM) (Godot 4).

This repository contains:
1. **The Final HD Textures**: Ready-to-use high-resolution assets for all in-world textures (aircraft, terrain, cockpits, gauges, damage indicators, sky, water) and 2D UI/menus.
2. **Handcrafted Vector Sources (SVG)**: Pixel-perfect vector reconstructions of dials, gauges, buttons, blueprint schematics, map pins, and HUD elements.
3. **The AI Upscaling Pipeline**: Automated VLM prompt generation, FLUX.2 image editing, frequency-separation color matching, chroma-keying, and review tools.
4. **The Mobile Review Web App**: A mobile-friendly swipe review interface with live undo/redo, auto-crop, alpha-fix, and reject-priority queueing.
5. **CSVM Engine Patch**: Minimal patch and metadata manifest enabling high-res UI & HUD elements in CSVM without layout distortion.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [How to Install & Play in CSVM](#how-to-install--play-in-csvm)
  - [1. In-World HD Texture Pack](#1-in-world-hd-texture-pack)
  - [2. High-Resolution UI & HUD (with CSVM Engine Patch)](#2-high-resolution-ui--hud-with-csvm-engine-patch)
- [How the Pipeline Works](#how-the-pipeline-works)
  - [1. Task Partitioning & Cataloging](#1-task-partitioning--cataloging)
  - [2. Vision-Language Model Prompt Generation](#2-vision-language-model-prompt-generation)
  - [3. Diffusion Image Editing (FLUX.2 klein 9B)](#3-diffusion-image-editing-flux2-klein-9b)
  - [4. Frequency Separation & Chroma Keying](#4-frequency-separation--chroma-keying)
  - [5. Vector (SVG) Reconstruction](#5-vector-svg-reconstruction)
  - [6. Human-in-the-Loop Web Review](#6-human-in-the-loop-web-review)
- [Running the Pipeline Locally](#running-the-pipeline-locally)
- [License & Credits](#license--credits)

---

## Project Overview

*Crimson Skies* (released by Microsoft in 2000) features alternate-1930s dieselpunk aviation, zeppelins, and aerial piracy. The original game assets were authored for software/early Direct3D renderers at resolutions ranging from 16×16 to 256×256 pixels.

Rather than relying on naive ESRGAN-style sharpening—which blurs text, hallucinates noisy plastic textures on cloth and metal, and leaves jagged cutout edges—this project uses a **hybrid AI-redraw and vector reconstruction pipeline**:

- **World Textures (1,488 unique assets)**:
  - **1,456** upgraded to 8× resolution (up to 2048×2048).
  - **32** flat utility/shadow assets intentionally kept at pristine original values.
- **UI & Menu Art (668 unique assets)**:
  - **576** upgraded with crisp vector SVG renders and high-resolution redraws.
  - **92** fullscreen backgrounds (≥ 720 px) and flat color elements preserved.
- **Vector Reconstruction (216 packages)**:
  - Instrument dials (altimeter, horizon indicator, speedometer, nitro, fuel gauges).
  - Damage status indicators for all player and enemy aircraft.
  - Multipart 2D UI panels, button strips, and tabs (`MP_LOBBY`, `PAPERBUTTON`, `PX_TAB`).
  - Aircraft blueprint schematics (both flat drafting sheets and torn scrapbook props).
  - Numbering, lettering, brush strokes, map pins, and HUD reticles.

---

## Repository Structure

```
├── textures/
│   ├── chosen/
│   │   ├── world/          # 1,488 ready-to-use in-world HD textures (PNG)
│   │   └── ui/             # 668 UI, menu, and scrapbook assets (PNG)
│   └── vectors/            # 216 vector packages: source .svg, .png render, and custom.json
├── tools/
│   ├── build_pack.py       # Assembles loose-folder chapter tiers (rtextureN) for CSVM
│   ├── export_winners.py   # Extracts all approved winner textures into clean folders
│   ├── link_ui_art.py      # Deploys HD UI art to CSVM with automatic .orig_bak backups
│   ├── serve_gallery.py    # Multi-threaded web review server (swipe review, crop tool)
│   ├── pipeline/
│   │   ├── run.py          # Core pipeline runner (batching, VRAM management, retry queues)
│   │   ├── finalize.py     # Alpha matting, silhouette validation & chroma-key cleanup
│   │   ├── colormatch.py   # Frequency separation in Lab color space
│   │   ├── greenkey.py     # Soft green-screen chroma keying with color unmixing
│   │   ├── magentakey.py   # Magenta-screen keying for green subjects/cast shadows
│   │   ├── flipsheet.py    # Multi-frame sprite sheet generator (keeps character identity)
│   │   ├── review.html     # Mobile-optimized swipe review application
│   │   └── live.html       # Real-time pipeline status dashboard
│   └── qwen/
│       ├── vlm_prompt.py   # VLM prompt writer (analyzes original, palette, OCR, context)
│       ├── vlm_review.py   # Automated VLM review pass
│       ├── klein_server.py # Persistent Vulkan sd-server manager for FLUX.2 klein 9B
│       └── klein.sh        # CLI invocation for stable-diffusion.cpp
├── tasks/                  # Task definitions and original low-resolution reference PNGs
├── patches/
│   ├── csvm_hd_ui_support.patch # Engine patch for CSVM (Godot 4) to support HD UI/HUD
│   └── art_sizes.json      # Authored 800x600 dimension manifest for UI scaling
├── catalog.json            # Master database of in-world texture metadata & mip relationships
├── ui_catalog.json         # Master database of UI & menu asset metadata
└── pipeline/
    ├── config.json         # Pipeline configuration (batch size, seeds, priority queues)
    └── reviews.jsonl       # Complete append-only review log (decisions, prompt history, scores)
```

---

## How to Install & Play in CSVM

[CSVM](https://github.com/Laeresh/CSVM) is the open-source Linux/Windows reimplementation of Crimson Skies. It prefers loose unpacked folders over archive zips, making texture replacement non-destructive.

### 1. In-World HD Texture Pack

To build and install the in-world textures directly into your game installation:

```bash
# 1. Build the chapter packs from approved winners (creates pack/C1 ... pack/C5)
python3 tools/build_pack.py build

# 2. Symlink the top-tier texture directory for each chapter into CSVM
python3 tools/build_pack.py install
```

This creates clean symlinks in your extracted game directory:
- `~/Games/CSVM/extracted/C1/rtexture15  -> pack/C1/rtexture15`
- `~/Games/CSVM/extracted/C1B/rtexture11 -> pack/C1B/rtexture11`
- `~/Games/CSVM/extracted/C1C/rtexture10 -> pack/C1C/rtexture10`
- `~/Games/CSVM/extracted/C2/rtexture14  -> pack/C2/rtexture14`
- `~/Games/CSVM/extracted/C2B/rtexture9   -> pack/C2B/rtexture9`
- `~/Games/CSVM/extracted/C3/rtexture12  -> pack/C3/rtexture12`
- `~/Games/CSVM/extracted/C4/rtexture14  -> pack/C4/rtexture14`
- `~/Games/CSVM/extracted/C5/rtexture14  -> pack/C5/rtexture14`

*To uninstall at any time:* `python3 tools/build_pack.py uninstall` safely removes only the symlinks.

### 2. High-Resolution UI & HUD (with CSVM Engine Patch)

In the retail 2000 engine, 2D menus were authored on an $800 \times 600$ layout grid (`LAYOUT.CSV`). Because widget dimensions were not authored in the CSV, the engine inferred widget bounding boxes directly from bitmap pixel dimensions.

To use high-resolution (8×) UI textures without inflating buttons or hitboxes:

1. **Apply the Engine Patch**:
   Copy `patches/csvm_hd_ui_support.patch` to your CSVM source tree and apply it:
   ```bash
   cd path/to/CSVM
   git apply /path/to/crimson-texture-upscale/patches/csvm_hd_ui_support.patch
   dotnet build CSVM/CSVM.csproj -c Release
   ```
2. **Deploy the Art Manifest**:
   Copy `patches/art_sizes.json` to your extracted game directory:
   ```bash
   cp patches/art_sizes.json ~/Games/CSVM/extracted/rof/art_sizes.json
   ```
3. **Install the HD UI Assets**:
   ```bash
   python3 tools/link_ui_art.py install
   ```
   *(This tool backs up any replaced retail file with an `.orig_bak` extension. Running `python3 tools/link_ui_art.py uninstall` restores all original files).*

---

## How the Pipeline Works

```
[Low-Res Original] 
       │
       ├──► 1. Analyze (OCR, Color Grid, Alpha Type, Edge Seam Ratio)
       │
       ├──► 2. VLM Prompt Writer (Gemini / Qwen3.8-Flash-Next)
       │         └─► Imperative prompt describing materials, lighting, details
       │
       ├──► 3. FLUX.2 klein 9B (Image-to-Image via stable-diffusion.cpp)
       │         └─► 4-step Euler distilled generation on green/magenta screen
       │
       ├──► 4. Post-Processing
       │         ├─► Lab Frequency Separation (Color matching)
       │         └─► Chroma-Keying & Alpha Matting (Feathered despill)
       │
       └──► 5. Web Review Interface (Phone/Desktop Swipe UI)
                 ├─► Approve (Ships to pack)
                 ├─► Reject (Immediate feedback reprompt to front of queue)
                 └─► Custom Vector Upgrade (Subagent SVG generation)
```

### 1. Task Partitioning & Cataloging
- Textures are cataloged into `catalog.json` by SHA-1 hash.
- Mipmap siblings (`_1`, `_2`) are mapped to parent units.
- Flipbook animation sequences (e.g. `bmanrun`, `wtr00000..00031`) are grouped into indivisible tasks.

### 2. Vision-Language Model Prompt Generation
- `tools/qwen/vlm_prompt.py` composites the original onto a neutral background (#00FF00 green or #FF00FF magenta).
- Tesseract OCR extracts any visible lettering.
- The VLM receives the original image, a brightened version (for dark night textures), OCR text boxes, and context from sibling textures in the same set.
- It produces an imperative prompt describing the subject as a real-world object while retaining composition, proportions, and lighting.

### 3. Diffusion Image Editing (FLUX.2 klein 9B)
- Driven by `tools/qwen/klein_server.py`, wrapping `sd-server` from `stable-diffusion.cpp`.
- Generates 1024×1024 image-to-image edits using 4-step Euler sampling with Q8_0 quantized weights.
- Multi-seed generation (seeds 1 & 2) produces alternative interpretations for review.

### 4. Frequency Separation & Chroma Keying
- **Color Matching (`colormatch.py`)**: Redraws often suffer from slight color shifts. The pipeline performs frequency separation in CIE Lab: high frequencies (fine texture detail) come from the AI redraw, while low frequencies (broad lighting and color tint) are pulled back to the original.
- **Chroma Keying (`finalize.py`, `greenkey.py`)**: Cleanly removes the background screen with soft color unmixing, preventing green halos around fine wires, antennas, and hair.
- **Alpha Bloom (`bloom_alpha.py`)**: For glowing particles (fire, muzzle flashes), alpha is expanded to preserve soft visual radiance.

### 5. Vector (SVG) Reconstruction
For cockpit instruments, gauges, typography, blueprints, and UI buttons, AI diffusion models often warp straight lines and misspell text. These assets were redrawn as clean vector SVGs:
- **Cockpit Gauges**: Dial tick marks placed mathematically in polar coordinates, accurate warning arcs, and authentic typography.
- **Aircraft Blueprints**: Procedural graph-paper grid matching retail spacing and color, layered vector linework, and translucent wing/fuselage fills.
- **Formations & Silhouettes**: Single master aircraft outline fitted via affine transforms with layered drop shadows.
- **Automated Verification**: Every SVG is rendered via CairoSVG and checked against the original using automated code:
  - Alpha mask Intersection over Union (IoU $\ge 0.95$).
  - Mean color difference (CIE Lab $\Delta E \le 6.0$).
  - Subpixel edge-alignment verification.

### 6. Human-in-the-Loop Web Review
The custom web gallery (`tools/serve_gallery.py`) serves a responsive UI:
- **Swipe Controls**: Swipe right or press `Y` to approve; swipe left or press `N` to reject.
- **Version Toggling**: Spacebar or tap to instantly blink between the original and the candidate.
- **Immediate Reject Priority**: When an asset is rejected with an issue tag (e.g. "lettering", "cartoon", "tiling"), the feedback is immediately sent to Gemini, and a new prompt is inserted at the front of the GPU render queue.
- **Interactive Tools**: Built-in interactive crop tool, gamma lifter, and alpha-fix options.

---

## Running the Pipeline Locally

### Prerequisites
- Linux (SteamOS / Arch / Ubuntu) with Python 3.10+.
- Modern GPU with Vulkan support (8 GB+ VRAM recommended).
- `stable-diffusion.cpp` built with Vulkan support.
- FLUX.2 klein 9B model files (`flux-2-klein-9b-Q8_0.gguf`, `flux2-vae.safetensors`, `Qwen3-8B-Q8_0.gguf`).
- CairoSVG, OpenCV, NumPy, scikit-image, Pillow.

### Setup
```bash
# Clone the repository
git clone https://github.com/phly95/crimson-texture-upscale.git
cd crimson-texture-upscale

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install pillow numpy opencv-python scikit-image cairosvg
```

### Launching the Review Web App
```bash
python3 tools/serve_gallery.py
# Serves at http://localhost:8765/<token>/live/review/
```

### Starting the Pipeline
```bash
# Run pipeline batch processor
python3 tools/pipeline/run.py
```

---

## License & Credits

- Original *Crimson Skies* game assets &copy; 2000 Microsoft Corporation / FASA Studio.
- CSVM open-source game engine by [Laeresh](https://github.com/Laeresh/CSVM) (GPL-3.0).
- Texture upscaling, vector reconstructions, and pipeline tooling created by [phly95](https://github.com/phly95).
- Pipeline code and tools in this repository are licensed under the MIT License.
