# Hunyuan3D-2.1 Platform Notes (Mac vs Windows)

This document summarizes what the current repo indicates about platform support and how to run the model locally without Docker. It is based strictly on the existing markdown files and `requirements.txt` files in this repo.

## Platform Signals Found in the Repo

- `README.md` says Hunyuan3D 2.1 **supports MacOS, Windows, Linux**, but the install commands are CUDA-only (`torch==2.5.1+cu124` from NVIDIA’s index), which implies Windows/Linux NVIDIA usage.
- `docker/README.md` says the Docker setup is **tested on Windows 10** (not relevant since Docker is not desired).
- `hy3dshape/tools/README.md` includes a **Mac-specific Blender Python install** example and states Blender 4.1 is required for data/rendering pipeline.
- VRAM recommendations:
  - Shape generation: **~10GB VRAM**
  - Texture generation: **~21GB VRAM**
  - Full pipeline: **~29GB VRAM**

## What Likely Works on Windows (A6000)

- **Core inference (shape + texture)**: The main `README.md` install flow is CUDA-based and fits an NVIDIA GPU workstation.
- **Texture generation**: `hy3dpaint` uses Real-ESRGAN weights and heavy VRAM; the A6000 should handle it well.
- **Local server (FastAPI)**: `API_DOCUMENTATION.md` + `API_TESTING_SUMMARY.md` indicate `api_server.py` is intended to run locally.

## What Likely Does Not Work on Mac As Written

- CUDA-only dependencies in `requirements.txt`:
  - `cupy-cuda12x==13.4.1`
  - README’s pinned CUDA PyTorch install
  - `deepspeed` (commonly Linux/CUDA-only)
- VRAM guidance assumes NVIDIA CUDA GPUs.

## What Likely Can Work on Mac With Changes

- **Blender rendering pipeline** (data/tools) is explicitly shown on Mac in `hy3dshape/tools/README.md`.
- **Inference on Mac** is not documented; would require a Mac-compatible PyTorch install and removing CUDA-only deps.

## Practical Cross‑Platform Strategy

1. **Windows workstation = full pipeline (shape + texture + PBR maps)**
   - Use CUDA-based install path from repo.
   - Use `api_server.py` for local server.

2. **Mac = shape-only or limited inference**
   - Requires Mac-compatible PyTorch build.
   - Remove/replace CUDA-only dependencies.
   - Expect slower performance.

## How the Model Works (From Repo Docs)

- **Stage 1: Shape generation** (`Hunyuan3DDiTFlowMatchingPipeline`)
  - Single image in → untextured mesh (`.glb`).
- **Stage 2: Texture generation** (`Hunyuan3DPaintPipeline`)
  - Mesh + image → textured mesh with PBR materials.
- **Training/data pipeline** (Blender + watertight mesh + SDF sampling) described in `hy3dshape/tools/README.md` is for training, not required for inference.

## Local‑Only / No‑Docker Alignment

- The repo already includes `api_server.py` for a local FastAPI server.
- A Blender addon or Electron app can call the local `/generate` or `/send` endpoints.

---

This document captures “how it works now” to help evaluate before porting into a new project.
