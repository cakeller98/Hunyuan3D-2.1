# PyTorch MPS Compatibility Research for Hunyuan3D-2.1

This document contains research on running Hunyuan3D-2.1 on macOS using PyTorch's Metal Performance Shaders (MPS) backend as an alternative to CUDA.

Research conducted: January 2026

## PyTorch MPS Overview

**MPS (Metal Performance Shaders)** is Apple's GPU acceleration framework for macOS. PyTorch has officially supported MPS since version 1.12, with the latest version being PyTorch 2.10 (as of June 2025).

### Installation

Standard installation (includes MPS support):
```bash
pip3 install --upgrade torch torchvision torchaudio
```

For nightly/preview build with latest MPS features:
```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

Verify MPS availability:
```python
import torch
print(torch.backends.mps.is_available())
```

### System Requirements

- macOS 10.15 (Catalina) or above
- Python 3.10 - 3.14 recommended
- Apple Silicon (M1/M2/M3) or Intel Mac with Metal support
- Xcode 13.3.1 or later (for building from source)
- MPS support available from torch>=1.13 onwards

### Performance Benefits

- **4.7x faster** than CPU-only execution for transformer-based pipelines
- Leverages Apple Silicon's unified memory architecture (M1/M2/M3)
- GPU gets direct access to full memory store
- However, performs at **~25-30% the speed** of mid-range NVIDIA CUDA GPUs
- Best performance with lightweight models and smaller batch sizes

## Key Limitations vs CUDA

### 1. Missing Operations
- Some PyTorch operations not yet implemented for MPS backend
- CUDA-specific libraries (like FlashAttention, bitsandbytes) have no MPS equivalent
- Requires fallback to CPU for unsupported operations
- Compatibility errors with operations that aren't fully supported by Metal Performance Shaders

### 2. API Differences
- CUDA: `x.cuda()`
- MPS: `x.to(torch.device('mps'))`
- Less seamless than CUDA integration

### 3. Numerical Accuracy Issues
- Models that converge on CUDA may fail to converge on MPS
- Different loss values produced compared to CUDA
- Training results can be poor compared to both CPU and CUDA

### 4. Memory Layout
- Metal initially didn't support strided access
- Requires contiguous memory layouts for some operations
- Some MPS operations implemented workarounds while others didn't
- Improved in newer Mac Operating Systems

### 5. Hardware Limitation
- Only uses GPU portion of Apple Silicon
- Neural Engine sits idle (no PyTorch integration yet)
- Suggests potential for future improvements

### 6. Maturity
- MPS backend is not as mature as CUDA backend
- Fewer community resources and troubleshooting guides
- Some edge cases and operations may have bugs

### 7. Known Issues
- macOS 26.0 (Tahoe) with PyTorch 2.9.1 and 2.10 nightly: MPS is built but not available

## CuPy Alternatives for macOS

For the **`cupy-cuda12x`** dependency in Hunyuan3D-2.1, you have these alternatives:

### 1. PyTorch with MPS (Most Compatible)
- Use PyTorch tensors instead of CuPy arrays
- Native MPS support via `device = torch.device("mps")`
- Most straightforward migration path
- Good ecosystem support

### 2. MLX (Apple's Native Framework - Recommended for New Projects)
- Developed specifically for Apple Silicon by Apple
- NumPy-like API for easy migration
- **1.59x faster** than PyTorch MPS for some workloads
- Uses lazy evaluation and operation fusion
- Fully exploits the SoC architecture
- GitHub: https://github.com/ml-explore/mlx

### 3. Standard NumPy (CPU Fallback)
- No GPU acceleration but universal compatibility
- Simplest migration but slowest performance
- Good for testing/debugging

### 4. JAX with MPS
- JAX can work with MPS backend
- More complex setup
- Good for research-oriented code

## Diffusion Models on MPS

Good news for Hunyuan3D-2.1's diffusion pipelines:

### Stable Diffusion Compatibility
- **Stable Diffusion works well on Apple Silicon** via MPS
- HuggingFace Diffusers library is compatible with MPS
- **10-20x speedup** over CPU with PyTorch 2.0+ MPS
- Multiple production implementations exist:
  - AUTOMATIC1111 (best features, harder to install)
  - Draw Things (easiest to install, good features)
  - Diffusers (easiest to install, fewer features)
  - DiffusionBee (easy to install, smaller feature set)
  - MochiDiffusion (native Mac app)

### Apple's Core ML Optimizations
- Apple released Core ML optimizations for Stable Diffusion in macOS 13.1 and iOS 16.2
- Extremely memory efficient: ~150MB with Neural Engine
- Maximum performance and speed on Apple Silicon
- Reduced memory requirements

### Performance Tips for Diffusion on Mac
- Attention slicing usually improves performance by ~20%
- Less effective on systems with 64GB+ RAM
- M1/M2/M3 performance very sensitive to memory pressure
- System swapping significantly degrades performance when memory pressure occurs
- Works best with smaller batch sizes
- Monitor memory usage closely

### Important 2026 Update
- As of January 2026, the current master branch of some Stable Diffusion implementations stops working
- Use the dev branch instead for latest compatibility

## Implications for Hunyuan3D-2.1 on macOS

### CUDA-Only Dependencies to Replace

From `requirements.txt` and README:

1. **`cupy-cuda12x==13.4.1`**
   - Replace with: PyTorch tensors or MLX arrays
   - Migration effort: Moderate (need to refactor array operations)

2. **`torch==2.5.1+cu124` (CUDA-specific)**
   - Replace with: `torch` (standard MPS-compatible version)
   - Migration effort: Easy (just change install command)

3. **`deepspeed`**
   - Status: Commonly Linux/CUDA-only
   - Replace with: May need to remove or mock
   - Migration effort: Complex (if heavily integrated)

### Code Modifications Required

1. **Device Management**
   ```python
   # CUDA version
   device = torch.device('cuda')
   model = model.cuda()

   # MPS version
   device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
   model = model.to(device)
   ```

2. **CuPy to PyTorch/NumPy Migration**
   - Convert CuPy array operations to PyTorch tensor operations
   - Or use NumPy with CPU fallback

3. **DeepSpeed Handling**
   - Remove DeepSpeed optimization code
   - May impact training performance (less relevant for inference)

4. **Error Handling**
   - Add fallbacks for unsupported MPS operations
   - Implement CPU fallback where needed

### Expected Performance

#### VRAM Requirements (from platform-notes.md)
- Shape generation: ~10GB VRAM
- Texture generation: ~21GB VRAM
- Full pipeline: ~29GB VRAM

#### Mac Compatibility
- **M1 Max/Pro**: 32GB unified memory (may struggle with full pipeline)
- **M1/M2 Ultra**: 64GB+ unified memory (better chance)
- **M3 Max**: Up to 128GB unified memory (most viable)

#### Speed Expectations
- **~70-75% slower** than NVIDIA A6000 on CUDA
- Shape generation: Feasible on M1 Max or better
- Texture generation: May require M1 Ultra or M3 Max
- Full pipeline: Challenging, requires high-end Mac

### Potential Challenges

1. **Numerical Stability**
   - Diffusion models may produce different results on MPS vs CUDA
   - Need thorough testing and validation

2. **Memory Management**
   - Unified memory helps but can't match dedicated VRAM
   - Memory pressure will severely impact performance
   - May need to reduce batch sizes

3. **Unsupported Operations**
   - Some operations may fall back to CPU
   - Need to identify and optimize hotspots

4. **No Official Support**
   - Untested configuration
   - Community support may be limited
   - Debugging will be more difficult

5. **DeepSpeed Dependency**
   - If critical to model architecture, removal may break functionality
   - Need to investigate actual usage in codebase

## Migration Strategy

### Phase 1: Dependency Analysis
1. Audit actual usage of CuPy in codebase
2. Identify DeepSpeed integration points
3. Map CUDA-specific operations

### Phase 2: Environment Setup
1. Create Mac-compatible requirements.txt
   - Replace CUDA PyTorch with standard PyTorch
   - Remove cupy-cuda12x
   - Remove or mock deepspeed
2. Test installation on macOS
3. Verify MPS availability

### Phase 3: Code Migration
1. Replace CuPy operations with PyTorch/NumPy
2. Update device management code
3. Add MPS/CPU fallback logic
4. Remove DeepSpeed dependencies

### Phase 4: Testing
1. Test shape generation pipeline
2. Test texture generation pipeline
3. Validate output quality vs CUDA version
4. Profile performance bottlenecks
5. Optimize memory usage

### Phase 5: Documentation
1. Document Mac-specific setup
2. Create troubleshooting guide
3. Document performance differences
4. Note any quality differences

## Verdict

**Running Hunyuan3D-2.1 on macOS is theoretically possible** with significant modifications:

### Feasibility: Medium
- Diffusion models work well on MPS (positive signal)
- Multiple dependencies need replacement (complexity)
- No official support (risk)
- Performance will be reduced (acceptable for local dev)

### Recommended Configuration
- **M1 Ultra or M3 Max** with 64GB+ unified memory
- macOS 13.1 or later (for Core ML optimizations)
- PyTorch 2.0+ with MPS support
- Patience for slower inference times

### Best Use Cases on Mac
1. **Development and testing** (acceptable slower speed)
2. **Shape-only generation** (lower VRAM requirements)
3. **Small batch processing** (memory constraints)
4. **Demo/proof-of-concept** (not production)

### Not Recommended on Mac
1. **Production workloads** (use Windows + A6000)
2. **Batch processing** (too slow)
3. **Full texture pipeline on <64GB RAM** (VRAM limitations)

## Alternative: Hybrid Approach

**Recommended**: Use both platforms for their strengths

1. **Mac (Development & Light Testing)**
   - Code development and debugging
   - Quick shape-only tests
   - UI/UX development for Blender addon or Electron app
   - Blender rendering pipeline (already documented for Mac)

2. **Windows + A6000 (Production Inference)**
   - Full shape + texture pipeline
   - Production API server
   - Batch processing
   - Final quality rendering

3. **Communication Layer**
   - Develop on Mac, deploy to Windows server
   - Use API calls to Windows machine for heavy inference
   - Mac runs frontend, Windows runs backend

This hybrid approach maximizes productivity while maintaining performance where it matters.

## Sources

- [Accelerated PyTorch training on Mac - Apple Developer](https://developer.apple.com/metal/pytorch/)
- [Introducing Accelerated PyTorch Training on Mac – PyTorch](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
- [MPS backend — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [How to Run PyTorch on a MacOS GPU with Metal](https://apxml.com/posts/pytorch-macos-metal-gpu/)
- [Metal Performance Shaders (MPS) - HuggingFace](https://huggingface.co/docs/diffusers/en/optimization/mps)
- [torch.mps — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/mps.html)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [Setting up PyTorch on Mac M1 GPUs](https://geekmonkey.org/setting-up-jupyter-lab-with-pytorch-on-a-mac-with-gpu/)
- [Array operations on Apple Silicon GPU alternatives](https://forum.image.sc/t/array-operations-on-apple-silicon-gpu-alternatives-to-cupy-jax/54689)
- [Scientific Python on M1 Macbook pro](https://carreau.github.io/posts/scientific-python-on-m1-macbook-pro.md/)
- [MLX-MASt3R: 1.59x Faster 3D Reconstruction on Apple Silicon](https://medium.com/@aedelon/mlx-mast3r-native-3d-reconstruction-apple-silicon-7ac819bc0426)
- [How to install and run Stable Diffusion on Apple Silicon Macs](https://stable-diffusion-art.com/install-mac/)
- [Apple ml-stable-diffusion GitHub](https://github.com/apple/ml-stable-diffusion)
- [Installation on Apple Silicon - AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)
- [Stable Diffusion with Core ML on Apple Silicon - Apple ML Research](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon)
- [How to use Stable Diffusion in Apple Silicon (M1/M2)](https://huggingface.co/docs/diffusers/v0.5.1/en/optimization/mps)
- [Run Stable Diffusion 3 on Apple Silicon Mac](https://replicate.com/blog/run-stable-diffusion-3-on-apple-silicon-mac)
- [MochiDiffusion GitHub](https://github.com/MochiDiffusion/MochiDiffusion)
- [Stable Diffusion on Apple Silicon: M1/M2/M3 Setup](https://neurocanvas.net/blog/stable-diffusion-apple-guide/)

---

**Document Version**: 1.0
**Last Updated**: January 27, 2026
**Research Status**: Complete - ready for implementation planning
