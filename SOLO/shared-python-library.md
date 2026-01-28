# Shared Python Library Strategy (Single Common Library Across Projects)

This document outlines a practical approach for maintaining **one shared, multi‑gigabyte Python library** (e.g., Torch/CUDA stacks) across many projects, while still keeping per‑project environments lightweight.

The goal is **shared runtime packages**, not just fast installs.

## Recommended Pattern: Base Env + Per‑Project Venvs That Inherit It

Create one **base environment** that contains the heavyweight packages, then create **project venvs** that can see those packages via `--system-site-packages`.

### 1) Create a shared base environment
```bash
python3 -m venv ~/venvs/hunyuan-base
source ~/venvs/hunyuan-base/bin/activate
pip install --upgrade pip
pip install torch torchvision  # add your heavy shared libs here
```

### 2) Create a project venv that inherits the base site‑packages
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### 3) Install only project‑specific deps
```bash
pip install -r requirements.txt
```

**Result:**
- Heavy packages live in `~/venvs/hunyuan-base` once.
- Each project has its own `.venv` and can install only the deltas.

## Using `uv` With This Pattern

`uv` can be used inside the project venv for dependency management, but the **inheritance is handled by the venv creation** (`--system-site-packages`), not by `uv` itself.

Typical flow:
```bash
# Create base
python3 -m venv ~/venvs/hunyuan-base
source ~/venvs/hunyuan-base/bin/activate
uv pip install torch torchvision

# Create project venv that sees base
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# Install project deps with uv
uv pip install -r requirements.txt
```

## Why This Works

`--system-site-packages` tells the project venv to **include the site‑packages from the interpreter that created it**. If you create the project venv *using the base env’s Python*, the project venv will see everything from the base env.

If you want to be explicit, create the project venv using the base env’s Python:
```bash
~/venvs/hunyuan-base/bin/python -m venv --system-site-packages .venv
```

## Stability & Safety Tips

- **Pin the base env**: Treat it like a shared runtime. Changes can affect all projects.
- **Don’t mix system Python**: Keep all shared libs in a single base env you control.
- **Project-specific overrides**: If you need a different version in one project, install it in that project venv; it will shadow the base package.

## When This Pattern Is Not Ideal

- If projects require **different CUDA/PyTorch builds** that conflict.
- If you need **strict isolation** (e.g., reproducibility for deployment).

In those cases, use separate full venvs per project or a dedicated environment manager (conda/mamba) per project.

---

This setup gives you **one shared heavyweight library** with **lightweight per‑project environments**, matching the “single common library” goal.
