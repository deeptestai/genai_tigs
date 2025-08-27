---
title: Getting Started
nav_order: 2
---

# Getting Started

This page gets you from zero → running **GIFTbench** in minutes.

---

## 1) Prerequisites

### Hardware
- **NVIDIA GPU:** ≥ **16 GB VRAM** (24 GB recommended for Diffusion)
- **Disk space:** ≥ **30 GB free** (image + preloaded models + cache)
- **RAM:** ≥ 16 GB (32 GB recommended)

### Software
- **Linux x86_64** host (WSL2 on Windows also works)
- **Docker** 24+  
- **NVIDIA drivers** (e.g., 525+)  
- **NVIDIA Container Toolkit** (for `--gpus all`)

**Quick checks (host):**
```bash
nvidia-smi
docker --version

