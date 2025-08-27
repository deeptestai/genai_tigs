
---
title: Hardware & Network Requirements
parent: User Guide
nav_order: 2
---

# Hardware & Network Requirements

## Hardware
- **GPU (NVIDIA):** ≥ **16 GB VRAM** (24 GB recommended).
- **System RAM:** ≥ **16 GB** (32 GB recommended).
- **Disk Space:** ≥ **30 GB free** (image + preloaded models + cache).

## Software
- **OS:** Linux x86_64.
- **Docker:** 24+ with GPU support.
- **NVIDIA Drivers:** 525+ (production branch).
- **NVIDIA Container Toolkit:** configured (`docker run --gpus all ...` works).

## Network / Firewall
- **Outbound:** needed once to pull the image.
- **Inbound (optional, if accessed remotely):**
  - **7860/tcp** → Gradio UI
- If inbound must stay closed, use an **SSH tunnel** instead.

## Quick Checks
```bash
nvidia-smi
```
To check hard disk space, follow this command:
```bash
df -h
```
