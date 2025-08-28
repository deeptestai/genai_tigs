---
title: Models & Storage
parent: User Guide
nav_order: 5
---

# Models & Storage

- Models are **preloaded** inside the image (no runtime downloads).
- Default paths inside container:
  - `/app/vae/`, `/app/sd/`, `/app/sa/`, `/app/cdcgan/`
- To persist results outside the container:
  
```bash
docker run --gpus all -p 7860:7860 -v /home/USER/giftbench_out:/app/output yourname/giftbench:v1.2.1
```

## RUN from CLI
If you want to run GIFTbench directly from the command line (without Docker),  
make sure the required model folders are placed **inside the cloned repository**.
1. Clone the repository:
   ```bash
   git clone https://github.com/deeptestai/genai_tigs.git
   cd genai_tigs
   git checkout tool

   ```
2. Download the required model zip files (VAE, SD, SSA, CDCGAN) from the provided links under QuickStart guideline.
   Unzip them and place the folders directly inside the cloned repo, e.g.:


```text
genai_tigs/
├── vae/          # VAE model files
├── sd/           # Stable Diffusion model files
├── ssa/          # SSA model files
├── cdcgan/       # CDCGAN model files
├── GIFTbench.py  # Main Gradio-based CLI entry point
└── ...           # Other source files and docs
```
