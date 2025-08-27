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

