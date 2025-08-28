---
title: Run with Docker
parent: User Guide
nav_order: 4
---

# Run with Docker

## Pull & run
```bash
docker pull maryam483/giftbench:v1.2.0

docker run --name giftbench-running --gpus all -p 7860:7860 maryam483/giftbench:v1.2.0

```
Once the container starts, the **Gradio app will automatically launch** inside the container and provide access URLs in the logs.

At the end of the log output, you will see two types of URLs:

- **Local URL:**  
  [http://0.0.0.0:7860] 
  → Accessible from your browser on the same machine where Docker is running.

- **Public URL (optional):**  
  A temporary link of the form:  
  `https://abcdef12345.gradio.live`  
  → Generated if `share=True` is set in `GIFTbench.py`.  
   This link expires after 72 hours or when the container stops.

---

###  Summary
- Use the **Local URL** for development or testing on the same machine.  
- Use the **Public URL** if you enabled sharing and need to test on another device or share access quickly.  
- For permanent external access, expose port **7860** and access via provided links.
