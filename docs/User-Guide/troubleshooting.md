---
title: Troubleshooting
parent: User Guide
nav_order: 6
---

# Troubleshooting Guide

This document lists common issues when running **GIFTbench** inside Docker and how to fix them.

---

## Only `http://0.0.0.0:7860` is shown, no public access

### Cause
Gradio by default binds to `127.0.0.1`.  
Also, the host firewall or cloud security group may block incoming connections on port **7860**.

### Fix
1. **Run the container with port mapping:**
   ```bash
   docker run --gpus all -p 7860:7860 maryam483/giftbench:v1.2.0
2. **Check firewall status (Linux with UFW):**
     ```bash
     sudo ufw status
     ```
   -Allow the port if blocked:

   ```bash
     sudo ufw allow 7860/tcp
     sudo ufw reload
    ```
4. **Verify port is listening:**
    ```bash
     sudo lsof -i:7860
         # or
     netstat -tulnp | grep 7860
    ```
## Port opens locally but not from another machine
 
-Ensure container binds to 0.0.0.0 (GIFTbench already sets this).
-Verify firewall/security group allows inbound connections on port 7860.
-If you cannot open the port, use an SSH tunnel:
- Use SSH tunnel:

```bash

ssh -L 7860:127.0.0.1:7860 user@server
```
Then access in your local browser:
http://localhost:7860

### Public Gradio link not working
- Public links (`*.gradio.live`) require `share=True` in `GIFTbench.py`.  
- Links are **temporary** (expire after 72h or when container stops).  
- For a persistent public URL, use:
  - [ngrok](https://ngrok.com/)  
  - [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/)  

---

### Out of Memory (OOM) errors
**Cause:** Stable Diffusion and BigGAN require high VRAM.  
**Fix:**
- Reduce **Generations**, **Population Size**, or **Images to Sample** in the Gradio UI.  
- Use a GPU with **≥ 15GB VRAM**.  
- Stop other GPU processes before running:
  ```bash
  nvidia-smi
  ```
### Downloaded weights not found

Cause: Weights not placed correctly or filenames changed.

