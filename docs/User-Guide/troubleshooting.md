---
title: Troubleshooting
parent: User Guide
nav_order: 6
---

# Troubleshooting Guide

This document lists common issues when running **GIFTbench** inside Docker and how to fix them.

---

## 1. Only `http://0.0.0.0:7860` is shown, no public access

### Cause
Gradio by default binds to `127.0.0.1`.  
Also, the host firewall or cloud security group may block incoming connections on port **7860**.

### Fix
1. **Run the container with port mapping:**
   ```bash
   docker run --gpus all -p 7860:7860 maryam483/giftbench:v1.2.0


## Port opens locally but not from another machine
- Use SSH tunnel:
```bash
ssh -L 7860:127.0.0.1:7860 user@server
```
