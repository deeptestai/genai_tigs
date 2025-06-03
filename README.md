# GIFTbench Gradio App (GPU-enabled)

This tool provides a Gradio-based interface with three generative models — **VAE**, **CDCGAN**, and **Diffusion** — on four benchmark datasets — **Mnist**, **SVHN**,**Cifar10**, and **Imagenet** — using GPU-accelerated PyTorch.

> It integrates **Test Input Generators (TIGs)** to evaluate classifier robustness through synthetic image generation.

> GPU-accelerated docker | Auto-downloads pretrained weights | One-command Gradio launch

---

##  Repository Structure:

- `GIFTbench.py`: Main Gradio app script
- `entrypoint.sh`: Downloads all models (`vae`, `sa`, `sd`, `cdcgan`) from Google Drive and flattens folders
- `Dockerfile`: GPU-ready Docker image with CUDA + PyTorch (PyTorch 2.2 + CUDA 12.1)
- `run.sh`: One-command runner that shows the Gradio public link

---

##  Prerequisites

- Docker (GPU-enabled)
- NVIDIA driver + CUDA installed (host machine)
- Internet access (for downloading models from Google Drive)

---

## How to Run?

###  1. Clone this repository and switch to `gradio-tool` branch:

```bash
git clone https://github.com/deeptestai/genai_tigs.git
cd genai_tigs
git checkout gradio-tool
```
------
###  2. Build Docker image and launch the Gradio App with One Command:

```bash
  ./run.sh
```

This script will:

-Run Docker in detached mode

-Auto-download all model folders (vae, sa, sd, cdcgan)

-Display the public or local Gradio link via docker logs -f giftbench-running

After running:

```bash
  ./run.sh
```
You will see output like this:

-Running on http://0.0.0.0:7860/   (for local RUN: use URL:http://localhost:7860/)

-Running on public URL: https://abcdef12345.gradio.live

###  3. How to View and Stop Watching Log:

-Press Ctrl + C to stop watching the logs — the app will keep running in the background.

If you want to check the public Gradio link again after closing the logs:

```bash
   docker logs -f giftbench-running
```
This will display the same output (without restarting anything).

###  4. How to Stop APP?

When you're ready to shut down the app completely, run this command:

```bash
   docker stop giftbench-running && docker rm giftbench-running
```
This will:

-Stop the running container

-Clean it up from the system

##  Notes
> The Dockerfile supports **any GPU-compatible Linux machine** with installed NVIDIA drivers and CUDA support.  
> **Model folders are not tracked in Git** — they are dynamically downloaded during runtime by `entrypoint.sh` using Google Drive links.  
> **No manual setup** is required. Just clone, build, and run — models are downloaded automatically.

##  License

This project is licensed under the [MIT License](LICENSE).



