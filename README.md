<p align="center">

  <!-- Docs (orange) -->
  <a href="https://deeptestai.github.io/genai_tigs/">
    <img src="https://img.shields.io/badge/Docs-Click%20Here-orange.svg" alt="Documentation">
  </a>

  <!-- DockerHub (Docker blue) -->
  <a href="https://hub.docker.com/r/maryam483/giftbench">
    <img src="https://img.shields.io/badge/DockerHub-maryam483%2Fgiftbench-2496ED.svg?logo=docker&logoColor=white" alt="DockerHub">
  </a>

  <!-- Zenodo DOI (official tab) -->
  <a href="https://doi.org/10.5281/zenodo.16966422">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.16966422.svg" alt="DOI: 10.5281/zenodo.16966422">
  </a>

  <!-- License (green) -->
  <a href="https://github.com/deeptestai/genai_tigs/blob/tool/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  </a>

  <!-- GitHub Stars (social style) -->
  <a href="https://github.com/deeptestai/genai_tigs/stargazers">
    <img src="https://img.shields.io/github/stars/deeptestai/genai_tigs?style=social" alt="GitHub stars">
  </a>

</p>


# GIFTbench: Generative Image Fuzz Testing Benchmark

GIFTbench is a modular framework for testing Deep Learning image classifiers that combines Generative AI with genetic algorithms. Its architecture integrates pretrained generative models with a user-friendly Gradio interface, enabling automated, reproducible, and interpretable robustness testing.

This tool provides a Gradio-based interface with three generative models — **VAE**, **GAN**, and **Diffusion** — on four benchmark datasets — **Mnist**, **SVHN**,**Cifar10**, and **Imagenet** — using GPU-accelerated PyTorch.

> It integrates **Test Input Generators (TIGs)** to evaluate classifier robustness through synthetic image generation.

> GPU-accelerated docker | Auto-downloads pretrained weights | One-command Gradio launch

--------
# Documentation

**For Documentation Click Here** <p align="center">
  <a href="https://deeptestai.github.io/genai_tigs/">
    <img src="https://img.shields.io/badge/📖%20Documentation-Click%20Here-blue?style=for-the-badge&logo=readthedocs&logoColor=white"
         alt="Documentation"
         width="800"/>
  </a>
</p>

# Repository Structure:

- `GIFTbench.py`: Main Gradio app script
- `entrypoint.sh`: Downloads all models (`vae`, `sa`, `sd`, `cdcgan`) from Google Drive and flattens folders
- `Dockerfile`: GPU-ready Docker image with CUDA + PyTorch (PyTorch 2.2 + CUDA 12.1)
- `run.sh`: One-command runner that shows the Gradio public link

---

# Prerequisites

- Docker (GPU-enabled)
- NVIDIA driver + CUDA installed (host machine)
- Internet access (for downloading models from Google Drive)

---
# Pretrained Classifier Checkpoints & Script File :
 To evaluate the classifier's performance under the test generator, you can obtain the pre-trained weight checkpoints from the provided link.
 
  -Mnist_classifier_ckpt: [Download ckpt here](https://drive.google.com/file/d/1IzkDC9Ql3B1XB9vLuFfXjttkZyoOiHg3/view?usp=sharing)
  
  -SVHN_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1vLS_9TT4ncrAfP3LVAOQzw-zdKUgoPBb/view?usp=sharing)
  
  -Cifar10_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1sxG5En1Vc1pEFhedebO8fRcvbb1NNE_y/view?usp=sharing)
  
  -Imagenet_classifir_ckpt:For Imagenet, we used pretrained classifier weights vgg-19-bn  directly from the PyTorch repository [see pytorch Link](https://drive.google.com/uc?export=download&id=YOUR_DIRECT_DOWNLOAD_LINK_ID)
### Training the Classifier from Scratch or Modifying Hyperparameters
  Navigate to the `sa/` directory, which contains subfolders for each dataset. Each dataset-specific folder includes its corresponding training script (`*_train.py`) along with a `model.py` file that defines the classifier architecture.
 
  -To make changes to hyperparameters or to train the classifier from scratch, execute the following command:
  
    python3 train_mnist.py    (for mnist, similar for other datasets)
---
# How to Run?
 GIFTbench can be run in two ways:
## Option A: Run from Docker Hub (Recommended)
Use this method if you want to **quickly run the tool without installing or configuring anything**.  
It is the fastest way for reviewers and users to reproduce our results.

```bash
docker pull maryam483/giftbench:v1.2.0
docker run --name giftbench-running --gpus all -p 7860:7860 maryam483/giftbench:v1.2.0
```
## Option B: Build from Source (Developers)
Use this if you want to see or modify the code, or rebuild the Docker image locally.
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
You will see output like this in both options:

-Running on http://0.0.0.0:7860/   (for local RUN: use URL:http://localhostip:7860/)

-Running on public URL: https://abcdef12345.gradio.live (you can access this link remotely)

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

##  Reference

```bibtex
 @software{maryam_giftbench_2025,
  author       = {Maryam and Matteo Biagiola and Andrea Stocco and Vincenzo Riccio},
  title        = {GIFTbench: Generative Image Fuzz Testing Benchmark},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16966422},
  url          = {https://doi.org/10.5281/zenodo.16966422}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16966422.svg)](https://doi.org/10.5281/zenodo.16966422)



