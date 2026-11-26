# Benchmarking Generative AI Models for Deep Learning Test Input Generation
  <p align="justify">This repository contains the source code and test generation input data of the paper “Benchmarking Generative AI Models for Deep Learning Test Input Generation” accepted at ICST 2025.</p> 

## Motivation:

 <p align="justify">The objective of this research is to benchmark and combine different GenAI models with test input generators, assessing their effectiveness, efficiency, and quality of the generated test images, in terms of domain validity and label preservation</p>

## Repository Structure:
Every folder follows the same order: **dataset → model → mutation level**. Datasets are `mnist`, `svhn`, `cifar10`, `imagenet_pizza`, `imagenet_teddy` (the two ImageNet classes of the paper); models are `vae`, `gan`, `sd`; levels are `high`, `low`.

```
tig/                    unified test input generator (python -m tig), see below
tests/                  pytest suite + golden files
legacy/<dataset>/       original scripts used for the paper
    classifier/         model.py, train.py, weights/
    vae/                tig_vae.py, model.py, train.py, weights/, samples/
    gan/                tig_gan.py (ImageNet: tig_biggan.py), train_cdcgan.py, weights/, samples/
    sd/                 tig_sd.py, generate_images.py
generated_images/<dataset>/<model>_<level>/     generated test inputs (.npy) + summary sheet
evaluation_results/
    rq1/<dataset>/      <model>_rq1.py scripts and <model>/ data
    rq2_rq3/            <dataset>_<model>_<level>.csv
    human_study/        MTurk results <dataset>.csv, survey/ pages and URL generator
    mutation_extent/    csv/ ranges and scripts/ that compute them
    npy_to_png/         converters from .npy to images
documentation/          Stable Diffusion fine-tuning guide and parameters
requirements.txt        dependencies of tig/ and the tests
requirements-sd.txt     old environment of the legacy Stable Diffusion scripts
```

Not in git: `weights/` (checkpoints from Google Drive), `data/` (test sets, downloaded or copied there) and `results/` (output of the runs).

## Installation

Python 3.9 or newer. Tested on Windows with an NVIDIA GPU (GTX 1650, CUDA 12.6), on macOS (CPU) and on Ubuntu in GitHub Actions (CPU).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# NVIDIA GPU only: install the CUDA build of PyTorch first, e.g.
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Without a CUDA GPU everything runs on the CPU (`--device` defaults to `cuda` when available, otherwise `cpu`).

## Weights

MNIST weights are in the repo. Download the others from Google Drive and put them at these paths (they are listed in `tig/config.py`):

| Network | File | Link |
|---|---|---|
| MNIST classifier, VAE, GAN | already in `legacy/mnist/*/weights/` | — |
| SVHN classifier | `weights/svhn/SVHN_vggnet.pth` | [Drive](https://drive.google.com/file/d/1vLS_9TT4ncrAfP3LVAOQzw-zdKUgoPBb/view?usp=sharing) |
| SVHN VAE | `weights/svhn/svhn_convend.pth` | [Drive](https://drive.google.com/file/d/13D8DXRQ41pNv29jZDuWKjjUXMaXlpeG1/view?usp=sharing) |
| CIFAR-10 classifier | `weights/cifar10/CIFAR10_cifar10_train.pynet.pth` | [Drive](https://drive.google.com/file/d/1sxG5En1Vc1pEFhedebO8fRcvbb1NNE_y/view?usp=sharing) |
| CIFAR-10 VAE | `weights/cifar10/cifar10_convend.pth` | [Drive](https://drive.google.com/file/d/1dLYUewBnDfOh6qsy8REWFbb57pktKg6k/view?usp=sharing) |
| SVHN and CIFAR-10 GAN | already in `legacy/<dataset>/gan/weights/` | — |
| ImageNet VAE | `weights/imagenet/imagenet_vae_model_epoch_41.pth` | [Drive](https://drive.google.com/file/d/1iM9Sp7l7zc5o_B5ZukQ4RP8fmkScdFBw/view?usp=sharing) |
| ImageNet classifier (VGG19-BN) and BigGAN-deep-256 | downloaded automatically on first use (~800 MB) | — |
| Stable Diffusion LoRA (5 files) | `weights/sd/*.safetensors` | [Drive](https://drive.google.com/file/d/1FauJR7XbPt_g0W4r-LPIbv7si79JHh4V/view?usp=sharing) |
| Stable Diffusion 1.5 base model | downloaded automatically from Hugging Face on first use (~2 GB) | — |

If a file is missing, the CLI stops and prints the path it expects.

## Usage

The `tig` package runs every generator through one genetic algorithm:

```bash
python -m tig --model vae --dataset mnist --mutation high --seed 0 --num-images 100
```

- `--model vae|gan|sd`, `--dataset mnist|svhn|cifar10|imagenet_pizza|imagenet_teddy`: all 15 combinations work (conditional DCGAN on MNIST/SVHN/CIFAR-10, BigGAN-deep-256 on ImageNet, Stable Diffusion 1.5 with the paper's LoRA on every dataset).
- `--mutation low|high`: the initial perturbation is `0.0002` (low) or `0.002` (high) times the latent range `max - min`, the rule behind every value in the legacy scripts. `--perturbation X` sets the value directly.
- `--seed N`: the same seed gives the same results, on the GPU too (the CLI turns on PyTorch's deterministic mode).
- `--num-images`, `--generations`, `--pop-size`, `--best-left`: number of seeds and GA settings (defaults: 100, 250, 25, 10).
- `--batch-size N`: decodes and classifies the population N images at a time to fit small GPUs; it doesn't change the search. BigGAN uses 5 by default (25 images at once need about 4.8 GB).
- Output: `results/<model>_<dataset>_<mutation>_s<seed>/summary.csv` (one row per seed: `misbehaviour`, `not_found` or `skipped` when the classifier already gets the starting image wrong) plus the original and generated image of each seed as `.npy`.

Where the seeds come from:

- VAE: real test images. MNIST, SVHN and CIFAR-10 download into `data/` on first use. For ImageNet, put images of the class in `data/imagenet_pizza/<any_folder>/*.jpg` (or `imagenet_teddy`), e.g. the ImageNet validation images of that class.
- GAN and Stable Diffusion: random latents with a random class (ImageNet: the dataset's class).

### Stable Diffusion

```bash
python -m tig --model sd --dataset mnist --mutation low --seed 0
```

- It runs in fp16 on the GPU as in the paper. GTX 16xx cards (e.g. GTX 1650) return NaN images in fp16: the run stops and asks for `--sd-precision fp32`, which moves the model to the GPU piece by piece and takes about 4 minutes per image on a GTX 1650. On the CPU it always runs in fp32, much more slowly.
- Each image takes 15-25 DDIM steps, so a full 250-generation run takes hours per seed; use `--generations`, `--pop-size` and `--num-images` for short runs.
- Unlike the legacy SVHN/CIFAR-10/ImageNet scripts it doesn't use T-GATE (which skips cross-attention after step 10 to save time), so images can differ slightly.
- The LoRAs were fine-tuned with kohya-ss: see the [fine-tuning guide](documentation/fine_tune_stable_diffusion.md).

## Tests

```bash
python -m pytest tests
```

63 tests: unit tests (fitness, seeds, GA, CLI), equivalence with the legacy networks, golden regression runs with the published weights, and Stable Diffusion checks with a fake pipeline (no download). Tests that need missing weights are skipped. GitHub Actions runs them on every push and pull request (`.github/workflows/tests.yml`, CPU only, without the Drive weights).

## Training and legacy scripts

The networks can be retrained with the original scripts:

```bash
python legacy/mnist/classifier/train.py      # classifier; same path for svhn and cifar10
python legacy/train_vae.py --dataset mnist   # VAE; also svhn, cifar10, imagenet
python legacy/mnist/gan/train_cdcgan.py      # conditional GAN; same path for svhn and cifar10
```

The original generators are kept in `legacy/<dataset>/<model>/tig_*.py`. They are replaced by `python -m tig`, which reproduces them. The legacy Stable Diffusion scripts need the environment in `requirements-sd.txt` (Python 3.10, e.g. `conda create --name stable_diffusion_env python=3.10`). Note that `legacy/svhn/sd/tig_sd.py` doesn't compile as published (a comment without `#` on its `init_perturbation` line).

### REFERENCE:
Maryam, M., Biagiola, M., Stocco, A., & Riccio, V. (2024). Benchmarking Generative AI Models for Deep Learning Test Input Generation. In Proceedings of the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)
