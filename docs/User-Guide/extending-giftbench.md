---
title: Extending GIFTbench
parent: User Guide
nav_order: 7
---
# Extending GIFTbench

GIFTbench is designed to be extensible at the **module level**, allowing users to add new datasets or integrate custom generative models without modifying the core search, evaluation, or visualization pipeline. This design ensures that extensions preserve correctness, reproducibility, and comparability across experiments.

This document describes how users can extend GIFTbench by (i) adding a new dataset and (ii) training and integrating dataset-specific generative models.

---

## Adding a New Dataset

Each dataset in GIFTbench is implemented as a **self-contained module** (e.g., `mnist/`, `svhn/`, `cifar10/`, `imagenet/`). To add support for a new dataset, users should create a new dataset directory following the structure of existing modules.

At a minimum, the dataset module must define:
- data loading logic,
- dataset-specific preprocessing,
- label handling and validation,
- a test-generation entry point compatible with the GIFTbench search pipeline.

Once implemented, the new dataset module can directly reuse the existing genetic search, evaluation, and reporting logic without modifying any core components. The dataset will then appear as an additional option in the Gradio interface.

> **Note:** Dataset integration is intentionally handled at the code level rather than through the UI to ensure consistent preprocessing and reproducible evaluations.

---

## Training Generative Models for a New Dataset

Extending GIFTbench to a new dataset requires training **dataset-specific generative models**. GIFTbench does not provide dataset-agnostic generators by design, as such models would compromise the validity of robustness assessments.

Depending on the generator type:
- **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)** must be trained from scratch on the target dataset.
- **Diffusion models** must be fine-tuned to the dataset distribution (e.g., using LoRA-based fine-tuning).

 Classifier training and fine-tuning scripts, along with example configurations, are provided in the repository and mentioned below:
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
# Stable Diffusion Setup and Script Execution
#### How to Fine-tune Stable Diffusion? 
Fine-tune stable diffusion using the khoya-ss platform on four different datasets. For a detailed description, please [click here](https://github.com/Maryammaryam877/genai_tigs/blob/main/documentation/fine-tune%20stable%20diffusion.md).
#### Download SD weights
Download the fine-tuned model weights from [this link](https://drive.google.com/file/d/1FauJR7XbPt_g0W4r-LPIbv7si79JHh4V/view?usp=sharing). 

---

## Integrating Custom Generative Models

After training, generative models can be integrated into GIFTbench in one of two ways:

1. **Model Replacement**  
   Users may replace the underlying checkpoint or configuration used by an existing VAE, GAN, or diffusion generator while reusing the same generator logic and search-based testing pipeline.

2. **Generator Wrapper**  
   If a generative model uses different input–output conventions or generation behavior, users may implement a lightweight wrapper that exposes the standard interface expected by GIFTbench.

In both cases, integration occurs at the module level and does not require changes to the core genetic search, evaluation, or validation components.

---

## Model-Specific Configuration Options

GIFTbench exposes configuration options selectively through the interface to prevent invalid settings. For example:
- prompt selection is enabled only for diffusion-based generators,
- truncation control is available only for BigGAN-based models.

This design ensures that users are presented exclusively with valid configuration choices for the selected dataset and generative model.

---

## Verifying an Extension

After adding a dataset or integrating a custom generative model:
- the dataset appears as a new tab in the Gradio interface,
- compatible generative models become selectable,
- existing search and evaluation controls remain unchanged.

This confirms that the extension is correctly integrated into the GIFTbench pipeline.

---

For guidance on running experiments and managing trained models, refer to the remaining documentation pages in the repository.
