---
title: Extending GIFTbench
parent: User Guide
nav_order: 7
---
# Extending GIFTbench

GIFTbench is designed to be extensible at the **module level**, allowing users to add new datasets or integrate custom generative models without modifying the core search, evaluation, or visualization pipeline. This design ensures that extensions preserve correctness, reproducibility, and comparability across experiments.

This document describes how users can extend GIFTbench by (i) adding a new dataset and (ii) training and integrating dataset-specific generative models.

---

## 1. Adding a New Dataset

Each dataset in GIFTbench is implemented as a **self-contained module** (e.g., `mnist/`, `svhn/`, `cifar10/`, `imagenet/`). To add support for a new dataset, users should create a new dataset directory following the structure of existing modules.

At a minimum, the dataset module must define:
- data loading logic,
- dataset-specific preprocessing,
- label handling and validation,
- a test-generation entry point compatible with the GIFTbench search pipeline.

Once implemented, the new dataset module can directly reuse the existing genetic search, evaluation, and reporting logic without modifying any core components. The dataset will then appear as an additional option in the Gradio interface.

> **Note:** Dataset integration is intentionally handled at the code level rather than through the UI to ensure consistent preprocessing and reproducible evaluations.

---

## 2. Training Generative Models for a New Dataset

Extending GIFTbench to a new dataset requires training **dataset-specific generative models**. GIFTbench does not provide dataset-agnostic generators by design, as such models would compromise the validity of robustness assessments.

Depending on the generator type:
- **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)** must be trained from scratch on the target dataset.
- **Diffusion models** must be fine-tuned to the dataset distribution (e.g., using LoRA-based fine-tuning).
  
---

## Classifier and GenAI Training Scripts
   Classifier and GenAI training or fine-tuning scripts, along with example configurations, are provided in the repository and detailed below.

### Pretrained Classifier Checkpoints & Script File :

### Training the Classifier from Scratch or Modifying Hyperparameters
To evaluate robustness, GIFTbench requires a reference classifier trained on the same dataset. This classifier serves as the system under test and is used to assess misclassification induced by the test generators. These scripts can be adapted to new datasets by adjusting:
-input resolution and channels,
-number of output classes,
-dataset-specific preprocessing.
  Navigate to the `sa/` directory, which contains subfolders for each dataset. Each dataset-specific folder includes its corresponding training script (`*_train.py`) along with a `model.py` file that defines the classifier architecture.
 
  -To make changes to hyperparameters or to train the classifier from scratch, execute the following command:
  
  ```
   python3 train_mnist.py    (for mnist, similar for other datasets)
  ```
### Pretrained Checkpoints
 
  To evaluate the classifier's performance under the test generator, you can obtain the pre-trained weight checkpoints from the provided link.
 
  -Mnist_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1IzkDC9Ql3B1XB9vLuFfXjttkZyoOiHg3/view?usp=sharing)
  
  -SVHN_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1vLS_9TT4ncrAfP3LVAOQzw-zdKUgoPBb/view?usp=sharing)
  
  -Cifar10_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1sxG5En1Vc1pEFhedebO8fRcvbb1NNE_y/view?usp=sharing)
  
  -Imagenet_classifir_ckpt:For Imagenet, we used pretrained classifier weights vgg-19-bn  directly from the PyTorch repository [see pytorch Link](https://drive.google.com/uc?export=download&id=YOUR_DIRECT_DOWNLOAD_LINK_ID)
  
---

## Generative AI pretrained checkpoints & Script File :
GIFTbench does not rely on dataset-agnostic generators. Instead, each supported dataset uses generative models trained or fine-tuned on that dataset’s distribution.

### 1. VAE:

Navigate to the `vae/` directory, which contains subfolders for each dataset. Each dataset-specific folder includes its corresponding training script
  
 Run the script by using a command.
 
 To train the VAE from scratch, run the following command:

```
python train_master.py --dataset mnist 
```
Replace mnist with svhn, cifar10, or imagenet to train on a different dataset.

### Pretrained Checkpoints
 We have trained the Variational Autoencoder (VAE) on all four datasets: MNIST, SVHN, CIFAR-10, and ImageNet. You can download the pretrained weights for all four models from the following link.

- Mnist_vae_ckpt: already uploaded mnist/mnist-vae/weights under repository structure
- SVHN_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/13D8DXRQ41pNv29jZDuWKjjUXMaXlpeG1/view?usp=sharing)
- Cifar10_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/1dLYUewBnDfOh6qsy8REWFbb57pktKg6k/view?usp=sharing)
- Imagenet_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/1iM9Sp7l7zc5o_B5ZukQ4RP8fmkScdFBw/view?usp=sharing)

### 2. GAN:
 Navigate to the `cdcgan/` directory, which contains subfolders for each dataset. Each dataset-specific folder includes its corresponding training script
 To train the CDCGAN for a specific dataset from scratch, use the following command:

```
python cdcgan_dataset-name.py 
```
Replace dataset-name with mnist, svhn or cifar10 to run the GAN for the other datasets.

### Pretrained Checkpoints

   We have trained Conditional GANs for three datasets: MNIST, SVHN, and CIFAR-10. The pretrained weights for these models are available in their respective dataset directories under the repository structure.

    
- weights are available [Click here](https://drive.google.com/file/d/1MnXSukCHhtajVxtJxWpXCtXE8SFSCNKh/view?usp=sharing)

-For ImageNet, we have chosen pytorch BigGAN as the Conditional GAN model and are utilizing its pretrained weights.A detail about configuration and environment settings [here](https://github.com/lukemelas/pytorch-pretrained-gans/tree/main)

 We utilize the 256x256 size Deep-BigGAN model with the specified pretrained weights by executing the following command:
 
 ```
 G = make_gan(gan_type='biggan', model_name='biggan-deep-256')
```
 We set the truncation value to 1.0 to produce images with greater variation.

### 3. Stable Diffusion Setup and Script Execution

####  How to Fine-tune Stable Diffusion?

Fine-tune stable diffusion using the khoya-ss platform on four different datasets. For a detailed description, please [click here](https://github.com/Maryammaryam877/genai_tigs/blob/main/documentation/fine-tune%20stable%20diffusion.md).

####  Download SD weights
Download the fine-tuned model weights from [this link](https://drive.google.com/file/d/1FauJR7XbPt_g0W4r-LPIbv7si79JHh4V/view?usp=sharing). 

---

## Integrating Custom Generative Models

After training, generative models can be integrated into GIFTbench in one of two ways:

1. **Model Replacement**  
   Users may replace the underlying checkpoint or configuration used by an existing VAE, GAN, or diffusion generator while reusing the same generator logic and search-based testing pipeline.

2. **Generator Wrapper**  
   If a generative model uses different input–output conventions (e.g., different latent formats, generation calls, or output representations) or generation behavior, users may implement a lightweight wrapper that exposes the standard interface expected by GIFTbench.

In both cases, integration occurs at the module level and does not require changes to the core genetic search, evaluation, or validation components.

---
**Generator Wrapper**
### When Do I Need a Wrapper?

In GIFTbench, wrappers are required **only when a new dataset or generative model does not follow the existing input–output conventions**.

### New Dataset
A dataset wrapper is needed when adding a dataset with different:
- image resolution or channels,
- preprocessing steps,
- label format.

The wrapper ensures the dataset exposes the same interface used by existing datasets so it can reuse the same testing pipeline.

### New Generative Model
A generator wrapper is needed only if the model:
- uses a different latent format,
- exposes a non-standard generation API,
- returns outputs in a non-image format.

If the model already supports `generate(latents, **kwargs)`, no wrapper is required.

Wrappers adapt datasets or models to GIFTbench’s standard interface without changing the core search, evaluation, or visualization components.

### Example: Simple Wrapper Class

A wrapper is a small class that adapts a dataset or generative model to the interface expected by GIFTbench.

```python
class SimpleWrapper:
    def __init__(self, model):
        self.model = model

    def generate(self, latents, **kwargs):
        return self.model.sample(latents)
```
This wrapper translates GIFTbench’s latent inputs into the model’s native generation call and returns images in the expected format. Only the wrapper changes; the rest of the framework remains unchanged.

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
