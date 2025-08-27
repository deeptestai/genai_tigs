---
title: Using the Gradio UI
parent: User Guide
nav_order: 1
---

# Gradio Interface Guide

The GIFTbench Gradio UI provides an interactive way to configure and run Test Input Generators (TIGs). 

The screenshot below shows the **ImageNet (class-pizza)** configuration tab in GIFTbench.

<p align="center">
  <img src="docs/images/gui.png" alt="GIFTbench Gradio Interface Example" width="300"/>
</p>


Below is an explanation of each option and control (screenshot example: **ImageNet – class pizza**).

---

###  Configuration Options

- **Model Type**  
  Choose the generative model to use:
  - **VAE (pizza):** Variational Autoencoder  
  - **GAN (pizza):** BigGAN model  
  - **DM (pizza):** Diffusion model  

- **Generations**  
  Number of evolutionary generations for the genetic algorithm (default: 250).  
  More generations = longer run, but higher chance of finding perturbations.

- **Population Size**  
  Number of candidate latent vectors per generation (default: 25).  
  Larger population = more exploration, but more computation.

- **Best Left (Selection)**  
  Number of top-performing candidates retained between generations (default: 10).  
  Controls selection pressure in the genetic algorithm.

- **Perturbation Size**  
  Mutation size applied to latent vectors:
  - **Low** → subtle changes  
  - **High** → stronger perturbations  

- **Initial Perturbation Size**  
  Mutation size applied in the **first generation** only. Helps diversify the initial population.

- **Images to Sample**  
  Number of test images generated in one run (default: 3).

- **Classifier Under Test**  
  The image classifier used for evaluation:
  - Built-in models (e.g., `VGG19bn`)  
  - Or upload your own TorchScript `.jit` classifier

- **Prompt (only for DM)**  
  Text prompt used in Diffusion models to guide generation (e.g., `"A photo of pizza slice"`).

---

### Buttons

- **Run `<Dataset>` TIG**  
  Starts the generation and evaluation process.  
  Outputs include:
  - Status updates
  - Generated images (original vs perturbed)
  - Downloadable `.zip` of images
  - Table of predictions (expected vs predicted labels, iterations)

- **Stop**  
  Immediately halts the generation process. Useful if runs take too long.

---

### Outputs

- **Status:** Live updates on progress (e.g., loading weights, generation step, classifier predictions).  
- **Results Table:** Displays:
  - Image ID
  - Expected label
  - Predicted label
  - Iterations used
- **Gallery:** Shows original and perturbed images side by side.  
- **Download:** Option to download all generated images as a `.zip` file for offline analysis.

