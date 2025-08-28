---
title: "Welcome to GIFTbench"
nav_order: 1
permalink: /
---

# Welcome to GIFTbench 
<p align="center">
  <img src="images/gradio.png" alt="Gradio UI Demo" width="500"/>
</p>

## Why Gradio?

The GIFTbench framework integrates a Gradio-powered graphical interface to bridge the gap between complex test input generation and user accessibility. Gradio serves as the interactive front-end, enabling:

- Rapid prototyping of test scenarios using VAE, GAN, and Diffusion models
- Real-time configuration of perturbation parameters and model settings
- Instant visualization of original and mutated test inputs
- Immediate display of classifier predictions and misclassification cases

By lowering the technical barrier, Gradio ensures that robustness testing is accessible to non-experts, supports reproducibility, and facilitates hands-on exploration of generative test pipelines.

---

##  What is GIFTbench?

GIFTbench stands for **Generative Image Fuzz Testing Benchmark**. **GIFTbench** is a modular framework for testing deep learning image classifiers using **Generative AI** and **Genetic Algorithms**. It automates robustness testing by generating and perturbing test inputs in the latent space of generative models such as **VAE**, **GAN**, and **Diffusion Models**.
It combines:

- Pre-trained generative models (VAE, GAN, Diffusion)
- Genetic algorithms to evolve latent vectors
- Classifiers under test
- Automated, reproducible evaluation
-  radio interface for user-friendly interaction

This setup helps researchers and practitioners identify vulnerabilities in classifiers and evaluate robustness with minimal manual coding.

---

## 🚀 Features

-  Supports VAE, GAN, and Diffusion models
-  Visual inspection of original vs perturbed images
-  Metric tracking status: Misclassified seeds,%Misclassification,Avg iterations
-  Built-in mutation strategies and validation hooks
-  Dockerized for easy reproducibility

---
