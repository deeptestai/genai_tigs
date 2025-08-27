# Welcome to GIFTbench 

  ![Gradio Web-App for Testing](images/gradio.png)

**GIFTbench** is a modular framework for testing deep learning image classifiers using **Generative AI** and **Genetic Algorithms**. It automates robustness testing by generating and perturbing test inputs in the latent space of generative models such as **VAE**, **GAN**, and **Diffusion Models**.

---

##  What is GIFTbench?

GIFTbench stands for **Generative Image Fuzz Testing Benchmark**. It combines:

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
