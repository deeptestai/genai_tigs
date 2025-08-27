# Welcome to GIFTbench 

This site contains the full documentation for **GIFTbench**,  
a modular framework for testing deep learning image classifiers using Generative AI and genetic algorithms.

---

##  How to Use This Documentation

Use the **left sidebar** (or the navigation menu) to browse different sections:

- **Getting Started** → [Installation and first steps](getting-started.md)
- **User Guide** → [Usage examples, workflows, and commands](user-guide.md)
- **Reference** → [Configuration, parameters, and advanced options](reference.md)

---

##  What is the GIFTbench Tool?

GIFTbench combines **Generative AI models** (VAE, GAN, Diffusion) with **evolutionary algorithms**  
to automatically generate test inputs that expose weaknesses in deep learning classifiers.  

It is designed for:

- Researchers evaluating **robustness** of classifiers
- Practitioners who need **reproducible testing pipelines**
- Reviewers verifying **model performance under perturbations**

---

##  Key Features

-  **Supports multiple generative models**: VAE, GAN, Diffusion
- **Automated perturbation with genetic algorithms**
- **Reproducible experiments** (via Docker + Gradio UI)
-  **Evaluation metrics**: validity, misclassification, label preservation
-  **Web-based interface** for interactive exploration

---

##  Quick Links

- [Installation Guide](getting-started.md)
- [Running the App](user-guide.md#running-the-gradio-interface)
- [Docker Setup](getting-started.md#docker-installation)
- [Configuration Reference](reference.md)

---

## Citation

If you use **GIFTbench** in your research, please cite:

```bibtex
@inproceedings{yourbibkey2025,
  title={GIFTbench: Generative Image Fuzz Testing Benchmark},
  author={Your Name and Collaborators},
  booktitle={Conference/Journal},
  year={2025}
}

