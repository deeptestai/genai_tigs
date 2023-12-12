# SINVAD_Latent_Based_Testing
  <p align="justify">The objective of this research is to evaluate the accuracy of various classifiers on datasets such as Mnist, Svhn, and cifar10. For this purpose, we utilized different generative models to introduce variations in images. Subsequently, we assessed the classifiers' ability to provide accurate predictions based on these altered images.</p>
  <p align="justify">In this study, we employed two unique generative models: the Generative Adversarial Network (GAN) and diffusion models. Each model uses a distinct approach in their respective functionalities. The GAN model operates on a game between a Generator and a Discriminator, while diffusion models add Gaussian noise into data and then recover it via denoising processes. Although the functionalities differ, the outcomes from both generative models display a notable diversity compared to their initial inputs.</p> 
  <p align="justify"> Our objective involves taking varied inputs to assess the accuracy of classifiers, utilizing SINVAD—a test case generation technique that navigates through a plausible image space while ensuring adherence to plausibility via a Variational Auto-encoder (VAE). In our test scenario, we produce images using test generators for both conditional and unconditional cases, aiming to scrutinize classifier behavior under these circumstances. Subsequently, these images are input into the SINVAD algorithm to evaluate predictions. We have engineered a latent-based SINVAD, evolving from its old version that operated on images, to instead arbitrarily select a latent vector from the space. Moreover, our fitness function scrutinizes the internal computations of the latent variables, instead of assigning a higher vector value through comparison to the original.</p>
   
## Experimental Detail:
  A pictorial representation below depicts the experimental Setup. All models are implemented on three types of datasets. 
  
  ![image](https://private-user-images.githubusercontent.com/129972846/274067116-e341c06d-c914-4038-ac93-d6002f4758ed.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY5NzQwMzAsIm5iZiI6MTY5Njk3MzczMCwicGF0aCI6Ii8xMjk5NzI4NDYvMjc0MDY3MTE2LWUzNDFjMDZkLWM5MTQtNDAzOC1hYzkzLWQ2MDAyZjQ3NThlZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAxMFQyMTM1MzBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hYTFmNzgzMzFjM2Y0MGQ4ZGNhYzllNGE2YzE1MGE2Njg1OGY2ZDE2ZDVkODc3YzljNjgzZjZjZjlkZjFhMDY4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.4sXJwUjLid5wQTMCNjtX_Ojno5nI9f-ynJc785zu2cM)
  
### Datasets:
   MNIST, SVHN, CIFAR10, Imagenet
  
**Table 1**: Models and its checkpoints 
|    TYPE            |         Models                     | Checkpoints/Safetensors |
|--------------------|------------------------------------|-------------------------|
| Generative Models  | Traditional GAN  (Unconditional)   |    [Link](#)            |
|                    | Conditional GAN  (Condition)       |    [Link](#)            |
|                    | Variational Autoencoder (VAE)      |    [Link](#)            |
| Diffusion Models   | DDPM       (Unconditional)         |    [Link](#)            |
|                    | Lora       (Conditional)           |    [Link](#)            |

### Download vae ckpt for all four types of datasets:
<p align="justify">This link provides a download for a singular zip file that contains four distinct sub-folders: mnist_vae, cifar10_vae, svhn_vae, and imagenet_vae. Each of these folders includes checkpoint files for their respective datasets [View and Download .ckpt Files](https://drive.google.com/uc?export=download&id=1MNyzsNZWE5QzoX_xIbIlm-IV6lNri3W3).</p>
### Download classifier ckpt for all three types of datasets:
<p align="justify">This link provides a download for a singular zip file that contains three distinct sub-folders: mnist_classifier, cifar10_classifier, and svhn_classifier. Each of these folders includes checkpoint files for their respective datasets [View and Download .ckpt Files](https://drive.google.com/uc?export=download&id=1JS8xBjJZoOjHme0RLITsk8AYjNeKf2Ns). Moreover, I prefer to use a pre-trained ckpt of imagenet classifier [see link here](https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/).</p>


