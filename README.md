# SINVAD_Latent_Based_Testing
  <p align="justify">The objective of this research is to evaluate the accuracy of various classifiers on datasets such as Mnist, Svhn, and cifar10. For this purpose, we utilized different generative models to introduce variations in images. Subsequently, we assessed the classifiers' ability to provide accurate predictions based on these altered images.</p>
   <p align="justify">In this study, we employed two unique generative models: the Generative Adversarial Network (GAN) and diffusion models. Each model uses a distinct approach in their respective functionalities. The GAN model operates on a game between a Generator and a Discriminator, while diffusion models add Gaussian noise into data and then recover it via denoising processes. Although the functionalities differ, the outcomes from both generative models display a notable diversity compared to their initial inputs. Our objective involves taking varied inputs to assess the accuracy of classifiers, utilizing SINVAD—a test case generation technique that navigates through a plausible image space while ensuring adherence to plausibility via a Variational Auto-encoder (VAE). In our test scenario, we produce images using test generators for both conditional and unconditional cases, aiming to scrutinize classifier behavior under these circumstances. Subsequently, these images are input into the SINVAD algorithm to evaluate predictions. We have engineered a latent-based SINVAD, evolving from its old version that operated on images, to instead arbitrarily select a latent vector from the space. Moreover, our fitness function scrutinizes the internal computations of the latent variables, instead of assigning a higher vector value through comparison to the original.</p>
##Experimental Detail: 
  Below pictorial representation below depicts the experimental Setup. All models are implemented on three types of datasets. 
  
  ![image](https://github.com/Maryammaryam877/SINVAD_Latent_Based_Testing/issues/1#issue-1936279035)
  
##Datasets:
  MNIST, SVHN, CIFAR10
  
**Table 1**: Models and its checkpoints 
|    TYPE            |         Models                     | Checkpoints/Safetensors |
|--------------------|------------------------------------|-------------------------|
| Generative Models  | Traditional GAN  (Unconditional)   |    [Link](#)            |
|                    | Conditional GAN  (Condition)       |    [Link](#)            |
|                    | Variational Autoencoder (VAE)      |    [Link](#)            |
| Diffusion Models   | DDPM       (Unconditional)         |    [Link](#)            |
|                    | Lora       (Conditional)           |    [Link](#)            |


