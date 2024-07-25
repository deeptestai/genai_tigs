# An Empirical Comparison of Test Input Generators for Image Classifiers
  <p align="justify">This repository contains the source code and test generation input data of the paper “ An Empirical Comparison of Test Input Generators for Image Classifiers” by ……….</p> 

##Abstract:
 <p align="justify"> This research aims to evaluate the accuracy of various classifiers on datasets such as Mnist, Svhn, cifar10, and an Imagenet dataset. For this purpose, we utilized different generative models to introduce misbehavior-inducing images. Subsequently, we assessed the classifiers' ability to provide accurate predictions based on these images.</p>

 ## Configurations:
  <p align="justify">\textbf{VAE:} VAE for all four types of datasets requires running the train.py file to train from scratch or downloading all VAE [checkpoints](URL) and then running the sinvad_vae_dataset.py file to collect misbehavior-inducing inputs.</p>

 <p align="justify">\textb{GAN:} Similarly, each dataset-name_experiments folder contains the gan folder as a subfolder. This folder contains one cdcgan folder and one sinvad_gai_dataset-name.py file. One can download the checkpoints from the weight directory under the cdcgan folder. After downloading it, run the script sinvad_gan_dataset-name.py.</p>

 Run the script by using a command.
 To train the VAE from scratch, run the following command:

```bash
python train_master.py --dataset mnist

Replace mnist with svhn, cifar10, or imagenet to train on a different dataset.

After downloading the checkpoints, run the following command to collect misbehavior-inducing inputs, run the command:
```bash
python sinvad_vae_mnist.py --checkpoint_path /path/to/checkpoint   

Replace/path/to/checkpoint with the path to your file.
  
### Repository Structure:
- experiments: contain subfolders of four separate datasets, each with three different generative AI model-based folders. These folders consist of the raw experimental data and the script used to obtain the results reported in the paper.


## Experimental Detail:
  A pictorial representation below depicts the experimental Setup. All models are implemented on three types of datasets. 
  
  ![image](https://private-user-images.githubusercontent.com/129972846/274067116-e341c06d-c914-4038-ac93-d6002f4758ed.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY5NzQwMzAsIm5iZiI6MTY5Njk3MzczMCwicGF0aCI6Ii8xMjk5NzI4NDYvMjc0MDY3MTE2LWUzNDFjMDZkLWM5MTQtNDAzOC1hYzkzLWQ2MDAyZjQ3NThlZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAxMFQyMTM1MzBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hYTFmNzgzMzFjM2Y0MGQ4ZGNhYzllNGE2YzE1MGE2Njg1OGY2ZDE2ZDVkODc3YzljNjgzZjZjZjlkZjFhMDY4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.4sXJwUjLid5wQTMCNjtX_Ojno5nI9f-ynJc785zu2cM)
  
### Datasets:
   MNIST, SVHN, CIFAR10, Imagenet1k
  
**Table 1**: Models and its checkpoints 

|       TYPE         |           Models                | datasets         | Checkpoints/Safetensors   |
|--------------------|---------------------------------|------------------|---------------------------|
|                    |                                 | MNIST_dcgan      |    Link [1]               |
|                    |                                 | SVHN_dcgan       |    Link[2]                |
| Generative Models  | Traditional GAN (Unconditional) | CIFAR10_dcgan    |    Link[3]                |
|                    |                                 | Imagenet1K_dcgan |    Link[4]                |
|                    |                                 |                  |                           |
|                    |                                 | MNIST_cgan       |     Link[5]               |
| Generative Models  | Conditional GAN  (Condition)    | SVHN_cgan        |     Link[6]               |
|                    |                                 | CIFAR10_cgan     |     Link[7]               |
|                    |                                 | Imagenet1K_cgan  |     Link[8]               |
|                    |  Variational Autoencoder (VAE)  | Four datasets    |[VAE models single Link][9]|
|                    |                                 |                  |                           |
|                    |                                 |                  |                           |
|                    |                                 |MNIST_ddpm        |                           |
|Diffusers_Generative| DDPM       (Unconditional)      |SVHN_ddpm         |   Hugging face Account    |
|    Models          |                                 |CIFAR10_ddpm      |   Complete Model script   |
|                    |                                 |Imagenet1K_ddpm   |   alongwith checkpoints   |
|                    |                                 |                  |                           |
|                    |                                 |                  |                           |
|Diffusers_Generative| SD_Lora_finetune(Conditional)   |MNIST_lora        |    Hugging face Account   |
|    Models          |                                 |SVHN_lora         |  Complete Model script    |
|                    |                                 |CIFAR10_lora      |  alongwith checkpoints    |
|                    |                                 |Imagenet 1k       |                           |

### Download classifier ckpt for all three types of datasets:
 <p align="justify">
    This link provides a download for a singular zip file that contains three distinct sub-folders: mnist_classifier, cifar10_classifier, and svhn_classifier. Each of these folders includes checkpoint files for their respective datasets 
    <a href="https://drive.google.com/uc?export=download&id=1JS8xBjJZoOjHme0RLITsk8AYjNeKf2Ns">View and Download .ckpt Files</a>. 
    Moreover, I prefer to use a pre-trained ckpt of imagenet classifier 
    <a href="https://drive.google.com/uc?export=download&id=YOUR_DIRECT_DOWNLOAD_LINK_ID">see link here</a>.
</p>

### REFERENCE:


[9].https://drive.google.com/uc?export=download&id=1MNyzsNZWE5QzoX_xIbIlm-IV6lNri3W3



