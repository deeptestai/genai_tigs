# An Empirical Comparison of Test Input Generators for Image Classifiers
  <p align="justify">This repository contains the source code and test generation input data of the paper “ An Empirical Comparison of Test Input Generators for Image Classifiers” by ……….</p> 

## Motivation:

 <p align="justify">The objective of this research is to evaluate the accuracy of various classifiers on datasets including MNIST, SVHN, CIFAR-10, and ImageNet. To achieve this, we employed different generative models to create images that induce misbehavior. We then assessed the classifiers' performance by determining their accuracy in making predictions based on these challenging images.</p>

## Repository Structure:
- **experiments:** contain subfolders of four separate datasets, each with three different generative AI model-based folders. These folders consist of the raw experimental data and the script used to obtain the results reported in the paper.
-  **documentation:**
- 


 ## Getting Started to RUN GAI-Tigs:
 ### STEP-1: Configure the envronment:

 ### STEP-2: Download Pretrained Classifier Checkpoints:
 To evaluate the performance of the classifier under the test generator, you can obtain the pretrained weight checkpoints from the provided link.
 
  -Mnist_classifier_ckpt: already uploaded mnist/mnist-classifier/weights under repository structure
  
  -SVHN_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1vLS_9TT4ncrAfP3LVAOQzw-zdKUgoPBb/view?usp=sharing)
  
  -Cifar10_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1sxG5En1Vc1pEFhedebO8fRcvbb1NNE_y/view?usp=sharing)
  
  -Imagenet_classifir_ckpt:For Imagenet, we used pretrained classifier weights vgg-19-bn  directly from the PyTorch repository [see pytorch Link](https://drive.google.com/uc?export=download&id=YOUR_DIRECT_DOWNLOAD_LINK_ID)
  
  -To make changes to hyperparameters or to train the classifier from scratch, execute the following command:
  
    python3 train_mnist.py    (for mnist, similar for other datasets)


### STEP-3: Execute Any One of the Three Generative AI Models Based on Test Input Generators
 
 ### 1. VAE:
 
 We have trained the Variational Autoencoder (VAE) on all four datasets: MNIST, SVHN, CIFAR-10, and ImageNet. You can download the pretrained weights for all four models from the following link.

- Mnist_vae_ckpt:already uploaded mnist/mnist-vae/weights under repository structure
- SVHN_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/13D8DXRQ41pNv29jZDuWKjjUXMaXlpeG1/view?usp=sharing)
- Cifar10_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/1dLYUewBnDfOh6qsy8REWFbb57pktKg6k/view?usp=sharing)
- Imagenet_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/1iM9Sp7l7zc5o_B5ZukQ4RP8fmkScdFBw/view?usp=sharing)
  
 Run the script by using a command.
 
 To train the VAE from scratch, run the following command:

```
python train_master.py --dataset mnist 
```

Replace mnist with svhn, cifar10, or imagenet to train on a different dataset.

After downloading the checkpoints, run the following command to collect misbehavior-inducing inputs, run the command:

```
python sinvad_vae_mnist.py --checkpoint_path /path/to/checkpoint
```

Replace/path/to/checkpoint with the path to your file.
### 2. GAN:

We have trained Conditional GANs for three datasets: MNIST, SVHN, and CIFAR-10. The pretrained weights for these models are available in their respective dataset directories under Repository structure. 

-For ImageNet, we have chosen pytorch BigGAN as the Conditional GAN model and are utilizing its pretrained weights.A detail about configuration and environment settings [here](https://github.com/lukemelas/pytorch-pretrained-gans/tree/main)

 We utilize the 256x256 size Deep-BigGAN model with the specified pretrained weights by executing the following command:
 
 ```
 G = make_gan(gan_type='biggan', model_name='biggan-deep-256')
```
 We set the truncation value to 1.0 to produce images with greater variation.
 To run the SINVAD CDCGAN for a specific dataset, use the following command:

```
python gan_master.py --dataset mnist
```
Replace mnist with svhn or cifar10 to run the GAN for the other datasets.

Similarly, after downloading or training the gan model, run the sinvad script for cdcgan.
```
python sinvad_cdcgan_master.py --dataset mnist
```
Replace mnist with other datasets such as svhn, cifar10, imagenet

### 3. Stable Diffusion Setup and Script Execution
#### How to Fine tune Stable Diffusion? 
Fine-tune stable diffusion using the khoya-ss platform on four different datasets. For a detailed description, please [click here](https://github.com/Maryammaryam877/genai_tigs/blob/main/documentation/fine-tune%20stable%20diffusion.md).
#### Download SD weights
Download the fine-tuned model weights from [this link](URL). 

#### configuration to excute Sd_tig

To run the test generator for Stable diffusion. You are required to install the following setup steps:
### Create a Virtual Environment and install packages to run the SD-based generator script

First, please make sure you have Conda installed. If not, you can download and install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

 **1. Create a Conda virtual environment**:
   Open a terminal or Anaconda Prompt and run the following command to create a new Conda virtual environment:
   
   ```
   conda create --name stable_diffusion_env python=3.10
   ```
   
 **2. Installing desired packages**
     The requirements setup is already provided; run requirement-sd.txt using pip.
   
   ```
     pip install requirements-sd.txt
   ```

 **3. Run SD-based Generator**
     To run the script of the SD generator, run the following command:

```bash
python3 sinvad_lora_mnist.py
```     
     
  Similar mnist is replaced with other dataset names, svhn, cifar10, and imagenet, according to their dataset name.
     
 > Note: keep upgrading diffusers and transformers to avoid errors.


### REFERENCE:





