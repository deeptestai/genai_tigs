# An Empirical Comparison of Test Input Generators for Image Classifiers
  <p align="justify">This repository contains the source code and test generation input data of the paper “ An Empirical Comparison of Test Input Generators for Image Classifiers” by ……….</p> 

## Abstract:

 <p align="justify"> This research aims to evaluate the accuracy of various classifiers on datasets such as Mnist, Svhn, cifar10, and an Imagenet dataset. For this purpose, we utilized different generative models to introduce misbehavior-inducing images. Subsequently, we assessed the classifiers' ability to provide accurate predictions based on these images.</p>

## Repository Structure:
- experiments: contain subfolders of four separate datasets, each with three different generative AI model-based folders. These folders consist of the raw experimental data and the script used to obtain the results reported in the paper.


 ## RUN Script:
 ### VAE:
  <p align="justify"> VAE for all four types of datasets requires running the train.py file to train from scratch or downloading all VAE [checkpoints here](URL), classifier checkpoints, and then running the sinvad_vae_dataset.py file to collect misbehavior-inducing inputs.</p>

 Run the script by using a command.//
 To train the VAE from scratch, run the following command:

```bash
python train_master.py --dataset mnist
```

Replace mnist with svhn, cifar10, or imagenet to train on a different dataset.

After downloading the checkpoints, run the following command to collect misbehavior-inducing inputs, run the command:

```bash
python sinvad_vae_mnist.py --checkpoint_path /path/to/checkpoint
```

Replace/path/to/checkpoint with the path to your file.
### GAN:
 <p align="justify"> Similarly, each dataset-name_experiments folder contains the gan folder as a subfolder. This folder contains one cdcgan folder and one sinvad_gai_dataset-name.py file. One can download the checkpoints from the weight directory under the cdcgan folder. After downloading it, run the script sinvad_gan_dataset-name.py.</p>

 To run the SINVAD CDCGAN for a specific dataset, use the following command:

```bash
python gan_master.py --dataset mnist
```
Replace mnist with svhn or cifar10 to run the GAN for the other datasets.

Similarly, after downloading or training the gan model, run the sinvad script for cdcgan.
```bash
python sinvad_cdcgan_master.py --dataset mnist
```
Replace mnist with other datasets such as svhn, cifar10, imagenet

### Stable diffusion configuration and Run script:
Fine-tune stable diffusion using the khoya-ss platform on four different datasets. For a detailed description, please [click here](URL).

Download the fine-tuned model weights from [this link](URL). 

To run the test generator for Stable diffusion. You are required to install the following setup steps:
### Create a Virtual Environment and install packages to run the SD-based generator script

First, please make sure you have Conda installed. If not, you can download and install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. **Create a Conda virtual environment**:
   Open a terminal or Anaconda Prompt and run the following command to create a new Conda virtual environment:
   
   ```bash
   conda create --name stable_diffusion_env python=3.10
   ```
   
2. **Installing desired packages**
     The requirements setup is already provided; run requirement-sd.txt using pip.
   
   ```bash
     pip install requirements-sd.txt
   ```

3. **Run SD-based Generator**
     ```bash
     python3 sinvad_lora_mnist.py
     ```
    Similar mnist is replaced with other dataset names, svhn, cifar10, and imagenet, according to their dataset 
    name.
     
4. **Additional Notes**: keep upgrading diffusers and transformers to avoid errors.


### Download classifier ckpt for all three types of datasets:


 <p align="justify">
   
-This link download checkpoints of trained classifiers[here](https://drive.google.com/uc?export=download&id=1JS8xBjJZoOjHme0RLITsk8AYjNeKf2Ns) a single zip file containing three distinct subfolders: mnist_classifier, cifar10_classifier, 
 and svhn_classifier. Each folder includes checkpoint files for its respective datasets.
 
-For Imagenet, we used pretrained classifier weights vgg-19-bn  directly from the PyTorch repository [see pytorch Link](https://drive.google.com/uc?export=download&id=YOUR_DIRECT_DOWNLOAD_LINK_ID)
</p>

### REFERENCE:


[9].https://drive.google.com/uc?export=download&id=1MNyzsNZWE5QzoX_xIbIlm-IV6lNri3W3



