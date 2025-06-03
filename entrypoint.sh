#!/bin/bash

echo "Cleaning old model folders and zips..."
rm -rf vae sa sd cdcgan vae.zip sd.zip sa.zip cdcgan.zip

echo " Downloading model ZIPs..."

# --- Download and extract VAE ---
gdown https://drive.google.com/uc?id=1eoBlv4YvH-yGCEx9zxc0xu2dsuapbZKV -O vae.zip
unzip -o vae.zip
rm vae.zip
if [ -d "vae/vae" ]; then
    mv vae/vae/* vae/
    rm -rf vae/vae
fi

# --- Download and extract SD ---
gdown https://drive.google.com/uc?id=1q6nCAOxaQ1Dd69Kh0m_iMU6126kDh6pL -O sd.zip
unzip -o sd.zip
rm sd.zip
if [ -d "sd/sd" ]; then
    mv sd/sd/* sd/
    rm -rf sd/sd
fi

# --- Download and extract SA ---
gdown https://drive.google.com/uc?id=1dKywynVg2SRHZBFrt101EwRzlKNuiVu- -O sa.zip
unzip -o sa.zip
rm sa.zip
if [ -d "sa/sa" ]; then
    mv sa/sa/* sa/
    rm -rf sa/sa
fi

# --- Download and extract CDCGAN ---
gdown https://drive.google.com/uc?id=1MnXSukCHhtajVxtJxWpXCtXE8SFSCNKh -O cdcgan.zip
unzip -o cdcgan.zip
rm cdcgan.zip
if [ -d "cdcgan/cdcgan" ]; then
    mv cdcgan/cdcgan/* cdcgan/
    rm -rf cdcgan/cdcgan
fi

# --- Add __init__.py to all Python folders for module imports ---
#for folder in vae sd sa cdcgan; do
 #   find $folder -type d -exec touch {}/__init__.py \;
#done

echo " All model folders extracted cleanly."
echo " Launching Gradio app..."

python -u GIFTbench.py
