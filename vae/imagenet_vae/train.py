import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import os
import logging
from PIL import Image
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define global variables
img_dim = 128  # Set to 128x128
img_chn = 3
latent_dim = 512
batch_size = 128
epochs = 100
learning_rate = 1e-3
weight_decay = 0.01  # Define weight decay parameter

# Setup logging
logging.basicConfig(filename='trainingvae.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Define the Model
class Model(nn.Module):

    def __init__(self, latent_dim=512):
        super().__init__()

        self.latent_dim = latent_dim
        self.shape = 32

        # encode
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)  # 128x128 -> 64x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2) # 64x64 -> 32x32
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (self.shape - 1) ** 2, 2 * self.latent_dim)
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.tensor([0.0]))

        # decode
        self.fc2 = nn.Linear(self.latent_dim, (self.shape ** 2) * 32)
        self.conv3 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)  # 32x32 -> 64x64
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 64x64 -> 128x128
        self.conv5 = nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1)   # 128x128 -> 128x128

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.flatten(x))
        x = self.fc1(x)
        mean, logvar = torch.split(x, self.latent_dim, dim=1)
        return mean, logvar

    def decode(self, eps):
        x = self.relu(self.fc2(eps))
        x = torch.reshape(x, (x.shape[0], 32, self.shape, self.shape))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

    def reparamatrize(self, mean, std):
        q = torch.distributions.Normal(mean, std)
        return q.rsample()

    def kl_loss(self, z, mean, std):
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        q = torch.distributions.Normal(mean, torch.exp(std / 2))
        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)
        kl_loss = (log_qzx - log_pz)
        kl_loss = kl_loss.sum(-1)
        return kl_loss

    def gaussian_likelihood(self, inputs, outputs, scale):
        dist = torch.distributions.Normal(outputs, torch.exp(scale))
        log_pxz = dist.log_prob(inputs)
        return log_pxz.sum(dim=(1, 2, 3))

    def loss_fn(self, inputs, outputs, z, mean, std):
        kl_loss = self.kl_loss(z, mean, std)
        rec_loss = self.gaussian_likelihood(inputs, outputs, self.scale)
        return torch.mean(kl_loss - rec_loss)

    def forward(self, inputs):
        mean, logvar = self.encode(inputs)
        std = torch.exp(logvar / 2)
        z = self.reparamatrize(mean, std)
        outputs = self.decode(z)
        loss = self.loss_fn(inputs, outputs, z, mean, std)
        return loss, (outputs, z, mean, std)

def run():
    def train_and_evaluate_loop(train_loader, model, optimizer, epoch, best_loss, lr_scheduler=None):
        train_loss = 0
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            model.train()
            loss, _ = model(inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if lr_scheduler:
                lr_scheduler.step()

        train_loss /= len(train_loader)

        print(f"Epoch:{epoch+1} | Train Loss:{train_loss}")  # Print training loss

        if train_loss <= best_loss:
            print(f"Loss Decreased from {best_loss} to {train_loss}")

            best_loss = train_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }
            torch.save(checkpoint, f'./vae_checkpoints/imagenet_vae_model_epoch_{epoch+1}.pth')

        return best_loss

    accelerator = Accelerator()
    print(f"{accelerator.device} is used")

    model = Model().to(accelerator.device)
    
    # Train
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=preprocess)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_dl))

    model, train_dl, optimizer, lr_scheduler = accelerator.prepare(model, train_dl, optimizer, lr_scheduler)

    best_loss = 9999999
    start_time = time.time()
    for epoch in range(epochs):
        print(f"Epoch Started:{epoch+1}")
        best_loss = train_and_evaluate_loop(train_dl, model, optimizer, epoch, best_loss, lr_scheduler)

        end_time = time.time()
        print(f"Time taken by epoch {epoch+1} is {end_time-start_time:.2f}s")
        start_time = end_time

    return best_loss

if __name__ == "__main__":
    # Preprocess transform definition
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),  # Resize to 256x256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet means and stds
        transforms.Resize((128, 128)),  # Resize to 128x128
    ])

    # Ensure the path to your dataset is correct
    DATASET_DIR = "./imagenet"  # Replace with your ImageNet dataset path
    TRAIN_DIR = os.path.join(DATASET_DIR, 'train')

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"The directory {TRAIN_DIR} does not exist. Please check your dataset path.")

    run()
