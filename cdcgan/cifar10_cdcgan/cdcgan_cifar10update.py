import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torchvision.utils import save_image

# Hyperparameters
epochs = 10000
batch_size = 50
noise_size = 2048
num_classes = 10
img_size = 32
lr = 0.0002
beta1 = 0.5
save_dir = './results'  # Directory to save weights and sample images

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

class Generator(nn.Module):
    def __init__(self, noise_size, num_classes, img_size):
        super(Generator, self).__init__()
        self.noise_size = noise_size
        self.label_emb = nn.Embedding(num_classes, noise_size)
        self.init_size = img_size // 16  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(noise_size, 512 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256, 0.9),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128, 0.9),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64, 0.9),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels).view(labels.size(0), -1)
        model_input = noise * label_input
        out = self.l1(model_input)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(4, 64, 3, 2, 1),
            nn.BatchNorm2d(64, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512, 0.9),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(512 * (img_size // 16) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_input = self.label_emb(labels).view(labels.size(0), 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label_input), dim=1)
        out = self.conv_blocks(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def train(epochs, dataloader, G, D, adversarial_loss, optimizer_G, optimizer_D):
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)

            valid = torch.ones((batch_size, 1), device='cuda', dtype=torch.float32) - (torch.rand(batch_size, 1, device='cuda') * 0.1)
            fake = torch.zeros((batch_size, 1), device='cuda', dtype=torch.float32) + (torch.rand(batch_size, 1, device='cuda') * 0.1)

            real_imgs = imgs.cuda()
            labels = labels.cuda()

            # Train Generator
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, noise_size, device='cuda')
            gen_labels = torch.randint(0, num_classes, (batch_size,), device='cuda')
            gen_imgs = G(z, gen_labels)
            
            validity = D(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            
            real_pred = D(real_imgs, labels)
            d_real_loss = adversarial_loss(real_pred, valid)
            
            fake_pred = D(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(fake_pred, fake)
            
            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            
            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            
            # Save sample images and model checkpoints
            if i % 100 == 0:
                save_image(gen_imgs.data[:25], f"{save_dir}/sample_{epoch}_{i}.png", nrow=5, normalize=True)
                torch.save(G.state_dict(), f"{save_dir}/generator_{epoch}.pth")
                torch.save(D.state_dict(), f"{save_dir}/discriminator_{epoch}.pth")

# Initialize models
G = Generator(noise_size, num_classes, img_size).cuda()
D = Discriminator(num_classes, img_size).cuda()

# Loss and optimizers
adversarial_loss = torch.nn.BCELoss().cuda()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
train(epochs, dataloader, G, D, adversarial_loss, optimizer_G, optimizer_D)
