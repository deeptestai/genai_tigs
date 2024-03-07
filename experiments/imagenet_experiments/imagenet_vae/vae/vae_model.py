import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

img_chn = 3
img_dim = 128


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Encoder(nn.Module):
    def __init__(self, img_dim, img_chn, latent_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_chn, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mean = nn.Linear(128 * (img_dim // 32) ** 2, latent_dim)
        self.fc_log_var = nn.Linear(128 * (img_dim // 32) ** 2, latent_dim)
        self.sampling = Sampling()

    def forward(self, x):
        x = self.conv_layers(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return [z_mean, z_log_var, self.sampling(z_mean, z_log_var)]


class Decoder(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * (img_dim // 32) ** 2)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, img_chn, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, img_dim // 32, img_dim // 32)
        x = self.deconv_layers(x)
        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return self.decoder(z), z_mean, z_log_var

    def training_step(self, img, optimizer):
        # Forward pass
        z_mean, z_log_var, z = self.encoder(img)
        reconstruction = self.decoder(z)

        # Reconstruction loss (Mean Squared Error)
        reconstruction_loss = F.mse_loss(reconstruction, img, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        # Total loss
        total_loss = reconstruction_loss + kl_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Optionally, you can track loss values here if you want
        return {
            "loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_loss": kl_loss.item(),
        }
