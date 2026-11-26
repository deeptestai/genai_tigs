"""VAEs whose latent space the GA explores (from legacy/*/vae/model.py)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split(pre_z, z_dim):
    return pre_z[:, :z_dim], pre_z[:, z_dim:]


class MlpVAE(nn.Module):
    """Fully connected VAE (MNIST)."""

    def __init__(self, img_size=28 * 28, h_dim=1600, z_dim=400):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Linear(img_size, h_dim), nn.ReLU(), nn.Linear(h_dim, z_dim * 2))
        self.dec = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, img_size))

    def encode(self, x):
        return _split(self.enc(x.view(x.shape[0], -1)), self.z_dim)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))


class ResBlock(nn.Module):
    def __init__(self, c_num):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_num, c_num, 3, padding=1), nn.ReLU(), nn.InstanceNorm2d(c_num),
            nn.Conv2d(c_num, c_num, 3, padding=1), nn.ReLU(), nn.InstanceNorm2d(c_num),
        )

    def forward(self, x):
        return x + self.layer(x)


def _conv_norm(inp, out):
    return nn.Sequential(nn.Conv2d(inp, out, 3, padding=1), nn.ReLU(), nn.InstanceNorm2d(out))


class ResConvVAE(nn.Module):
    """Residual convolutional VAE (SVHN)."""

    def __init__(self, img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800):
        super().__init__()
        self.z_dim = z_dim
        self.img_h = img_size[0]
        flat = (self.img_h ** 2) * 64 // 16
        self.enc_conv = nn.Sequential(
            _conv_norm(c_num, 32), ResBlock(32), ResBlock(32), nn.MaxPool2d(2),
            _conv_norm(32, 64), ResBlock(64), ResBlock(64), nn.MaxPool2d(2),
        )
        self.enc_linear = nn.Sequential(
            nn.Sequential(nn.Dropout(), nn.Linear(flat, h_dim), nn.ReLU()),
            nn.Sequential(nn.Dropout(), nn.Linear(h_dim, 2 * z_dim)),
        )
        self.dec_linear = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Dropout(), nn.Linear(h_dim, flat), nn.ReLU()
        )
        self.dec_deconv = nn.Sequential(
            ResBlock(64), ResBlock(64),
            nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.InstanceNorm2d(32)),
            ResBlock(32), ResBlock(32),
            nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(), nn.InstanceNorm2d(16)),
            nn.Sequential(nn.Conv2d(16, c_num, 3, padding=1), nn.Sigmoid()),
        )

    def encode(self, x):
        out = self.enc_conv(x).view(-1, (self.img_h ** 2) * 64 // 16)
        return _split(self.enc_linear(out), self.z_dim)

    def decode(self, z):
        out = self.dec_linear(z).view(-1, 64, self.img_h // 4, self.img_h // 4)
        return self.dec_deconv(out)


class StridedConvVAE(nn.Module):
    """Strided convolutional VAE (CIFAR-10)."""

    def __init__(self, c_num=3, h_dim=4000, z_dim=1024):
        super().__init__()
        self.z_dim = z_dim
        self.enc_conv1 = nn.Conv2d(c_num, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.enc_linear1 = nn.Linear(512 * 4 * 4, h_dim)
        self.enc_linear2 = nn.Linear(h_dim, 2 * z_dim)
        self.dec_linear1 = nn.Linear(z_dim, h_dim)
        self.dec_linear2 = nn.Linear(h_dim, 512 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(64, c_num, 3, padding=1)

    def encode(self, x):
        for conv in (self.enc_conv1, self.enc_conv2, self.enc_conv3, self.enc_conv4):
            x = F.relu(conv(x))
        h = F.relu(self.enc_linear1(x.view(-1, 512 * 4 * 4)))
        return _split(self.enc_linear2(h), self.z_dim)

    def decode(self, z):
        h = F.relu(self.dec_linear2(F.relu(self.dec_linear1(z)))).view(-1, 512, 4, 4)
        for deconv in (self.dec_conv1, self.dec_conv2, self.dec_conv3):
            h = F.relu(deconv(h))
        return torch.sigmoid(self.dec_conv4(h))


class ImageNetVAE(nn.Module):
    """Convolutional VAE on 128x128 ImageNet crops, in VGG-normalised pixel space."""

    def __init__(self, latent_dim=512):
        super().__init__()
        self.z_dim = latent_dim
        self.shape = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * (self.shape - 1) ** 2, 2 * latent_dim)
        self.scale = nn.Parameter(torch.tensor([0.0]))  # training-only, kept so checkpoints load
        self.fc2 = nn.Linear(latent_dim, (self.shape ** 2) * 32)
        self.conv3 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1)

    def encode(self, x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        return _split(self.fc1(F.relu(x.flatten(1))), self.z_dim)

    def decode(self, z):
        x = F.relu(self.fc2(z)).view(-1, 32, self.shape, self.shape)
        return self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))
