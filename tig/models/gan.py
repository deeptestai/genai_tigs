"""Conditional DCGAN generators (from legacy/*/gan/train_cdcgan.py).

All three datasets share the same conditioning (a 10-d label embedding
concatenated to ``z``); only the transposed-convolution stack differs.
The SVHN and CIFAR-10 legacy generators are identical.
"""

import torch
import torch.nn as nn


def _up(inp, out, kernel, stride, padding):
    return [nn.ConvTranspose2d(inp, out, kernel, stride, padding, bias=False), nn.BatchNorm2d(out), nn.ReLU(True)]


class CDCGANGenerator(nn.Module):
    def __init__(self, layers, nz=100, n_classes=10):
        super().__init__()
        self.nz = nz
        self.label_emb = nn.Embedding(n_classes, 10)
        self.main = nn.Sequential(*layers)

    def forward(self, z, labels):
        x = torch.cat([z.view(z.size(0), -1), self.label_emb(labels)], 1)
        return self.main(x.view(-1, self.nz + 10, 1, 1))


def mnist_cdcgan(nz=100):
    """110x1x1 -> 1x28x28."""
    return CDCGANGenerator(
        [*_up(nz + 10, 256, 7, 1, 0), *_up(256, 128, 4, 2, 1), nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False), nn.Tanh()],
        nz=nz,
    )


def rgb32_cdcgan(nz=100):
    """110x1x1 -> 3x32x32 (SVHN, CIFAR-10)."""
    return CDCGANGenerator(
        [
            *_up(nz + 10, 256, 4, 1, 0), *_up(256, 128, 4, 2, 1), *_up(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh(),
        ],
        nz=nz,
    )


def biggan_deep_256():
    """Pretrained BigGAN-deep 256x256 (pip install pytorch-pretrained-biggan), downloaded on first use."""
    from pytorch_pretrained_biggan import BigGAN

    return BigGAN.from_pretrained("biggan-deep-256")
