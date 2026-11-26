import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LinearGenerator:
    """Tiny deterministic stand-in for a VAE/GAN: z -> 1x4x4 image."""

    latent_dim = 8

    def __init__(self):
        g = torch.Generator().manual_seed(123)
        self.w = torch.randn(self.latent_dim, 16, generator=g)

    def encode(self, images):
        return images.view(images.shape[0], -1) @ torch.linalg.pinv(self.w)

    def decode(self, z, labels=None):
        return (z @ self.w).view(-1, 1, 4, 4)  # linear, so encode(decode(z)) == z


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.w = torch.randn(16, 3, generator=g)

    def forward(self, x):
        return x.view(x.shape[0], -1) @ self.w


@pytest.fixture
def toy():
    return LinearGenerator(), LinearClassifier()
