import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every global RNG used by the generators so a run can be replayed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_deterministic() -> None:
    """Make GPU runs repeatable: without this, cuDNN picks convolution algorithms that sum in a
    different order on every run, so the same seed gave different GAN SVHN and VAE CIFAR-10
    results on the same GPU. Call it before the first CUDA operation."""
    # cuBLAS needs a fixed workspace to be deterministic (see the PyTorch reproducibility notes).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
