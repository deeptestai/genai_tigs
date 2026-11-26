"""Where the GA starting points come from.

Each source yields ``(key, z0, label)``: ``key`` identifies the seed in the
results, ``z0`` has shape ``[1, latent_dim]`` and ``label`` is the class the
classifier is expected to predict.
"""

import numpy as np
import torch


def from_test_set(generator, test_set, num, seed, device):
    """Embed real test images (needs a generator with ``encode``, e.g. a VAE)."""
    rng = np.random.default_rng(seed)
    for idx in rng.choice(len(test_set), size=min(num, len(test_set)), replace=False):
        image, label = test_set[int(idx)]
        yield int(idx), generator.encode(image.unsqueeze(0).to(device)), int(label)


def from_prior(generator, num, seed, device, classes=None, bound=None):
    """Sample ``z ~ N(0, I)`` (truncated to ``[-bound, bound]`` if given) and a class
    from ``classes`` (default: the 10 classes 0-9), for conditional generators."""
    classes = tuple(range(10)) if classes is None else tuple(classes)
    g = torch.Generator().manual_seed(seed)
    for n in range(num):
        z = torch.randn(1, generator.latent_dim, generator=g)
        if bound is not None:
            while (outside := z.abs() > bound).any():  # resample: a true truncated normal
                z[outside] = torch.randn(int(outside.sum()), generator=g)
        label = classes[int(torch.randint(len(classes), (1,), generator=g))]
        yield n, z.to(device), label
