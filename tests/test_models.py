"""The networks in tig/models must be drop-in replacements for the legacy ones:
same checkpoint keys and shapes, and the same outputs for the same weights.

This is what lets the SVHN/CIFAR-10 checkpoints from Google Drive load without
having them in the repo.
"""

import importlib.util

import pytest
import torch

from tig.config import LEGACY, get_config
from tig.models.classifiers import MnistClassifier, VGGNet
from tig.models.gan import mnist_cdcgan, rgb32_cdcgan
from tig.models.vae import MlpVAE, ResConvVAE, StridedConvVAE


def _legacy(rel_path, cls_name):
    spec = importlib.util.spec_from_file_location(f"legacy_{cls_name}", LEGACY / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, cls_name)


CASES = [
    ("mnist/vae/model.py", "VAE", dict(img_size=784, h_dim=1600, z_dim=400),
     MlpVAE, dict(img_size=784, h_dim=1600, z_dim=400), (1, 784)),
    ("svhn/vae/model.py", "ConvVAE", dict(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800),
     ResConvVAE, dict(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800), (1, 3, 32, 32)),
    ("cifar10/vae/model.py", "ConvVAE", dict(c_num=3, h_dim=4000, z_dim=1024),
     StridedConvVAE, dict(c_num=3, h_dim=4000, z_dim=1024), (1, 3, 32, 32)),
]


@pytest.mark.parametrize("path,legacy_cls,legacy_kw,new_cls,new_kw,in_shape", CASES, ids=lambda v: getattr(v, "__name__", None))
def test_vae_matches_legacy(path, legacy_cls, legacy_kw, new_cls, new_kw, in_shape):
    torch.manual_seed(0)
    legacy = _legacy(path, legacy_cls)(**legacy_kw).eval()
    new = new_cls(**new_kw).eval()
    new.load_state_dict(legacy.state_dict())  # raises on any key/shape mismatch

    x = torch.rand(in_shape)
    with torch.no_grad():
        legacy_mu, _ = legacy.encode(x)
        new_mu, _ = new.encode(x)
        torch.testing.assert_close(new_mu, legacy_mu)
        torch.testing.assert_close(new.decode(new_mu), legacy.decode(legacy_mu))


@pytest.mark.parametrize("path,cls,new_cls,in_shape", [
    ("mnist/classifier/model.py", "MnistClassifier", MnistClassifier, (2, 1, 28, 28)),
    ("svhn/classifier/model.py", "VGGNet", VGGNet, (2, 3, 32, 32)),
])
def test_classifier_matches_legacy(path, cls, new_cls, in_shape):
    torch.manual_seed(0)
    legacy = _legacy(path, cls)().eval()
    new = new_cls().eval()
    new.load_state_dict(legacy.state_dict())
    x = torch.rand(in_shape)
    with torch.no_grad():
        torch.testing.assert_close(new(x), legacy(x))


GAN_CASES = [
    ("mnist/gan/train_cdcgan.py", dict(nc=1), mnist_cdcgan),
    ("svhn/gan/train_cdcgan.py", dict(nc=3), rgb32_cdcgan),
    ("cifar10/gan/train_cdcgan.py", dict(nc=3), rgb32_cdcgan),
]


@pytest.mark.parametrize("path,legacy_kw,factory", GAN_CASES, ids=["mnist", "svhn", "cifar10"])
def test_gan_matches_legacy(path, legacy_kw, factory):
    pytest.importorskip("matplotlib")  # imported at the top of the legacy training scripts
    torch.manual_seed(0)
    legacy = _legacy(path, "Generator")(ngpu=1, nz=100, **legacy_kw).eval()
    new = factory().eval()
    new.load_state_dict(legacy.state_dict())
    z, labels = torch.randn(4, 100), torch.tensor([0, 3, 6, 9])
    with torch.no_grad():
        out = new(z, labels)
        torch.testing.assert_close(out, legacy(z.view(4, 100, 1, 1), labels).view_as(out))


@pytest.mark.parametrize("dataset,shape", [("mnist", (1, 28, 28)), ("svhn", (3, 32, 32)), ("cifar10", (3, 32, 32))])
def test_gan_loads_published_weights(dataset, shape):
    from tig.generators import _load

    config = get_config("gan", dataset)
    net = _load(config, "generator", torch.device("cpu"))
    with torch.no_grad():
        images = net(torch.randn(3, net.nz), torch.tensor([0, 5, 9]))
    assert images.shape == (3, *shape)
    assert images.min() >= -1 and images.max() <= 1  # tanh output
