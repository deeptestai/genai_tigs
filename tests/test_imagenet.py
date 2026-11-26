"""ImageNet pieces that can be checked without downloading the pretrained networks."""

import pytest
import torch
from PIL import Image

from tig import seeds
from tig.config import PIZZA, TEDDY_BEAR, LEGACY, get_config
from tig.datasets import load_test_set
from tig.generators import BigGANGenerator, to_vgg_input
from tig.models.vae import ImageNetVAE


def test_imagenet_vae_matches_legacy():
    pytest.importorskip("accelerate")  # imported at the top of the legacy training script
    pytest.importorskip("transformers")
    import importlib.util

    spec = importlib.util.spec_from_file_location("legacy_imagenet_vae", LEGACY / "imagenet/vae/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(0)
    legacy = module.Model(latent_dim=512).eval()
    new = ImageNetVAE(latent_dim=512).eval()
    new.load_state_dict(legacy.state_dict())
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(new.encode(x)[0], legacy.encode(x)[0])
        z = torch.randn(2, 512)
        torch.testing.assert_close(new.decode(z), legacy.decode(z))


def test_to_vgg_input_matches_legacy_pil_pipeline():
    from torchvision import transforms

    legacy = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = torch.rand(2, 3, 256, 256) * 2 - 1  # BigGAN output range
    expected = torch.stack([legacy(transforms.ToPILImage()((img + 1) / 2)) for img in images])
    torch.testing.assert_close(to_vgg_input(images), expected)


class FakeBigGAN:
    def __call__(self, z, class_vector, truncation):
        self.calls = (z.shape, class_vector, truncation)
        return torch.zeros(z.shape[0], 3, 256, 256)


def test_biggan_generator_one_hot_and_shape():
    fake = FakeBigGAN()
    out = BigGANGenerator(fake).decode(torch.zeros(2, 128), torch.tensor([PIZZA, TEDDY_BEAR]))
    assert out.shape == (2, 3, 224, 224)
    _, class_vector, truncation = fake.calls
    assert class_vector.shape == (2, 1000) and truncation == 1.0
    assert class_vector[0, PIZZA] == 1 and class_vector[1, TEDDY_BEAR] == 1 and class_vector.sum() == 2


def test_prior_seeds_use_fixed_class_and_truncation():
    config = get_config("gan", "imagenet_pizza")
    drawn = list(seeds.from_prior(BigGANGenerator(None), 20, 0, "cpu", classes=config.classes, bound=config.prior_bound))
    assert {label for _, _, label in drawn} == {PIZZA}
    assert all(z.shape == (1, 128) and z.abs().max() <= 2.0 for _, z, _ in drawn)


def test_imagenet_test_set_from_folder(tmp_path):
    folder = tmp_path / "imagenet_teddy" / "val"
    folder.mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (300, 260), (i * 100, 50, 50)).save(folder / f"{i}.jpg")
    test_set = load_test_set("imagenet_teddy", tmp_path)
    image, label = test_set[0]
    assert image.shape == (3, 128, 128) and label == TEDDY_BEAR


def test_missing_imagenet_folder_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="imagenet_pizza"):
        load_test_set("imagenet_pizza", tmp_path)
