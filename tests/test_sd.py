"""Stable Diffusion pieces that can be checked without the 2 GB base model or the LoRA weights."""

import ast
import os

import numpy as np
import pytest
import torch
from PIL import Image

from tig.config import LEGACY, PIZZA, TEDDY_BEAR, get_config
from tig.ga import crossover
from tig.generators import SD_PREPROCESS, StableDiffusionGenerator

cv2 = pytest.importorskip("cv2")


class FakePipe:
    """Records each call and returns solid-colour 512x512 images in [0, 1], as the real pipeline
    does with ``output_type="np"``."""

    def __init__(self, colour=(1.0, 0.0, 0.0)):
        self.unet = type("Unet", (), {"dtype": torch.float16})()
        self.colour = colour
        self.calls = []

    def __call__(self, prompts, guidance_scale, num_inference_steps, latents, output_type):
        assert output_type == "np"
        self.calls.append((prompts, guidance_scale, num_inference_steps, latents))
        images = np.empty((len(prompts), 512, 512, 3), dtype=np.float32)
        images[:] = self.colour
        return type("Out", (), {"images": images})()


def _legacy_prompts(dataset):
    """The ``prompts = [...]`` list of legacy/<dataset>/sd/tig_sd.py, read from its line: the
    SVHN script doesn't compile ("init_perturbation = 0.0021...  (low_mutation value)")."""
    for line in (LEGACY / dataset / "sd" / "tig_sd.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("prompts"):
            return ast.literal_eval(line.split("=", 1)[1].split("]")[0] + "]")
    raise AssertionError("no prompts list")


@pytest.mark.parametrize("dataset", ["mnist", "svhn", "cifar10"])
def test_prompts_match_legacy(dataset):
    assert list(get_config("sd", dataset).sd.prompts.values()) == _legacy_prompts(dataset)


def test_imagenet_prompts():
    assert get_config("sd", "imagenet_pizza").sd.prompts == {PIZZA: "A photo of 1pizza pizza_slice"}
    assert get_config("sd", "imagenet_teddy").sd.prompts == {TEDDY_BEAR: "A photo of 1toy_bear teddy_bear"}


def test_decode_sends_prompts_guidance_and_fp16_latents():
    pipe = FakePipe()
    gen = StableDiffusionGenerator(pipe, {3: "three", 7: "seven"}, "rgb32", steps=25)
    z = torch.randn(2, gen.latent_dim)

    out = gen.decode(z, torch.tensor([3, 7]))
    prompts, guidance, steps, latents = pipe.calls[-1]
    assert prompts == ["three", "seven"] and guidance == 1.4 and steps == 25
    assert latents.shape == (2, 4, 64, 64) and latents.dtype == torch.float16
    assert torch.equal(latents.float().view(2, -1), z.half().float())
    assert out.shape == (2, 3, 32, 32)

    assert torch.all(out[:, 0] == 1) and torch.all(out[:, 1:] == 0)  # red survives the uint8 round trip

    gen.decode_seed(z[:1], torch.tensor([3]))
    assert pipe.calls[-1][1] == 3.5  # the seed image uses the legacy init guidance


def test_nan_images_stop_with_a_hint():
    # On a GTX 1650 the fp16 UNet overflows; the pipeline then returns NaN, i.e. black images.
    gen = StableDiffusionGenerator(FakePipe(colour=(np.nan, 0.0, 0.0)), {0: "zero"}, "mnist", steps=15)
    with pytest.raises(RuntimeError, match="--sd-precision fp32"):
        gen.decode(torch.zeros(1, gen.latent_dim), torch.tensor([0]))


def test_mnist_preprocess_is_legacy_grayscale():
    out = SD_PREPROCESS["mnist"](Image.new("RGB", (512, 512), (255, 0, 0)))
    assert out.shape == (1, 28, 28)
    # cv2 RGB->GRAY: 0.299 * 255 rounds to 76
    assert torch.allclose(out, torch.full_like(out, 76 / 255))


def test_rgb32_preprocess_is_nearest_neighbour():
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[::16, ::16] = 255  # only the pixels that nearest-neighbour keeps are white
    out = SD_PREPROCESS["rgb32"](Image.fromarray(image))
    assert out.shape == (3, 32, 32) and torch.all(out == 1)


def test_imagenet_preprocess_shape_and_normalisation():
    out = SD_PREPROCESS["imagenet"](Image.new("RGB", (512, 512), (0, 0, 0)))
    assert out.shape == (3, 224, 224)
    assert torch.allclose(out[0], torch.full((224, 224), -0.485 / 0.229))


def test_crossover_cuts_only_between_channels():
    unit = StableDiffusionGenerator.crossover_unit
    mom = torch.zeros(1, StableDiffusionGenerator.latent_dim)
    dad = torch.ones_like(mom)
    rng = np.random.default_rng(0)
    cuts = set()
    for _ in range(200):
        child = crossover(mom, dad, rng, unit)
        cut = int((child == 0).sum())  # mom's genes come first
        assert cut % unit == 0
        cuts.add(cut // unit)
    assert cuts == {0, 1, 2, 3}  # np.random.choice(4) in the legacy GA


def test_flat_crossover_keeps_the_legacy_random_stream():
    # unit=1 must draw exactly what the GA drew before, so the golden files stay valid.
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    mom, dad = torch.zeros(1, 400), torch.ones(1, 400)
    cut = int(a.integers(400))
    assert torch.equal(crossover(mom, dad, b, 1), torch.cat([mom[:, :cut], dad[:, cut:]], dim=1))


def test_make_deterministic_sets_torch_flags():
    from tig.seed import make_deterministic

    before = torch.are_deterministic_algorithms_enabled()
    try:
        make_deterministic()
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"]
    finally:
        torch.use_deterministic_algorithms(before)
