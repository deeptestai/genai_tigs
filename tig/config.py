"""Registry of every (model, dataset) combination: networks, weights and GA bounds.

Values are copied from the legacy ``legacy/<dataset>/<model>/tig_*.py`` scripts. Adding a
combination means adding one ``TigConfig`` entry; nothing else changes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# Original scripts of the paper, one folder per dataset and model: legacy/<dataset>/<model>/.
LEGACY = REPO_ROOT / "legacy"
# Checkpoints published on Google Drive (see README) go here, one folder per dataset.
WEIGHTS = REPO_ROOT / "weights"

MODELS = ("vae", "gan", "sd")
DATASETS = ("mnist", "svhn", "cifar10", "imagenet_pizza", "imagenet_teddy")
MUTATIONS = ("low", "high")
# Every legacy script sets the initial perturbation to a fixed fraction of the
# latent range (max_val - min_val); the fraction is the mutation extent.
MUTATION_FRACTION = {"low": 0.0002, "high": 0.002}
# ImageNet-1k class indices used by the paper.
PIZZA, TEDDY_BEAR = 963, 850
IMAGENET_CLASSES = {"imagenet_pizza": PIZZA, "imagenet_teddy": TEDDY_BEAR}


@dataclass(frozen=True)
class SDSettings:
    """Stable Diffusion specifics, copied from legacy/<dataset>/sd/tig_sd.py."""

    prompts: Dict[int, str]  # class index -> prompt with the LoRA trigger word
    steps: int  # DDIM inference steps
    preprocess: str  # key of tig.generators.SD_PREPROCESS


@dataclass(frozen=True)
class TigConfig:
    model: str
    dataset: str
    min_val: float = float("-inf")
    max_val: float = float("inf")
    # Checkpoint files; a network missing here downloads its own pretrained weights.
    weights: Dict[str, Path] = field(default_factory=dict)
    # Name -> factory for the generator's network and the classifier.
    networks: Dict[str, Callable[[], torch.nn.Module]] = field(default_factory=dict)
    image_shape: Tuple[int, int, int] = (1, 28, 28)
    # Classes the seeds are drawn from (None = all 10 digits/classes).
    classes: Optional[Tuple[int, ...]] = None
    # Prior seeds are drawn from N(0, I) truncated to [-bound, bound] (None = untruncated).
    prior_bound: Optional[float] = None
    # Default GA batch size (None = whole population); keeps large models within small GPUs.
    batch_size: Optional[int] = None
    sd: Optional[SDSettings] = None
    legacy_script: Optional[Path] = None

    @property
    def ported(self) -> bool:
        return self.legacy_script is None

    def perturbation(self, mutation: str) -> float:
        return MUTATION_FRACTION[mutation] * (self.max_val - self.min_val)


def _mnist_classifier():
    from tig.models.classifiers import MnistClassifier
    return MnistClassifier()


def _vgg():
    from tig.models.classifiers import VGGNet
    return VGGNet()


def _vgg19_bn_imagenet():
    from tig.models.classifiers import vgg19_bn_imagenet
    return vgg19_bn_imagenet()


def _mnist_vae():
    from tig.models.vae import MlpVAE
    return MlpVAE(img_size=28 * 28, h_dim=1600, z_dim=400)


def _svhn_vae():
    from tig.models.vae import ResConvVAE
    return ResConvVAE(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800)


def _cifar10_vae():
    from tig.models.vae import StridedConvVAE
    return StridedConvVAE(c_num=3, h_dim=4000, z_dim=1024)


def _imagenet_vae():
    from tig.models.vae import ImageNetVAE
    return ImageNetVAE(latent_dim=512)


def _mnist_gan():
    from tig.models.gan import mnist_cdcgan
    return mnist_cdcgan(nz=100)


def _rgb32_gan():
    from tig.models.gan import rgb32_cdcgan
    return rgb32_cdcgan(nz=100)


def _biggan():
    from tig.models.gan import biggan_deep_256
    return biggan_deep_256()


_SVHN_CLASSIFIER = WEIGHTS / "svhn/SVHN_vggnet.pth"
_CIFAR10_CLASSIFIER = WEIGHTS / "cifar10/CIFAR10_cifar10_train.pynet.pth"
_MNIST_CLASSIFIER = LEGACY / "mnist/classifier/weights/mnist_classifier.pth"
_RGB = (3, 32, 32)

CONFIGS = {
    ("vae", "mnist"): TigConfig(
        model="vae", dataset="mnist",
        min_val=-3.86494088172913, max_val=3.45633792877197,
        weights={"generator": LEGACY / "mnist/vae/weights/mnist_vae.pth", "classifier": _MNIST_CLASSIFIER},
        networks={"generator": _mnist_vae, "classifier": _mnist_classifier},
    ),
    ("vae", "svhn"): TigConfig(
        model="vae", dataset="svhn",
        min_val=-5.26724052429199, max_val=4.72675895690918,
        weights={"generator": WEIGHTS / "svhn/svhn_convend.pth", "classifier": _SVHN_CLASSIFIER},
        networks={"generator": _svhn_vae, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("vae", "cifar10"): TigConfig(
        model="vae", dataset="cifar10",
        min_val=-4.15868806838989, max_val=3.85224413871765,
        weights={"generator": WEIGHTS / "cifar10/cifar10_convend.pth", "classifier": _CIFAR10_CLASSIFIER},
        networks={"generator": _cifar10_vae, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("gan", "mnist"): TigConfig(
        model="gan", dataset="mnist",
        min_val=-4.78276300430298, max_val=4.08758640289307,
        weights={"generator": LEGACY / "mnist/gan/weights/generator_epoch59.pth", "classifier": _MNIST_CLASSIFIER},
        networks={"generator": _mnist_gan, "classifier": _mnist_classifier},
    ),
    ("gan", "svhn"): TigConfig(
        model="gan", dataset="svhn",
        min_val=-4.07097911834717, max_val=4.55557680130005,
        weights={"generator": LEGACY / "svhn/gan/weights/generator_epoch99.pth", "classifier": _SVHN_CLASSIFIER},
        networks={"generator": _rgb32_gan, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("gan", "cifar10"): TigConfig(
        model="gan", dataset="cifar10",
        min_val=-4.4389796257019, max_val=4.66634130477905,
        weights={
            "generator": LEGACY / "cifar10/gan/weights/generator_epoch699.pth",
            "classifier": _CIFAR10_CLASSIFIER,
        },
        networks={"generator": _rgb32_gan, "classifier": _vgg},
        image_shape=_RGB,
    ),
}

# ImageNet: one fixed class per dataset, torchvision's pretrained VGG19-BN as classifier.
for _dataset, _cls in IMAGENET_CLASSES.items():
    CONFIGS[("vae", _dataset)] = TigConfig(
        model="vae", dataset=_dataset,
        min_val=-5.22431135177612, max_val=5.62481260299683,
        weights={"generator": WEIGHTS / "imagenet/imagenet_vae_model_epoch_41.pth"},
        networks={"generator": _imagenet_vae, "classifier": _vgg19_bn_imagenet},
        image_shape=(3, 128, 128),
        classes=(_cls,),
    )
    CONFIGS[("gan", _dataset)] = TigConfig(
        model="gan", dataset=_dataset,
        min_val=-1.99990940093994, max_val=1.99996650218964,
        networks={"generator": _biggan, "classifier": _vgg19_bn_imagenet},
        image_shape=(3, 224, 224),
        classes=(_cls,),
        prior_bound=2.0,  # truncated_noise_sample(truncation=1) in the legacy script
        batch_size=5,  # 25 BigGAN-deep-256 images at once need about 4.8 GB of GPU memory
    )

# Stable Diffusion 1.5 + one LoRA per dataset (Google Drive, see README) in weights/sd/.
# Latent bounds, prompts and steps come from legacy/<dataset>/sd/tig_sd.py.
_SD = WEIGHTS / "sd"
# Verbatim, trailing spaces included (a test compares them with the legacy scripts).
_SD_MNIST_PROMPTS = [
    "A photo of Z0ero Number0", "A photo of one1 Number1", "A photo of two2 Number2 ", "A photo of three3 Number3",
    "A photo of Four4 Number4", "A photo of Five5 Number5", "A photo of Six6 Number6", "A photo of Seven7 Number7 ",
    "A photo of Eight8 Number8", "A photo of Nine9 Number9",
]
_SD_SVHN_PROMPTS = [f"A photo of HouseNo{i} Hnumber{i}" + (" " if i in (2, 7) else "") for i in range(10)]
_SD_CIFAR10_PROMPTS = [
    "A photo of A1plane0 cifar10_0", "A photo of car1 cifar10_1", "A photo of bird2 cifar10_2",
    "A photo of cat3 cifar10_3", "A photo of deer4 cifar10_4", "A photo of dog5 cifar10_5",
    "A photo of frog6 cifar10_6", "A photo of horse7 cifar10_7", "A photo of ship8 cifar10_8",
    "A photo of truck9 cifar10_9",
]
CONFIGS[("sd", "mnist")] = TigConfig(
    model="sd", dataset="mnist",
    min_val=-5.41362476348877, max_val=5.42081117630005,
    weights={"lora": _SD / "Mnist_Lora_sdv1.5-000005.safetensors", "classifier": _MNIST_CLASSIFIER},
    networks={"classifier": _mnist_classifier},
    batch_size=1,
    sd=SDSettings(dict(enumerate(_SD_MNIST_PROMPTS)), steps=15, preprocess="mnist"),
)
CONFIGS[("sd", "svhn")] = TigConfig(
    model="sd", dataset="svhn",
    min_val=-5.62070274353027, max_val=5.27061462402344,
    weights={"lora": _SD / "svhn_finetune_lorav1.5-000005.safetensors", "classifier": _SVHN_CLASSIFIER},
    networks={"classifier": _vgg},
    image_shape=_RGB,
    batch_size=1,
    sd=SDSettings(dict(enumerate(_SD_SVHN_PROMPTS)), steps=25, preprocess="rgb32"),
)
CONFIGS[("sd", "cifar10")] = TigConfig(
    model="sd", dataset="cifar10",
    min_val=-5.53951740264893, max_val=5.377610206604,
    weights={"lora": _SD / "cifar10_finetune_lorav1.5-000005.safetensors", "classifier": _CIFAR10_CLASSIFIER},
    networks={"classifier": _vgg},
    image_shape=_RGB,
    batch_size=1,
    sd=SDSettings(dict(enumerate(_SD_CIFAR10_PROMPTS)), steps=25, preprocess="rgb32"),
)
# The published weights have one LoRA per ImageNet class. The legacy script has only the pizza
# prompt; the teddy one is the training folder stored in its LoRA ("25_1toy_bear teddy_bear").
_SD_IMAGENET = {
    PIZZA: ("pizza_imgnet-000005.safetensors", "A photo of 1pizza pizza_slice"),
    TEDDY_BEAR: ("teddy_bear_imagenet-000005.safetensors", "A photo of 1toy_bear teddy_bear"),
}
for _dataset, _cls in IMAGENET_CLASSES.items():
    _lora, _prompt = _SD_IMAGENET[_cls]
    CONFIGS[("sd", _dataset)] = TigConfig(
        model="sd", dataset=_dataset,
        min_val=-5.3243408203125, max_val=5.55196666717529,
        weights={"lora": _SD / _lora},
        networks={"classifier": _vgg19_bn_imagenet},
        image_shape=(3, 224, 224),
        classes=(_cls,),
        batch_size=1,
        sd=SDSettings({_cls: _prompt}, steps=25, preprocess="imagenet"),
    )


def get_config(model: str, dataset: str) -> TigConfig:
    try:
        return CONFIGS[(model, dataset)]
    except KeyError:
        raise ValueError(f"unknown combination: {model} x {dataset}") from None
