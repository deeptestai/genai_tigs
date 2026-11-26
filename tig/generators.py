"""Uniform view over generative models for the GA.

Every generator exposes ``latent_dim`` and ``decode(z, labels)``, returning
images shaped as the classifier expects. Generators that can embed a real
image (VAEs) also implement ``encode``; the others draw seeds from the prior.
"""

import torch
import torch.nn.functional as F

# torchvision ImageNet normalisation, as used by the pretrained VGG19-BN.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VAEGenerator:
    conditional = False

    def __init__(self, vae, image_shape):
        self.vae = vae
        self.image_shape = image_shape
        self.latent_dim = vae.z_dim

    @torch.no_grad()
    def encode(self, images):
        mu, _ = self.vae.encode(images)
        return mu

    @torch.no_grad()
    def decode(self, z, labels=None):
        return self.vae.decode(z).view(-1, *self.image_shape)


class ConditionalGANGenerator:
    """The class label is part of the generator input, so it is fixed while the GA mutates ``z``."""

    conditional = True

    def __init__(self, generator, latent_dim):
        self.generator = generator
        self.latent_dim = latent_dim

    @torch.no_grad()
    def decode(self, z, labels):
        return self.generator(z, labels)


class BigGANGenerator:
    """BigGAN conditioned on a one-hot ImageNet class; output prepared for VGG19-BN."""

    conditional = True
    latent_dim = 128

    def __init__(self, biggan, truncation=1.0, num_classes=1000):
        self.biggan = biggan
        self.truncation = truncation
        self.num_classes = num_classes

    @torch.no_grad()
    def decode(self, z, labels):
        one_hot = F.one_hot(labels, self.num_classes).float()
        return to_vgg_input(self.biggan(z, one_hot, self.truncation))


class StableDiffusionGenerator:
    """Stable Diffusion conditioned on a text prompt per class; the GA mutates the initial noise
    latent (4x64x64 for 512x512 images) and DDIM denoises it deterministically.

    As in the legacy scripts, the seed image uses a stronger guidance (3.5) than the population
    (1.4), and each image is reduced to the classifier's input with the legacy preprocessing.
    """

    conditional = True
    latent_shape = (4, 64, 64)
    latent_dim = 4 * 64 * 64
    crossover_unit = 64 * 64  # cut between channels, as np.random.choice(4) did in the legacy GA

    def __init__(self, pipe, prompts, preprocess, steps, guidance=1.4, seed_guidance=3.5):
        self.pipe = pipe
        self.prompts = prompts
        self.preprocess = preprocess
        self.steps = steps
        self.guidance = guidance
        self.seed_guidance = seed_guidance

    @torch.no_grad()
    def decode(self, z, labels, guidance=None):
        import numpy as np
        from PIL import Image

        latents = z.view(-1, *self.latent_shape).to(self.pipe.unet.dtype)
        images = self.pipe(
            [self.prompts[int(label)] for label in labels],
            guidance_scale=self.guidance if guidance is None else guidance,
            num_inference_steps=self.steps,
            latents=latents,
            output_type="np",
        ).images
        if np.isnan(images).any():
            # Otherwise they silently become black images that the classifier labels at random.
            raise RuntimeError("Stable Diffusion returned NaN images: this GPU overflows in fp16, use --sd-precision fp32")
        # The same uint8 conversion the pipeline does for output_type="pil", as the legacy scripts used.
        pil = [Image.fromarray((image * 255).round().astype("uint8")) for image in images]
        return torch.stack([SD_PREPROCESS[self.preprocess](image) for image in pil]).to(z.device)

    def decode_seed(self, z, labels):
        return self.decode(z, labels, guidance=self.seed_guidance)


def _sd_mnist(image):
    """legacy/mnist/sd: grayscale, 28x28 (cv2 default bilinear), [0, 1]."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return torch.from_numpy(cv2.resize(gray, (28, 28))).float().div(255).unsqueeze(0)


def _sd_rgb32(image):
    """legacy/{svhn,cifar10}/sd: 32x32 nearest-neighbour, [0, 1]."""
    import cv2
    import numpy as np

    small = cv2.resize(np.array(image), (32, 32), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(small).float().div(255).permute(2, 0, 1)


def _sd_imagenet(image):
    """legacy/imagenet/sd: resize 256x256, centre crop 224, ImageNet normalisation."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])(image)


SD_PREPROCESS = {"mnist": _sd_mnist, "rgb32": _sd_rgb32, "imagenet": _sd_imagenet}


def to_vgg_input(images):
    """Legacy preprocessing of a [-1, 1] 256x256 batch: 8-bit quantisation (via PIL in the
    original), centre crop to 224 and ImageNet normalisation."""
    pixels = ((images + 1) / 2 * 255).to(torch.uint8).float() / 255
    top = (pixels.shape[-2] - 224) // 2
    left = (pixels.shape[-1] - 224) // 2
    crop = pixels[..., top:top + 224, left:left + 224]
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (crop - mean) / std


def can_encode(generator) -> bool:
    return hasattr(generator, "encode")


def _load(config, name, device) -> torch.nn.Module:
    path = config.weights.get(name)
    if path is None:  # pretrained network that downloads its own weights
        return config.networks[name]().to(device).eval()
    if not path.exists():
        raise FileNotFoundError(f"missing {name} weights for {config.model} x {config.dataset}: {path} (see README)")
    net = config.networks[name]()
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state.get("model_state_dict", state))  # training checkpoints wrap the weights
    return net.to(device).eval()


def build(config, device, sd_precision="fp16"):
    """Return ``(generator, classifier)`` for a ported combination."""
    if not config.ported:
        raise NotImplementedError(
            f"{config.model} x {config.dataset} is not ported yet; legacy script: {config.legacy_script}"
        )
    if config.model == "sd":
        lora = config.weights["lora"]
        if not lora.exists():
            raise FileNotFoundError(f"missing Stable Diffusion LoRA for {config.dataset}: {lora} (see README)")
        from tig.models.sd import stable_diffusion

        sd = config.sd
        pipe = stable_diffusion(lora, device, precision=sd_precision)
        generator = StableDiffusionGenerator(pipe, sd.prompts, sd.preprocess, sd.steps)
        return generator, _load(config, "classifier", device)
    net = _load(config, "generator", device)
    classifier = _load(config, "classifier", device)
    if config.model == "vae":
        return VAEGenerator(net, config.image_shape), classifier
    if config.model == "gan" and config.dataset.startswith("imagenet"):
        return BigGANGenerator(net), classifier
    if config.model == "gan":
        return ConditionalGANGenerator(net, latent_dim=net.nz), classifier
    raise NotImplementedError(f"no wrapper for model {config.model}")
