from pathlib import Path

from tig.config import IMAGENET_CLASSES


def load_test_set(dataset: str, root: Path):
    """Test images as tensors. Torchvision datasets are downloaded into ``root`` on first use;
    ImageNet classes are read from ``root/<dataset>/<any subfolder>/*.jpg`` (e.g. ImageNet
    validation images of that class), labelled with the class index."""
    import torchvision
    from torchvision import transforms

    if dataset in IMAGENET_CLASSES:
        # Same pipeline as the ImageNet VAE training: VGG-normalised 128x128 crops.
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((128, 128), antialias=True),
        ])
        folder = Path(root) / dataset
        if not folder.is_dir():
            raise FileNotFoundError(f"put {dataset} images in {folder}/<subfolder>/ (see README)")
        label = IMAGENET_CLASSES[dataset]
        return torchvision.datasets.ImageFolder(str(folder), transform=transform, target_transform=lambda _: label)

    to_tensor = transforms.ToTensor()
    if dataset == "mnist":
        return torchvision.datasets.MNIST(root=str(root), train=False, transform=to_tensor, download=True)
    if dataset == "svhn":
        return torchvision.datasets.SVHN(root=str(root), split="test", transform=to_tensor, download=True)
    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root=str(root), train=False, transform=to_tensor, download=True)
    raise NotImplementedError(f"test set loader for {dataset} not ported yet")
