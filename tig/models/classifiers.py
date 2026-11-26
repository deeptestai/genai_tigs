"""Classifiers under test (from legacy/*/classifier/model.py, which are identical copies)."""

import torch.nn as nn


class MnistClassifier(nn.Module):
    """Same architecture as in the original SA paper."""

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()),
            nn.MaxPool2d(2),
            nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Sequential(nn.Dropout(), nn.Linear(3136, 512), nn.ReLU()),
            nn.Sequential(nn.Dropout(), nn.Linear(512, 10)),
        )

    def forward(self, x):
        out = self.conv_layers(x)
        return self.dense_layers(out.view(-1, 64 * 7 * 7))


class VGGBlock(nn.Module):
    def __init__(self, inp, outp, lnum):
        super().__init__()
        self.first_lyr = nn.Sequential(
            nn.Conv2d(inp, outp, kernel_size=3, padding=1), nn.BatchNorm2d(outp), nn.ReLU(inplace=True)
        )
        rest = []
        for _ in range(1, lnum):
            rest += [nn.Conv2d(outp, outp, kernel_size=3, padding=1), nn.BatchNorm2d(outp), nn.ReLU(inplace=True)]
        self.next_lyrs = nn.Sequential(*rest)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(self.next_lyrs(self.first_lyr(x)))


class VGGClassifier(nn.Module):
    def __init__(self, inp, class_num):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(inp, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, class_num),
        )

    def forward(self, x):
        return self.classifier(x.view(x.size(0), -1))


class VGGNet(nn.Module):
    """32x32 RGB classifier used for SVHN and CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.layer1 = VGGBlock(3, 64, 2)
        self.layer2 = VGGBlock(64, 128, 2)
        self.layer3 = VGGBlock(128, 256, 4)
        self.layer4 = VGGBlock(256, 512, 4)
        self.layer5 = VGGBlock(512, 512, 4)
        self.vggc = VGGClassifier(512, 10)

    def forward(self, x):
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4, self.layer5):
            x = layer(x)
        return self.vggc(x)


def vgg19_bn_imagenet(pretrained=True):
    """torchvision VGG19-BN (same weights as torch.hub pytorch/vision vgg19_bn), downloaded on first use."""
    from torchvision.models import VGG19_BN_Weights, vgg19_bn

    return vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None)
