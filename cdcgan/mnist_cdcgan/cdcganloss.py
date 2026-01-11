import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import argparse
from cdcgan_mnist import Generator, Discriminator
# Set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", default="", help="path to the MNIST dataset")
parser.add_argument("--workers", type=int, default=2, help="number of data loading workers")
parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
parser.add_argument("--img_size", type=int, default=28, help="the height / width of the input image to network")
parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument("--cuda", action="store_true", help="enables cuda")
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument("--netG", default="./datasets/netG_epoch_10cdcgan.pth", help="path to netG (to continue training)")
parser.add_argument("--netD", default="./datasets/netD_epoch_10cdcgan.pth", help="path to netD (to continue training)")

opt = parser.parse_args()

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)

# Initialize the models
netG = Generator(ngpu, nz).to(device)
netD = Discriminator(ngpu).to(device)

# Load the weights
netG.load_state_dict(torch.load(opt.netG))
netD.load_state_dict(torch.load(opt.netD))

# Set the models to evaluation mode
netG.eval()
netD.eval()

# Prepare the dataset and dataloader
dataset = dset.MNIST(
    root=opt.dataroot,
    download=True,
    train=False,  # Use test set
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

# Define the loss function
criterion = nn.BCELoss()

# Evaluate the models
total_loss_D = 0
total_loss_G = 0
with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        real_cpu = data[0].to(device)
        labels = data[1].to(device)
        batch_size = real_cpu.size(0)

        # Evaluate Discriminator
        output_real = netD(real_cpu, labels)
        label_real = torch.full((batch_size,), 1, device=device, dtype=torch.float32)
        errD_real = criterion(output_real, label_real)

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise, labels)
        output_fake = netD(fake, labels)
        label_fake = torch.full((batch_size,), 0, device=device, dtype=torch.float32)
        errD_fake = criterion(output_fake, label_fake)

        errD = errD_real + errD_fake
        total_loss_D += errD.item()

        # Evaluate Generator
        output = netD(fake, labels)
        errG = criterion(output, label_real)
        total_loss_G += errG.item()

    avg_loss_D = total_loss_D / len(dataloader)
    avg_loss_G = total_loss_G / len(dataloader)

print(f"Average Loss of Discriminator: {avg_loss_D:.4f}")
print(f"Average Loss of Generator: {avg_loss_G:.4f}")
