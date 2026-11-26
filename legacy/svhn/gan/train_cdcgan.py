# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

from __future__ import print_function
import argparse
import os
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# python dcgan.py --dataset svhn --dataroot /scratch/users/vision/yu_dl/raaz.rsk/data/cifar10 --imageSize 28 --cuda --outf . --manualSeed 13 --niter 100

class_num = 10  # Number of classes in the svhn dataset


# Generator
class Generator(nn.Module):
    def __init__(self, ngpu, nc=3, nz=100):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu

        # Embedding for the labels
        self.label_emb = nn.Embedding(10, 10)

        # Main generator model
        self.main = nn.Sequential(
            # Concatenated input is Z + label embedding, going into a convolution
            nn.ConvTranspose2d(
                self.nz + 10, 256, 4, 1, 0, bias=False
            ),  # output: [64, 256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, 4, 2, 1, bias=False
            ),  # output: [64, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, 4, 2, 1, bias=False
            ),  # output: [64, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # output: [64, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        z = x.view(x.size(0), -1)
        x = torch.cat([z, c], 1).view(-1, self.nz + 10, 1, 1)
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # Embedding for the labels:
        # We'll transform each label to a 1x32x32 tensor
        # (1 channel, spatial dimensions matching the image)
        self.embed = nn.Embedding(10, 32 * 32)

        self.main = nn.Sequential(
            # input is (3 + label embedding) x 32 x 32
            nn.Conv2d(nc + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (64, 16, 16)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (128, 8, 8)
            nn.Conv2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (64, 4, 4)
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        # Embed the labels and concatenate them with the image
        embedded_labels = self.embed(labels)
        y = embedded_labels.view(-1, 1, 32, 32)
        x = torch.cat([x, y], 1)
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output.view(-1, 1).squeeze(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=False,
        help="svhn | lsun | mnist |imagenet | folder | lfw | fake",
    )
    parser.add_argument(
        "--dataroot",
        default="/home/maryam/Documents/SEDL/SINVAD/cdcgan_svhn/datasets",
        required=False,
        help="path to  the svhn dataset",
    )
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=2
    )
    parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--img_size",
        type=int,
        default=32,
        help="the height / width of the input image to network",
    )
    parser.add_argument(
        "--nz", type=int, default=100, help="size of the latent z vector"
    )
    parser.add_argument(
        "--class_num", type=int, default=10, help="total number of classes"
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument(
        "--niter", type=int, default=100, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument(
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument(
        "--outf",
        default="Resultscgansvhn/",
        help="folder to output images and model checkpoints",
    )
    parser.add_argument("--manualSeed", default="123", type=int, help="manual seed")

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    dataset = dset.SVHN(
        root=opt.dataroot,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), (0.5,)),
            ]
        ),
    )
    nc = 3

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers)
    )

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    img_size = int(opt.img_size)
    if not os.path.exists(opt.outf):
       os.makedirs(opt.outf)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = Generator(ngpu, nz).to(device)
    netG.apply(weights_init)
    if opt.netG != "":
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if opt.netD != "":
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            labels = data[1].to(device)
            batch_size = real_cpu.size(0)

            output_real = netD(real_cpu, labels)
            label = torch.full(
                (batch_size,),
                real_label,
                device=device,
                dtype=torch.float32,
            )
            errD_real = criterion(output_real, label)
            errD_real.backward(retain_graph=True)
            D_x = output_real.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            fake = netG(noise, labels)

            output_fake = netD(fake.detach(), labels)
            # Create a tensor of fake labels with the same size as the batch
            label.fill_(fake_label)  # Target for fake images

            errD_fake = criterion(output_fake, label)

            errD_fake.backward(retain_graph=True)
            D_G_z1 = output_fake.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake, labels)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print(
                "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                % (
                    epoch,
                    opt.niter,
                    i,
                    len(dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )

            if i % 100 == 0:
                vutils.save_image(
                    real_cpu, "%s/real_samples.png" % opt.outf, normalize=True
                )
                fake = netG(fixed_noise, labels)
                vutils.save_image(
                    fake.detach(),
                    "%s/fake_samples_epoch_%03d.png" % (opt.outf, epoch),
                    normalize=True,
                )

        # do checkpointing
        torch.save(netG.state_dict(), "%s/netG_epoch_%d_ganbsvh.pth" % (opt.outf, epoch))
        torch.save(netD.state_dict(), "%s/netD_epoch_%d_gansvh.pth" % (opt.outf, epoch))

