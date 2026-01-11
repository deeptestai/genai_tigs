import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse
from cdcgan_cifar10 import Generator,Discriminator
def evaluate_model(netG, netD, criterion, dataloader, device, nz):
    netG.eval()
    netD.eval()
    D_losses = []
    G_losses = []

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            labels = data[1].to(device)
            batch_size = real_cpu.size(0)

            # Compute the discriminator loss on real images
            output_real = netD(real_cpu, labels)
            label_real = torch.full((batch_size,), 1, device=device, dtype=torch.float32)
            errD_real = criterion(output_real, label_real)
            D_losses.append(errD_real.item())

            # Compute the discriminator loss on fake images
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise, labels)
            output_fake = netD(fake, labels)
            label_fake = torch.full((batch_size,), 0, device=device, dtype=torch.float32)
            errD_fake = criterion(output_fake, label_fake)
            D_losses.append(errD_fake.item())

            # Compute the generator loss
            label = torch.full((batch_size,), 1, device=device, dtype=torch.float32)  # Create a tensor of real labels
            output = netD(fake, labels)
            errG = criterion(output, label)
            G_losses.append(errG.item())

    average_D_loss = np.mean(D_losses)
    average_G_loss = np.mean(G_losses)
    return average_D_loss, average_G_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False, help="cifar10 | lsun | mnist |imagenet | folder | lfw | fake")
    parser.add_argument("--dataroot", default="./data", required=False, help="path to the dataset")
    parser.add_argument("--workers", type=int, help="number of data loading workers", default=2)
    parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
    parser.add_argument("--img_size", type=int, default=32, help="the height / width of the input image to network")
    parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
    parser.add_argument("--class_num", type=int, default=10, help="total number of classes")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--niter", type=int, default=50, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate, default=0.0002")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--netG", default="./weights/netG_epoch_699.pth", help="path to netG (to continue training)")
    parser.add_argument("--netD", default="./weights/netD_epoch_699.pth", help="path to netD (to continue training)")
    parser.add_argument("--outf", default="ResultscdcganCifar/", help="folder to output images and model checkpoints")
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

    dataset = dset.CIFAR10(root=opt.dataroot, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5,))
    ]))
    nc = 3

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    img_size = int(opt.img_size)

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

    average_D_loss, average_G_loss = evaluate_model(netG, netD, criterion, dataloader, device, nz)
    print(f"Average Loss of Discriminator: {average_D_loss:.4f}")
    print(f"Average Loss of Generator: {average_G_loss:.4f}")





 
