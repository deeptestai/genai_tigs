import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from model import VAE  # Make sure to have the VAE model definition in model.py

# Parameters (should match those used during training)
img_size = 1 * 28 * 28
h_dim = 1600
z_dim = 400
batch_size = 128

# Load the trained model
model = VAE(img_size=img_size, h_dim=h_dim, z_dim=z_dim).cuda()
model.load_state_dict(torch.load('models/MNIST_EnD.pth'))
model.eval()

# MNIST dataset (validation/test)
test_dataset = torchvision.datasets.MNIST(root='../data',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

# Data loader
test_data_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

# Compute loss on test data
total_reconst_loss = 0
total_kl_div = 0
num_samples = 0

with torch.no_grad():
    for x, _ in test_data_loader:
        x = x.cuda().view(-1, img_size)
        x_reconst, mu, log_var = model(x)
        
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        total_reconst_loss += reconst_loss.item()
        total_kl_div += kl_div.item()
        num_samples += x.size(0)

avg_reconst_loss = total_reconst_loss / num_samples
avg_kl_div = total_kl_div / num_samples

print(f"Average Reconstruction Loss: {avg_reconst_loss:.4f}, Average KL Divergence: {avg_kl_div:.4f}")
