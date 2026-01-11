import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from model import VAE
import time  # Import the time module

# Create sample directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
img_size = 1 * 28 * 28
h_dim = 1600
z_dim = 400
num_epochs = 100
batch_size = 128
learning_rate = 2e-4

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

model = VAE(img_size=img_size, h_dim=h_dim, z_dim=z_dim)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start the overall training time measurement
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Start time for the current epoch
    
    for i, (x, _) in enumerate(data_loader):
        x = x.cuda().view(-1, img_size)
        x_reconst, mu, log_var = model(x)
        
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))

    epoch_end_time = time.time()  # End time for the current epoch
    print("Epoch {} completed in {:.2f} seconds".format(epoch+1, epoch_end_time - epoch_start_time))
    
    with torch.no_grad():
        z = torch.randn(batch_size, z_dim).cuda()
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

end_time = time.time()  # End the overall training time measurement
total_training_time = end_time - start_time
print("Total training time: {:.2f} seconds".format(total_training_time))

torch.save(model.state_dict(), 'models/MNIST_EnD2.pth')
