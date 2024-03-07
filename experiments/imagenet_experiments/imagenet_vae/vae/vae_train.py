import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from vae_model import VAE, Encoder, Decoder  # Import from your vae_model.py
from PIL import Image
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define global variables
img_dim = 128
img_chn = 3
latent_dim = 1024
num_epochs = 60


def save_checkpoint(encoder, decoder, epoch, checkpoint_dir="checkpoints"):
    """Save a model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    encoder_path = os.path.join(checkpoint_dir, f"encoder_epoch_{epoch}.pth")
    decoder_path = os.path.join(checkpoint_dir, f"decoder_epoch_{epoch}.pth")
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Checkpoint saved for epoch {epoch}: {encoder_path}, {decoder_path}")


# Now use your RobustImageFolder class instead of ImageFolder
transform = transforms.Compose(
    [
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# Assuming ImageNet data is stored in 'DATASET_DIR'
DATASET_DIR = "./imagenet"  # Replace with your ImageNet dataset path
TRAIN_DIR = os.path.join(DATASET_DIR, "train")  # Path to the training data
imagenet_train = ImageFolder(TRAIN_DIR, transform=transform)
train_loader = DataLoader(imagenet_train, batch_size=128, shuffle=True, num_workers=4)

# Initialize VAE, criterion, and optimizer
vae = VAE(Encoder(img_dim, img_chn, latent_dim), Decoder(latent_dim, img_dim)).to(
    device
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(vae.parameters())

# Training loop
for epoch in range(num_epochs):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        metrics = vae.training_step(imgs, optimizer)
        print(
            f"Epoch {epoch}, Loss: {metrics['loss']}, Recon Loss: {metrics['reconstruction_loss']}, KL Loss: {metrics['kl_loss']}"
        )

    # Save checkpoints every 20 epochs
    if (epoch + 1) % 20 == 0:
        save_checkpoint(vae.encoder, vae.decoder, epoch + 1)
