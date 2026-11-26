import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
from vae_model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the VAE
vae = Model().to(device)
# Load the checkpoint
checkpoint = torch.load(
    "./imgnt_vae/Imgnt_tfvae_ptrcvae/checkpoints/imagenet_vae_model_epoch_41.pth",
    map_location="cpu",  # Or 'cpu' if running on a CPU-only environment
)

# Correctly load the state dictionary
vae.load_state_dict(checkpoint["model_state_dict"])
vae.eval()
vae.eval()

# Load the classifier
classifier = models.vgg19_bn(pretrained=True).to(device)
classifier.eval()
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((128, 128)),
    ]
)

dataset = datasets.ImageFolder(
    "/home/maryam/Documents/SEDL/SINVAD/experiments_complete/imagenet_experiments/imgnt_vae/Imgnt_tfvae_ptrcvae/test1_pizza",
    transform=transform,
)
data_loader = DataLoader(
    dataset, batch_size=1, shuffle=False
)  # Batch size set to 1 for individual processing

# Function to denormalize images
def denormalize(image):
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return image.clamp(0, 1)


# Process images
output_folder = "reconstructed_images"
os.makedirs(output_folder, exist_ok=True)
# Known label for all images
total = 0
correct = 0
expected_label = 963  # Assuming all images have this label / for teddy_class change into 850

with torch.no_grad():
    for idx, (images, _) in enumerate(data_loader):
        if idx >= 100:  # Process only 100 images
            break

        images = images.to(device)

        # Encode and Decode Images
        mean, logvar = vae.encode(images)
        z = vae.reparamatrize(mean, torch.exp(0.5 * logvar))
        reconstructed_images = vae.decode(z)

        # Classify reconstructed images
        outputs = classifier(reconstructed_images)
        _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        total += images.size(0)
        correct += (predicted == expected_label).sum().item()

        # Save reconstructed images with predicted labels
        for i in range(images.size(0)):
            img = reconstructed_images[i]
            img = denormalize(img)
            img_pil = transforms.ToPILImage()(img.cpu())
            img_pil.save(
                os.path.join(
                    output_folder, f"image_{idx}_label_{predicted[i].item()}.png"
                )
            )
            np.save(
                os.path.join(
                    output_folder, f"image_{idx}_label_{predicted[i].item()}.npy"
                ),
                img.cpu().numpy(),
            )

# Print accuracy
accuracy = 100 * correct / total
print(f"Classification accuracy of reconstructed images: {accuracy:.2f}%")
