import torch

# import wandb
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import trange
from torchvision import datasets
import torchvision
import os

from sa.model import MnistClassifier
from vae.model import VAE
import time
# run = wandb.init(project="sinvad_fitness_sd_mnist")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28 * 28 * 1
torch.no_grad()  # since nothing is trained here

### Prep (e.g. Load Models) ###
vae = VAE(img_size=28 * 28, h_dim=1600, z_dim=400).to(device)
classifier = MnistClassifier(img_size=img_size).to(device)
vae.load_state_dict(torch.load("./vae/models/MNIST_EnD.pth", map_location=device,))
vae.eval()
classifier.load_state_dict(
    torch.load(
        "./sa/models/MNIST_conv_classifier.pth",
        map_location=device,
    )
)
classifier.eval()
result_dir = "./mnist_vae_rq1"  # Directory to save the images
# Subdirectory for original images
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
image_pil_dir = os.path.join(original_images_dir,'images')
os.makedirs(image_pil_dir, exist_ok=True)

print("models loaded...")
# Transforms: Convert image to tensor and normalize it
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=True
)
imgs_to_samp = 100
correct_predictions = 0
total_images = 0

for img_idx in trange(imgs_to_samp):
    for i, (x, x_class) in enumerate(test_data_loader):
        samp_img = x[0:1]
        samp_class = x_class[0].item()

    img_enc, _ = vae.encode(samp_img.view(-1, img_size).to(device))
    original_lv = img_enc
    original_image = vae.decode(original_lv).view(-1, 1, 28, 28)
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()

    # Increment total_images counter
    total_images += 1

    # Check if the predicted label matches the true label
    if original_label == samp_class:
        correct_predictions += 1

    # Save the image and numpy array as before
    if original_label == samp_class:
        original_image_pil = transforms.ToPILImage()(original_image.squeeze().cpu())
        original_image_path = os.path.join(
            original_images_dir, f"original_image_{img_idx}_X{original_label}.png"
        )
        original_image_pil.save(original_image_path)
        
        original_image_np = original_image.squeeze().detach().cpu().numpy()
        filename = f"original_image_{img_idx}_X_{original_label}.npy"
        file_path = os.path.join(original_images_dir, filename)
        np.save(file_path, original_image_np)


# Calculate accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"Model accuracy based on {total_images} images: {accuracy:.2f}%")
