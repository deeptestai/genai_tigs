import torch
import os
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
from vae.svhn_vae.svhn_vae.model import ConvVAE
from unconditional_gan.classifier.model import VGGNet
import time
def save_image(tensor, filename):
    """
    Save a tensor as an image
    """
    img = vutils.make_grid(tensor, normalize=True)  # Normalize ensures [0,1] range
    img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    # Now, transfer the tensor to CPU and convert it to numpy array for saving
    img = img.to('cpu', torch.uint8).numpy()  # Convert tensor to numpy array in [0,255] range
    img = Image.fromarray(img)
    img.save(filename)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32 * 32 * 3

# Load the trained models
vae = ConvVAE(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800).to(device)
classifier = VGGNet().to(device)
vae.load_state_dict(
    torch.load(
        "./vae/svhn_vae/svhn_vae/svhn_convend.pth",
        map_location=device,
    )
)
classifier.load_state_dict(
    torch.load(
        "./unconditional_gan/classifier/SVHN_vggnet.pth",
        map_location=device,
    )
)
# Set models to evaluation mode
vae.eval()
classifier.eval()

# Directory to save the images
result_dir = "./svhn_vae_rq1"
# Subdirectory for original images
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "original_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(prtb_img_dir, exist_ok=True)

test_dataset = torchvision.datasets.SVHN(root='./data', split="test", transform=transforms.ToTensor(), download=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

print("Data loader ready...")


# Initialize variables
correct_predictions = 0
total_images = 0
imgs_to_samp = 100
# Iterate over images
for img_idx in trange(imgs_to_samp):  # Process 100 images
    for i, (x, x_class) in enumerate(test_data_loader):
        samp_img = x[0:1]
        samp_class = x_class[0].item()

    # Pass the image through VAE
    img_enc, _ = vae.encode(samp_img.to(device))
    original_lv = img_enc
    original_image = vae.decode(original_lv).view(-1, 3, 32, 32)

    # Classify the reconstructed image
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()

    # Increment total images count
    total_images += 1

    # If the predicted label matches the true label, count it as a correct prediction
    if original_label == samp_class:
        correct_predictions += 1

        # Save the image data as a NumPy file
        original_image_np = original_image.squeeze().detach().cpu().numpy()
        filename = f"original_image_{img_idx}_X_{original_label}.npy"
        file_path = os.path.join(original_images_dir, filename)
        np.save(file_path, original_image_np)


# Calculate accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"Model accuracy based on {total_images} images: {accuracy:.2f}%")
