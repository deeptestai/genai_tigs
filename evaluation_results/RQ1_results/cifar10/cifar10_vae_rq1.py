import torch
import os
import numpy as np
from PIL import Image
import torchvision
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import trange
from vae_model import ConvVAE
from cifar10_classifier.model import VGGNet
import time
def save_image(tensor, filename):
    """
    Save a tensor as an image.
    Assumes tensor is normalized in the format [channels, height, width].
    """
    img = vutils.make_grid(tensor, normalize=True)  # Normalize ensures [0,1] range
    img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    # Now, transfer the tensor to CPU and convert it to numpy array for saving
    img = img.to('cpu', torch.uint8).numpy()  # Convert tensor to numpy array in [0,255] range
    img = Image.fromarray(img)
    img.save(filename)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 32 * 32 * 3

# Load the trained models
vae = ConvVAE(c_num=3, h_dim=4000, z_dim=1024).to(device)
classifier = VGGNet().to(device)
vae.load_state_dict(
    torch.load(
        "/home/maryam/Documents/SEDL/SINVAD/experiments/cifar10_experiments/cifar10_vae/weights/cifar10_convend.pth",
        map_location=device,
    )
)
classifier.load_state_dict(
    torch.load(
        "./cifar10_classifier/CIFAR10_cifar10_train.pynet.pth",
        map_location=device,
    )
)
# Set models to evaluation mode
vae.eval()
classifier.eval()

# Directory to save the images
result_dir = "./cifar10_vae_rq1"
# Subdirectory for original images
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "or_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(prtb_img_dir, exist_ok=True)

# Transforms: Convert image to tensor and normalize it
test_dataset = torchvision.datasets.CIFAR10(root='./datat', train=False, transform=transforms.ToTensor(), download=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

print("Data loader ready...")

# Initialize counters for accuracy calculation
correct_predictions = 0
total_images = 0
imgs_to_samp = 100
### Multi-image sample loop ###
for img_idx in trange(imgs_to_samp):  # Loop runs for 100 iterations
    # Load the image and class inside the loop
    for i, (x, x_class) in enumerate(test_data_loader):
        samp_img = x[0:1]  # Load one sample image
        samp_class = x_class[0].item()  # Get the class label

    # Perform VAE encoding and decoding
    img_enc, _ = vae.encode(samp_img.to(device))
    original_lv = img_enc
    original_image = vae.decode(original_lv).view(-1, 3, 32, 32)

    # Predict the class for the original image using the classifier
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()

    # Compare the predicted label with the true label
    total_images += 1  # Increment total image count
    if original_label == samp_class:
        correct_predictions += 1  # Increment correct prediction count

        print(f"True class: {samp_class}, Predicted class: {original_label}")

        # Save the original image
        original_image_np = original_image.squeeze().detach().cpu().numpy()
        filename = f"original_image_{img_idx}_X_{original_label}.npy"
        file_path = os.path.join(original_images_dir, filename)
        np.save(file_path, original_image_np)


# Calculate accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"Model accuracy based on {total_images} images: {accuracy:.2f}%")
