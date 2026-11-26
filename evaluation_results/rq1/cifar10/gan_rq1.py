import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os
from cifar10_VGGNETclassifier.model import VGGNet
from cdcgan.cdcgan_cifar10 import Generator
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32 * 32 * 1
num_samples = 100  # Number of samples to generate and test

# Load the trained models
G = Generator(ngpu=1, nz=100, nc=3).to(device)  # Modify the Generator architecture
classifier = VGGNet().to(device)

# Load the weights of the models
G.load_state_dict(
    torch.load(
        "./cdcgan/weights/netG_epoch_699.pth",
        map_location=device,
    )
)
classifier.load_state_dict(
    torch.load(
        "./cifar10_VGGNETclassifier/CIFAR10_cifar10_train.pynet.pth",
        map_location=device,
    )
)

# Set models to evaluation mode
G.eval()
classifier.eval()

# Directory to save the images
result_dir = "./cifar10_cdcgan_rq1"
original_images_dir = os.path.join(result_dir, 'original_images')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "or_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(prtb_img_dir, exist_ok=True)

latent_space = torch.randn(num_samples, 100, 1, 1).to(device)
random_labels = torch.randint(0, 10, (num_samples,)).to(device)

import torch
import torchvision.utils as vutils
from tqdm import trange
#import wandb
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from cifar10_VGGNETclassifier.model import VGGNet
from cdcgan.cdcgan_cifar10 import Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#run = wandb.init(project="Sinvad_fitness_CDCGAN_Cifar10")
# Set device

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



img_size = 32 * 32 * 1
# Load the trained models
G = Generator(ngpu=1, nz=100, nc=3).to(device)  # Modify the Generator architecture
classifier = VGGNet().to(device)

# Load the weights of the models
G.load_state_dict(
    torch.load(
        "./cdcgan/weights/netG_epoch_699.pth",
        map_location=device,
    )
)
classifier.load_state_dict(
    torch.load(
        "./cifar10_VGGNETclassifier/CIFAR10_cifar10_train.pynet.pth",
        map_location=device,
    )
)
# Set models to evaluation mode
G.eval()
classifier.eval()
result_dir = "./result23_cdcgan_cifar10" # Directory to save the images
original_images_dir = os.path.join(result_dir, 'original_images')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "or_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(prtb_img_dir, exist_ok=True)

def calculate_fitness(logit, label):
    expected_logit = logit[label]
    # Select the two best indices [perturbed logit]
    best_indices = np.argsort(-logit)[:2]
    best_index1, best_index2 = best_indices
    if best_index1 == label:
        best_but_not_expected = best_index2
    else:
        best_but_not_expected = best_index1
    new_logit = logit[best_but_not_expected]
    # Compute fitness value
    fitness = expected_logit - new_logit
    return fitness


image_info = []
predictions = []
imgs_to_samp = 100
# mutation_perturbation_size = 0.1  # Mutation perturbation size
# Generate a random latent vector
latent_space = torch.randn(imgs_to_samp, 100, 1, 1).to(device)
random_labels = torch.randint(0, 10, (imgs_to_samp,)).to(device)
all_img_lst = []
# Initialize counters for accuracy calculation
correct_predictions = 0
total_images = 0


# Iterate through the images
for img_idx in trange(imgs_to_samp):
    # Generate the non-perturbed image
    original_latent = latent_space[img_idx].unsqueeze(0)
    original_labelr = random_labels[img_idx]
    expected_label = original_labelr.item()

    # Generate the non-perturbed image using the generator
    original_image = G(original_latent, original_labelr.view(1))

    # Classify the generated image
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()

    # Increment total_images
    total_images += 1

    # Compare the predicted label with the expected label
    if original_label == expected_label:
        correct_predictions += 1  # Count this as a correct prediction
        print(f"Match found: True label {original_labelr.item()}, Predicted label {original_label}")

        # Save the original image
        original_image_path = os.path.join(img_pil_dir, f'original_image{img_idx}_X{original_label}.png')
        save_image(original_image, original_image_path)

        # Convert the tensor to a NumPy array
        original_image_np = original_image.squeeze().detach().cpu().numpy()

        # Define the filename and path for saving the NumPy file
        filename = f"original_image_{img_idx}_X_{original_label}.npy"
        file_path = os.path.join(original_images_dir, filename)
        np.save(file_path, original_image_np)



# Calculate accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"Model accuracy based on {total_images} images: {accuracy:.2f}%")
