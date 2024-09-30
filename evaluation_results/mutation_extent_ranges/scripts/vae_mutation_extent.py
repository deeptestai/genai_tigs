import torch
import os
import numpy as np
from PIL import Image
import torchvision
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from vae_model import ConvVAE
import pandas as pd
#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 32 * 32 * 3

# Load the trained models
vae = ConvVAE(c_num=3, h_dim=4000, z_dim=1024).to(device)
vae.load_state_dict(
    torch.load(
        "./weights/cifar10_convend.pth",
        map_location=device,
    )
)                                               # load vae weights according to different datasets
# Set models to evaluation mode
vae.eval()

# Transforms: Convert image to tensor and normalize it
test_dataset = torchvision.datasets.CIFAR10(root='./datat', train=False, transform=transforms.ToTensor(), download=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)    # download different dataset mnist, cifar10 or svhn according to findout ranges

print("Data loader ready...")

# Set parameters
num_samples = 1000  # Define how many samples you'd like to process
result_dir = "./result_cifar10smplrng_vae"
os.makedirs(result_dir, exist_ok=True)
min_max_filename = os.path.join(result_dir, "encoded_samples_min_max_cifar10.csv")

# Load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=True
)
print("Data loader ready...")

# Initialize a list to store min and max values for each encoded sample
encoded_samples_min_max = []
sample_ranges = []  # For storing min/max values for further computations

# Corrected loop to process exactly num_samples samples
for i, (x, x_class) in tqdm(enumerate(test_data_loader), total=num_samples):
    if i >= num_samples:
        break  # Stop after processing num_samples
    samp_img = x.to(device)  # Move image to device

    # Encode the image using VAE
    img_enc, _ = vae.encode(samp_img.view(-1, 3, 32, 32))  # Adjust view as per VAE input size
    original_lv = img_enc.detach().cpu().numpy()  # Detach and move to CPU

    # Calculate min and max for the encoded latent vector
    sample_min = np.min(original_lv)
    sample_max = np.max(original_lv)

    # Append the min and max values to the list for storing in the CSV
    encoded_samples_min_max.append([sample_min, sample_max])

    # Append min and max to sample_ranges for further analysis
    sample_ranges.append((sample_min, sample_max))

# Compute the min of all min values and max of all max values
min_of_mins = min([min_val for min_val, _ in sample_ranges])
max_of_maxs = max([max_val for _, max_val in sample_ranges])

# Compute the difference (max - min)
diff = max_of_maxs - min_of_mins

# Low perturbation calculations (diff / 10000)
low_init_perturbation = diff / 10000
low_perturbation_size = 2 * low_init_perturbation

# High perturbation calculations (diff / 1000)
high_init_perturbation = diff / 1000
high_perturbation_size = 2 * high_init_perturbation

# Prepare results for saving in Excel and CSV
results = {
    "Sample Index": list(range(len(sample_ranges))),
    "Min Value": [min_val for min_val, _ in sample_ranges],
    "Max Value": [max_val for _, max_val in sample_ranges],
}

# Create a DataFrame and save the ranges
df = pd.DataFrame(results)

# Add computed values for low and high perturbation
df.loc["Summary", "Min Value"] = min_of_mins
df.loc["Summary", "Max Value"] = max_of_maxs
df.loc["Summary", "Difference (Max - Min)"] = diff

# Low perturbation
df.loc["Low Perturbation", "Init Perturbation"] = low_init_perturbation
df.loc["Low Perturbation", "Perturbation Size"] = low_perturbation_size

# High perturbation
df.loc["High Perturbation", "Init Perturbation"] = high_init_perturbation
df.loc["High Perturbation", "Perturbation Size"] = high_perturbation_size

# Save the DataFrame to an Excel file
output_excel_file = os.path.join(result_dir, "cifar10_gan_muex.xlsx")
df.to_excel(output_excel_file, index=False)

# Save the DataFrame to a CSV file
output_csv_file = os.path.join(result_dir, "cifar10_vae_muex.csv")
df.to_csv(output_csv_file, index=False)

print(f"Results saved to {output_excel_file} and {output_csv_file}")

