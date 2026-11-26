import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os
import pandas as pd
from cdcgan.cdcgan_cifar10 import Generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32 * 32 * 1


# Load the trained models
G = Generator(ngpu=1, nz=100, nc=3).to(device)  # Modify the Generator architecture

# Load the weights of the models
G.load_state_dict(
    torch.load(
        "./cdcgan/weights/netG_epoch_699.pth",
        map_location=device,
    )
)                         # load different genartors according to dataset to comute its perspective ranges

# Set models to evaluation mode
G.eval()
num_samples = 1000
latent_space = torch.randn(num_samples, 100, 1, 1).to(device)
random_labels = torch.randint(0, 10, (num_samples,)).to(device)

results_file = "./cdcgan_muex.txt"  # File to save the min and max values

# Initialize list to store min and max values for each sample
sample_ranges = []

# Loop over each sample
with open(results_file, "w") as file:
    file.write("Sample Index, Min Value, Max Value\n")  # Header for the results file

    for i in range(num_samples):
        original_latent = latent_space[i]  # Get the i-th latent vector
        # Compute min and max values of the latent vector
        min_val = torch.min(original_latent).item()
        max_val = torch.max(original_latent).item()

        # Write the min and max values of the latent vector to the file
        file.write(f"{i}, {min_val}, {max_val}\n")

        # Append the range to the list for further computations
        sample_ranges.append((min_val, max_val))

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

# Save the results to an Excel file using pandas
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

# Save to an Excel file
output_file = "gan_mutex.xlsx"
df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
