import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
import numpy as np
import torch
import wandb
import pandas as pd
from torch import autocast
from PIL import Image
from torchvision import transforms
from collections import Counter
#run =wandb.init(project="sinvadtestfitness")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
generator = torch.Generator(device = 'cuda')
torch.use_deterministic_algorithms(True)

prompt = "A photo of truck9" # prompt to dream about
proj_name = "test"
num_inference_steps = 25
width = 512
height = 512
sample_ranges = []
num_samples = 10
proj_path = "./cifar10_sd_range/"+proj_name+"_"
os.makedirs(proj_path, exist_ok=True)

print('Creating init image')
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./img_sd/cifar10_finetune_lorav1.5-000005.safetensors"   //change checkpoints and compute ranges for other dataset models

pipe = StableDiffusionPipeline.from_pretrained(
base_model_id, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)
seed1 =1024
num_samples1 = 1000
for _ in range(num_samples1):
    generator = generator.manual_seed(seed1)
    # Generate a random latent vector for each sample
    original_lv1 = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), device=device)
    
  
    min_val = torch.min(original_lv1).item()
    max_val = torch.max(original_lv1).item()
    
    # Append the range to the list
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
output_file = "sample_ranges_sdcifar_perturbations.xlsx"
df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
