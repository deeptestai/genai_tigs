import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from tgate import TgateSDDeepCacheLoader
import re
import random
import numpy as np
import torch
import wandb
from PIL import Image
from torchvision import transforms


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
generator = torch.Generator(device='cuda')
torch.use_deterministic_algorithms(True)

# Define the classifier
classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).to(device)
classifier.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define prompt and paths
prompt = "A photo of 1toy_bear teddy_bear"
proj_name = "test"
proj_path = "./imagenet_sd_teddy_rq1/" + proj_name + "_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path + '/perturbresult', exist_ok=True)

# Define Stable Diffusion parameters
num_inference_steps = 25
width = 512
height = 512

# Load Stable Diffusion model
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./teddy_bear_imagenet-000005.safetensors"
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id, variant="fp16", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)
pipe = TgateSDDeepCacheLoader(pipe, cache_interval=3, cache_branch_id=0).to(device)

# Set seed and initialize counters
seed = 0
correct_matches = 0
imgs_to_samp = 100
expected_label = 850  # Index for the desired label 'teddy_bear'

# Generate and classify images
for n in range(imgs_to_samp):
    seedSelect = seed + n
    generator = generator.manual_seed(seedSelect)
    original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device).to(torch.float16)

    with torch.inference_mode():
         init_img = pipe.tgate(prompt,guidance_scale=3.5, num_inference_steps=num_inference_steps,gate_steps=10, latents=original_lv)["images"][0]
         

    tensor_image = transform(init_img)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    predicted_label = np.argmax(original_logit).item()
    
    init_img_path = os.path.join(proj_path, f'image_{n}_prompt{prompt}_X{expected_label}_Y{predicted_label}.png')
    init_img.save(init_img_path)
    print(f"Image {n} saved at {init_img_path}")
    # Save the same image as a NumPy array
    npy_path = os.path.join(proj_path, f'image_{n}_prompt{prompt}_X{expected_label}_Y{predicted_label}.npy')
    np_image = tensor_image.squeeze().cpu().numpy()  # Convert tensor to numpy
    np.save(npy_path, np_image)    
    if predicted_label == expected_label:
        correct_matches += 1

# Calculate and print accuracy
accuracy = correct_matches / imgs_to_samp
print(f"Accuracy of classifier with prompt and generated images: {accuracy * 100:.2f}%")
