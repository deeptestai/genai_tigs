import os
import torch
from diffusers import DDPMScheduler, DDPMPipeline
from torchvision import transforms
from PIL import Image
from torch import autocast
from cifar10_classifier.model import VGGNet
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load classifier model
classifier = VGGNet().to(device)
classifier.load_state_dict(torch.load("./cifar10_classifier/CIFAR10_cifar10_train.pynet.pth", map_location=device))
classifier.eval()

# Set transform
transform = transforms.Compose([transforms.ToTensor()])

# Configuration
num_inference_steps = 20
num_samples = 100
batch_size = 6
walk_steps = 60
proj_path = "./evolution_ddpmlatentwalk_cifar10/test_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path + '/Newresultjump', exist_ok=True)

# Initialize the diffusion model
scheduler = DDPMScheduler.from_pretrained("Maryamm/ddpm_finetune_cifar10")
model = DDPMPipeline.from_pretrained("Maryamm/ddpm_finetune_cifar10").to(device)
scheduler.set_timesteps(1000)
generator = torch.Generator(device=device).manual_seed(0)

# Generate images
for n in range(num_samples):
    # Generate initial latent vector
    original_lv = torch.randn((1, 3, 32, 32), device=device)
    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = model.unet(original_lv, t).sample
            prev_noisy_sample = scheduler.step(noisy_residual, t, original_lv).prev_sample
            original_lv = prev_noisy_sample

    # Process and save initial image
    image = (original_lv / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    init_img = Image.fromarray(image)
    init_img_path = os.path.join(proj_path, f'_origin_{n}.png')
    init_img.save(init_img_path)
    print(f"Original image {n} saved at {init_img_path}")

    # Classify initial image
    tensor_image = transform(init_img).unsqueeze(0).to(device)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    walked_latent = []
    # Prepare for latent walk
    delta = torch.randn_like(original_lv) * 0.001  # Perturbation
    for step_index in range(walk_steps):
        walked_latent.append(original_lv)
        original_lv += delta
    walked_encodings = torch.stack(walked_latent)

    # Walk through latent space and generate images
    all_perturb_imgs = []
    for step in range(0, walk_steps, batch_size):
        batched_encodings = walked_encodings[step:step + batch_size]
        # Remove extra dimension if necessary
        batched_encodings = batched_encodings.squeeze(1)
        for t in scheduler.timesteps:
            with torch.no_grad():
             with autocast("cuda"):
                noisy_residual = model.unet(batched_encodings, t).sample
                prev_noisy_sample = scheduler.step(noisy_residual, t, batched_encodings).prev_sample
                batched_encodings = prev_noisy_sample

        # Process images from last step
        images = (batched_encodings / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        for img_array in images:
            img_array = (img_array * 255).round().astype("uint8")
            perturb_img = Image.fromarray(img_array)
            all_perturb_imgs.append(perturb_img)

    # Classify last image and save
    tensor_image2 = transform(all_perturb_imgs[-1]).unsqueeze(0).to(device)
    all_logits = classifier(tensor_image2).detach().cpu().numpy()
    perturb_label = np.argmax(all_logits).item()
    image_filename = f'image_{n}_X{original_label}_Y{perturb_label}.png'
    last_image_path = os.path.join(proj_path, 'Newresultjump', image_filename)
    all_perturb_imgs[-1].save(last_image_path)
    print(f"Last image saved at {last_image_path}")
