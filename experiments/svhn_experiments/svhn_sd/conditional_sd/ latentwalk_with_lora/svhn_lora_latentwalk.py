import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import numpy as np
import torch
from torch import autocast
from PIL import Image
from torchvision import transforms
from svhn_classifier.model import VGGNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier = VGGNet().to(device)
# Load pretrained model
classifier.load_state_dict(
    torch.load(
        "./svhn_classifier/model/SVHN_vggnet.pth",
        map_location=device,
   )
)
classifier.eval()
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # Converts the image to a PyTorch tensor
   # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
])

prompt = "A photo of HouseNo9 Hnumber9" # prompt to dream about
proj_name = "test"
num_inference_steps = 25
width = 512
height = 512
walk_steps = 150
batch_size = 3
batches = walk_steps // batch_size
step_size = 0.005
all_img_lst = []
num_samples = 10
proj_path = "./svhn_sdlike_evolution/"+proj_name+"_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/Newresultjump', exist_ok=True)
print('Creating init image')
lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./svhn_finetune_lorav1.5-000005.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
base_model_id, scheduler=lms, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
#pipe = StableDiffusionPipeline.from_pretrained(weights_path, scheduler=lms, use_auth_token=False)
pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)
for iteration in range(num_samples):
    print(f'Creating init image for sample {iteration}')

    original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), device=device)

    with autocast("cuda"):
        init_img = pipe(prompt, num_inference_steps=25, latents=original_lv, width=width, height=height)["images"][0]
    tensor_image = transform(init_img)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    init_img_path = os.path.join(proj_path, f'original_{iteration}_X{original_label}.png')
    init_img.save(init_img_path)
    delta = torch.randn_like(original_lv) * step_size  # Perturbation

    walked_latent = []

    for step_index in range(walk_steps):
        walked_latent.append(original_lv)
        original_lv += delta

    walked_latent_length = len(walked_latent)
    print("Length of walked_latent:", walked_latent_length)
    walked_encodings = torch.stack(walked_latent)

    for step in range(0, walk_steps, batch_size):
        # Create a batch of latent encodings for this step
        batched_encodings = walked_encodings[step:step + batch_size]
        with torch.no_grad():
            with autocast("cuda"):
                # Decode the latent vectors into images
                walk_output = pipe(prompt, num_inference_steps=25, latent=batched_encodings)['images']

    # Save the last generated image with its perturb label
    last_generated_image = walk_output[-1]
    last_generated_perturb_label = np.argmax(classifier(transform(last_generated_image).unsqueeze(0).to(device)).detach().cpu().numpy()).item()

    # Save the last generated image
    last_generated_img_pil = Image.fromarray(np.uint8(last_generated_image))
    last_generated_img_path = os.path.join(proj_path,'Newresultjump', f'img_{iteration}_X{original_label}_Y{last_generated_perturb_label}.png')
    last_generated_img_pil.save(last_generated_img_path)
    print(f"Last generated image for sample {iteration} saved at {last_generated_img_path} with perturb label {last_generated_perturb_label}")
