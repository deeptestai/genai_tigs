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
from cifar10_classifier.model import VGGNet


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
generator = torch.Generator(device='cuda')
torch.use_deterministic_algorithms(True)

# Define the classifier
classifier = VGGNet().to(device)
classifier.load_state_dict(
    torch.load(
        "./cifar10_classifier/CIFAR10_cifar10_train.pynet.pth",
        map_location=device,
    )
)
classifier.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Define prompts
prompts = [
    "A photo of A1plane0 cifar10_0", "A photo of car1 cifar10_1", "A photo of bird2 cifar10_2",
    "A photo of cat3 cifar10_3", "A photo of deer4 cifar10_4", "A photo of dog5 cifar10_5",
    "A photo of frog6 cifar10_6", "A photo of horse7 cifar10_7", "A photo of ship8 cifar10_8",
    "A photo of truck9 cifar10_9"
]

# Define paths and create directories
proj_name = "test"
proj_path = "./cifar10_sd_rq1gs3.5/" + proj_name + "_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path + '/perturbresult', exist_ok=True)

# Define Stable Diffusion parameters
num_inference_steps = 25
width = 512
height = 512

# Load Stable Diffusion model
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./cifar10_finetune_lorav1.5-000005.safetensors"
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

# Generate and classify images
for n in range(imgs_to_samp):
    seedSelect = seed + n
    generator = generator.manual_seed(seedSelect)
    original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device).to(torch.float16)
    
    randprompt = random.choice(prompts)
    expected_label = int(re.search(r"cifar10_(\d+)", randprompt).group(1))
    
    with torch.inference_mode():
        init_img = pipe.tgate(prompt=randprompt,guidance_scale = 3.5,gate_step=10, num_inference_steps=num_inference_steps, latents = original_lv)["images"][0]
    
    tensor_image = transform(init_img)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    predicted_label = np.argmax(original_logit).item()
    
    init_img_path = os.path.join(proj_path, f'image_{n}_prompt_{randprompt}_X{expected_label}_Y_{predicted_label}.png')
    init_img.save(init_img_path)
    print(f"Image {n} saved at {init_img_path}")
    # Convert the tensor image to numpy and save as .npy
    npy_path = os.path.join(proj_path, f'image_{n}_prompt_{randprompt}_X{expected_label}_Y_{predicted_label}.npy')
    np_image = tensor_image.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy
    np.save(npy_path, np_image)
    
    if predicted_label == expected_label:
        correct_matches += 1

# Calculate and print accuracy
accuracy = correct_matches / num_samples
print(f"Accuracy of classifier with prompt and generated images: {accuracy * 100:.2f}%")
