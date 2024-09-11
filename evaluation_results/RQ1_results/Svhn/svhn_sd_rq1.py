import os
import re
import random
from diffusers import StableDiffusionPipeline
from tgate import TgateSDDeepCacheLoader
from diffusers.schedulers import DPMSolverMultistepScheduler
import numpy as np
import torch
import wandb
from PIL import Image
from torchvision import transforms
from svhn_classifier.model import VGGNet

#run = wandb.init(project="sinvadsvhnfitness")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
generator = torch.Generator(device='cuda')
torch.use_deterministic_algorithms(True)

classifier = VGGNet().to(device)
classifier.load_state_dict(
    torch.load(
        "./svhn_classifier/model/SVHN_vggnet.pth",
        map_location=device,
    )
)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

prompts = [
    "A photo of HouseNo0 Hnumber0", "A photo of HouseNo1 Hnumber1", "A photo of HouseNo2 Hnumber2",
    "A photo of HouseNo3 Hnumber3", "A photo of HouseNo4 Hnumber4", "A photo of HouseNo5 Hnumber5",
    "A photo of HouseNo6 Hnumber6", "A photo of HouseNo7 Hnumber7", "A photo of HouseNo8 Hnumber8",
    "A photo of HouseNo9 Hnumber9"
]

proj_name = "test"
num_inference_steps = 25
width = 512
height = 512

gen_steps = 250
pop_size = 25
fitness_scores = []
all_img_lst = []
num_samples = 100
proj_path = "./svhn_sd_rq1/" + proj_name + "_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path + '/Newresult', exist_ok=True)
os.makedirs(os.path.join(proj_path, 'generated_images'), exist_ok=True)

base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./svhn_finetune_lorav1.5-000005.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id, variant="fp16", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)
pipe = TgateSDDeepCacheLoader(
    pipe,
    cache_interval=3,
    cache_branch_id=0,
).to(device)

seed = 0
correct_matches = 0

for img_idx in range(imgs_to_samp):
    seedSelect = seed + n
    generator = generator.manual_seed(seedSelect)
    original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device).to(torch.float16)
    randprompt = random.choice(prompts)
    expected_label = int(re.search(r"HouseNo(\d+)", randprompt).group(1))
    
    with torch.inference_mode():
        init_img = pipe.tgate(prompt=randprompt, guidance_scale = 3.5, gate_step=10, num_inference_steps=num_inference_steps,latents = original_lv )["images"][0]
    
    tensor_image = transform(init_img)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    predicted_label = np.argmax(original_logit).item()
    
    init_img_path = os.path.join(proj_path, f'image_{img_idx}_prompt_{randprompt}_X{expected_label}_Y_{predicted_label}.png')
    init_img.save(init_img_path)
    print(f"Image {n} saved at {init_img_path}")
    # Convert the tensor image to numpy and save as .npy
    npy_path = os.path.join(proj_path, f'image_{img_idx}_prompt_{randprompt}_X{expected_label}_Y_{predicted_label}.npy')
    np_image = tensor_image.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy
    np.save(npy_path, np_image)
    
    
    if predicted_label == expected_label:
        correct_matches += 1

accuracy = correct_matches / imgs_to_samp
print(f"Accuracy of classifier with prompt and generated images: {accuracy * 100:.2f}%")
