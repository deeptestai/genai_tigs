import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import numpy as np
import torch
from torch import autocast
from PIL import Image
from torchvision import transforms


def disableNSFWFilter(pipe):
    """Disables the trigger happy nsfw filter. tread carefully"""
    def dummy(images, **kwargs):
        return images, [False] * len(images)
    pipe.safety_checker = dummy

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
# Load pretrained model
classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
classifier.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other necessary transformations here
])
prompt = "A photo of 1pizza" # prompt to dream about
seed = 7852
proj_name = "test"
num_inference_steps = 50
width = 512
height = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_perturbation = 0.5
best_left = 10
frame_index = 0
gen_steps = 500
fitness_scores = []
images1 = []
num_samples = 2
torch.manual_seed(seed)
proj_path = "./evolution/"+proj_name+"_"+str(seed)
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/Newresultjump', exist_ok=True)

print('Creating init image')
lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "/home/maryam/Documents/SEDL/SINVAD/img_sd/fine_tune_imgnet-000005.safetensors"
pipe = StableDiffusionPipeline.from_pretrained(
base_model_id, scheduler=lms, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
disableNSFWFilter(pipe)
for i in range(num_samples):
    start = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=device)
    with autocast("cuda"):
         init_img = pipe(prompt, num_inference_steps=50, latents=start, width=width, height=height)["images"][0]
         init_img_path = os.path.join(proj_path, f'_origin_{i}.png')
         init_img.save(init_img_path)
         print(f"Original image {i} saved at {init_img_path}")
    # Assuming 'image' is your PIL Image or similar
    tensor_image = transform(init_img)
    tensor_image = tensor_image.unsqueeze(0)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    original_latent = start
    prev_best = np.inf
    for i in range(gen_steps):
        perturb_latents = original_latent + init_perturbation * torch.randn((4, height // 8, width // 8), device=device)
        print(perturb_latents.shape)
        with autocast("cuda"):
            perturb_img = pipe([prompt],
                num_inference_steps=num_inference_steps,
                latents=perturb_latents,
            )["images"]
        if isinstance(perturb_img, list) and perturb_img:
            last_image = perturb_img[0]
        tensor_image2 = transform(last_image)
        tensor_image2 = tensor_image2.unsqueeze(0)
        perturb_logits = classifier(tensor_image2).squeeze().detach().cpu().numpy()
        perturb_label = np.argmax(perturb_logits).item()
        fitness = calculate_fitness(perturb_logits, original_label)
        fitness_scores.append(fitness)
        # Perform selection
        selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
            )[-best_left:]
        now_best = np.min(fitness_scores)
        print("now_best " + str(now_best) + " average_best " + str(np.mean(fitness_scores)))
        if now_best < 0:
           break
        elif now_best == prev_best:
             init_perturbation *= 2
        else:
            init_perturbation = init_perturbation

        original_latent = perturb_latents
        prev_best = now_best

    # After the loop, save the last image if it exists
    if last_image is not None:
        last_image_path = '{}/Newresultjump/last_image.png'.format(proj_path)
        last_image.save(last_image_path)
        print(f"Last image saved at {last_image_path}")
    else:
        print("Error: No image was generated")

