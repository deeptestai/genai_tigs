import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from tgate import TgateSDDeepCacheLoader
import re
import random
import numpy as np
import torch
import wandb
from torch import autocast
from PIL import Image
from torchvision import transforms
from collections import Counter
#from cifar10_classifier.model import VGGNet
run =wandb.init(project="sinvadtestfitness2")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
generator = torch.Generator(device = 'cuda')
torch.use_deterministic_algorithms(True)
import torch

def scale_and_unnormalize_tensor(tensor):
    """
    Scale and unnormalize a tensor to the range [0, 1] based on its current min and max values.

    Parameters:
    - tensor (torch.Tensor): The tensor to be scaled and unnormalized.

    Returns:
    - torch.Tensor: The scaled and unnormalized tensor.
    """
    if tensor is None:
        raise ValueError("No tensor provided")

    # Find the current minimum and maximum values of the tensor
    min_val = tensor.min()
    max_val = tensor.max()

    # Scale the tensor to [0, 1]
    scaled_tensor = (tensor - min_val) / (max_val - min_val)

    return scaled_tensor

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

#classifier = VGGNet().to(device)
# Load pretrained model
classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).to(device)
classifier.eval()
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
prompt = "A photo of 1pizza pizza_slice" # prompt to dream about
proj_name = "test"
num_inference_steps = 25
width = 512
height = 512
min_val =-5.54950523376465
max_val =5.33688974380493
init_perturbation = 0.0217727899551392
perturbation_size = 0.0108863949775696
best_left = 10
gen_steps = 250
pop_size = 25
fitness_scores = []
all_img_lst = []
num_samples = 100
image_info = []
predicted_labels = []
proj_path = "./imagenet_sd_result1/"+proj_name+"_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/perturbresult', exist_ok=True)


print('Creating init image')
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./fine_tune_imgnet-000005.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
base_model_id,variant="fp16", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)
pipe = TgateSDDeepCacheLoader(
       pipe,
       cache_interval=5,
       cache_branch_id=0,
).to(device)
seed = 0
# Initialize a counter for consecutive label mismatch
desired_label_index = 963  # Index for the desired label 'X963'
for n in range(num_samples):
    match_found = False
    while not match_found:
        seedSelect = seed + n
        generator = generator.manual_seed(seedSelect)
        original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device).to(torch.float16)
        
        with torch.inference_mode():
            init_img = pipe.tgate(prompt, guidance_scale=1.4, gate_step=10, num_inference_steps=num_inference_steps, latents=original_lv)["images"][0]

        tensor_image = transform(init_img)
        tensor_image = tensor_image.unsqueeze(0).to(device)
        original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit)

        if original_label == desired_label_index:
            init_img_path = os.path.join(proj_path, f"matched_image_{n}_{original_label}_prompt_{prompt}.png")
            init_img.save(init_img_path)
            print(f"Saved image with matching label {original_label} at: {init_img_path}")
            match_found = True  # Exit the while loop and proceed to the next n
        else:
            # Change seed or modify generation parameters here if desired before retrying
            seed += 1  # Optionally increment seed to try a different generation approach

   # print(f"Saved image with matching label {original_label} at: {init_img_path}")

    init_pop = [
        original_lv + init_perturbation * torch.randn((4, height // 8, width // 8), device=device).to(torch.float16)
        for _ in range(pop_size)
    ]

    best_fitness_score = float('inf')
    best_image_tensor = None
    best_image_index = -1
    now_pop = init_pop
    prev_best = np.inf
    for i in range(gen_steps):

        indivs_lv = torch.cat(now_pop, dim=0).view(-1, 4, height // 8, width // 8).to(torch.float16)
       # print(indivs_lv.shape)
        with torch.inference_mode():
            perturb_img = pipe.tgate([prompt]*(pop_size),guidance_scale = 1.4,gate_step = 10,
                num_inference_steps=num_inference_steps,generator = generator,
                latents=indivs_lv,
            )["images"]
        # all_img_lst.append(perturb_img)
        torch.cuda.empty_cache()
        
        tensor_image2 =torch.stack([transform(image) for image in perturb_img])
        # tensor_image2 = transform(last_image)
        tensor_image2 = tensor_image2.to(device)
        all_logits = classifier(tensor_image2).detach().cpu().numpy()
        perturb_label1 = np.argmax(all_logits).item()
        os.makedirs(os.path.join(proj_path, 'generated_images'), exist_ok=True)
         
        fitness_scores = [
            calculate_fitness(all_logits[k_idx], original_label)
            for k_idx in range(pop_size)
        ]
        # Find the minimum fitness score in the current generation
        current_min_index = np.argmin(fitness_scores)
        current_min_fitness = fitness_scores[current_min_index]
    
        # Update best tracking variables if the current minimum is less than the tracked best
        if current_min_fitness < best_fitness_score:
           best_fitness_score = current_min_fitness
           best_image_tensor  = tensor_image2[current_min_index]  # Ensure a deep copy
           best_image_index = current_min_index

       # print("print fitness",len(fitness_scores))
        # Perform selection
        selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True,
           )[-best_left:]
        now_best = np.min(fitness_scores)
        parent_pop = [now_pop[idx] for idx in selected_indices]
        print("now_best " + str(now_best) + " average_best " + str(np.mean(fitness_scores)))
        wandb.log({"ft_score":now_best})
        if now_best < 0:
           break
        elif now_best == prev_best:
            perturbation_size *= 2
        else:
            perturbation_size = init_perturbation
        k_pop = []
       # print("Size of parent_pop:", len(parent_pop))
        # select k-idx for cross_over genes
        for k_idx in range(pop_size - best_left):
            mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
           # print("mom_idx:", mom_idx, "pop_idx:", pop_idx)
            spl_idx = np.random.choice(4, size=1)[0]
            k_gene = torch.cat(
                [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                dim=1,
            )  # crossover

            # Mutation
            diffs = (k_gene != original_lv).float()
            k_gene += (
                   perturbation_size * torch.randn(k_gene.size(), device=k_gene.device) * diffs
            )  # random adding noise only to diff places
            k_pop.append(k_gene)
           # print("Size of k_pop:", len(k_pop))

        # Combine parent_pop and k_pop for the next generation
        now_pop = parent_pop + k_pop
        prev_best = now_best
        # Apply a final clamp across all now_pop before next iteration starts
        now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

    # After the loop, save the last image if it exists
    # After the loop, check if the best image tensor exists
    if best_image_tensor is not None:

       best_image_tensor = best_image_tensor.to(device)
       best_image_tensor1 =scale_and_unnormalize_tensor(best_image_tensor).to(device)

    # Prepare tensor for classification
    tensor_image = best_image_tensor.unsqueeze(0)  # Ensure it has a batch dimension if needed
    tensor_image_np = tensor_image.squeeze().detach().cpu().numpy()
    perturb_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    perturb_label = np.argmax(perturb_logit)
    predicted_labels.append(perturb_label)
    all_img_lst.append(tensor_image_np)

    # Save the tensor as a numpy array
    image_filename = f'image_{n}_iteration{i}_X{original_label}_Y{perturb_label}.npy'
    np.save(os.path.join(proj_path, 'generated_images', image_filename), tensor_image_np)
    tensor_image1 = best_image_tensor1.unsqueeze(0)  # Ensure it has a batch dimension if needed

    # Save image for visualization
    if tensor_image1.dim() == 4 and tensor_image1.shape[0] == 1:  # Squeeze if single batch item
        tensor_image1 = tensor_image1.squeeze(0)
    tensor_image1 = tensor_image1.cpu()  # Move to CPU for saving
    best_image_pil = transforms.ToPILImage()(tensor_image1)
    image_filename = f'image_{n}_iteration{i}_X{original_label}_Y{perturb_label}.png'
    last_image_path = os.path.join(proj_path, 'perturbresult', image_filename)
    os.makedirs(os.path.dirname(last_image_path), exist_ok=True)  # Ensure directory exists
    best_image_pil.save(last_image_path)
else:
    print("No image tensor available")


   # best_image_pil.save(last_image_path)
    image_info.append((n, i, original_label, perturb_label))
# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(proj_path, "bound_imgs_imgnt_sd.npy"), all_imgs)

