import os
import random
import cv2
import re
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler                                
import numpy as np
#import wandb
import torch
from torch import autocast
from PIL import Image
from torchvision import transforms
from collections import Counter
from experiments.mnist_experiments.mnist_gan.conditional_gan.sa.model import MnistClassifier
#run =wandb.init(project="sinvadtestfitness_mnist")
img_size = 28 * 28
def process_image(image):
    """
    Convert a 3-channel RGB PIL Image to grayscale, resize it to 28x28 pixels,
    and convert it to a PyTorch tensor.

    Parameters:
    - image (PIL.Image): The input RGB image.

    Returns:
    - tensor (torch.Tensor): The processed image as a PyTorch tensor.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided image needs to be a PIL Image.")

    # Convert PIL Image to numpy array (RGB)
    img_np = np.array(image)

    # Convert the image from RGB to grayscale
    gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(gray_image, (28, 28))

    # Convert the numpy array back to PIL Image (to use torchvision transforms)
    img_pil = Image.fromarray(resized_image)

    # Convert PIL Image to PyTorch Tensor
    transform = transforms.ToTensor()
    tensor = transform(img_pil)

    return tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
generator = torch.Generator(device = 'cuda')
torch.use_deterministic_algorithms(True)
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
classifier = MnistClassifier(img_size=img_size).to(device)
# Load pretrained model
classifier.load_state_dict(
    torch.load(
        "./sa/models/MNIST_conv_classifier.pth",
        map_location=device,
   )
)
classifier.eval()

prompts =[ "A photo of Z0ero Number0","A photo of one1 Number1","A photo of two2 Number2 ","A photo of three3 Number3","A photo of Four4 Number4","A photo of Five5 Number5","A photo of Six6 Number6","A photo of Seven7 Number7 ","A photo of Eight8 Number8","A photo of Nine9 Number9"]  # prompt to dream about
proj_name = "test"
num_inference_steps = 15
width = 512
height = 512
init_perturbation = 0.00216688718795776 # (low_mutation_value)
best_left = 10
perturbation_size = 0.00108344359397888 # (low_mutation_value)
max_val =5.42081117630005
min_val = -5.41362476348877
gen_num = 250
pop_size = 25
fitness_scores = []
predicted_labels = []
all_img_lst = []
imgs_to_samp = 100
image_info = []
proj_path = "./mnist_sd/"+proj_name+"_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/Newresult', exist_ok=True)
os.makedirs(os.path.join(proj_path, 'generated_images'), exist_ok=True)

print('Creating init image')
#lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./Mnist_Lora_sdv1.5.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
base_model_id,variant="fp16", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config , rescale_betas_zero_snr=True)
pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)
seed = 0
saved_images = 0

for n in range(imgs_to_samp):  # Loop exactly 100 times
    seedSelect = seed + n  # Use current iteration index to update the seed
    generator = generator.manual_seed(seedSelect)

    original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), device=device).to(torch.float16)

    # Generate the prompt and expected label
    randprompt = random.choice(prompts)
    expected_label = int(re.search(r"Number(\d+)", randprompt).group(1))  # Extract the expected label

    with torch.inference_mode():
        init_img = pipe(prompt=randprompt, guidance_scale=3.5, num_inference_steps=num_inference_steps, latents=original_lv)["images"][0]

    # Process the generated image
    tensor_image = process_image(init_img)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()

    # Check if the predicted label matches the expected label
    if original_label == expected_label:
        # If the label matches, save the image and proceed with the genetic algorithm
        init_img_path = os.path.join(proj_path, f'image_{saved_images}_X{original_label}_prompt_{expected_label}.png')
        init_img.save(init_img_path)
        print(f"Image {saved_images} with matching label saved at {init_img_path}")
        saved_images +=1
        # Proceed with the genetic algorithm now that the label matches
        init_pop = [
            original_lv + init_perturbation * torch.randn((4, height // 8, width // 8), device=device).to(torch.float16)
            for _ in range(pop_size)
        ]

        best_fitness_score = float('inf')
        best_image_tensor = None
        best_image_index = -1
        now_pop = init_pop
        prev_best = np.inf

        for g_idx in range(gen_num):  # Start from 1 for genetic algorithm steps
            indivs_lv = torch.cat(now_pop, dim=0).view(-1, 4, height // 8, width // 8).to(torch.float16)
            print(indivs_lv.shape)
            with torch.inference_mode():
                perturb_img = pipe([randprompt] * pop_size, guidance_scale=1.4, generator=generator, 
                    num_inference_steps=num_inference_steps,
                    latents=indivs_lv,
                )["images"]

            torch.cuda.empty_cache()
            tensor_image2 = torch.stack([process_image(image) for image in perturb_img])
            tensor_image2 = tensor_image2.to(device)
            all_logits = classifier(tensor_image2).detach().cpu().numpy()
            perturb_label1 = np.argmax(all_logits).item()
            print(all_logits.shape)
            print(tensor_image2.shape)

            fitness_scores = [
                calculate_fitness(all_logits[k_idx], original_label)
                for k_idx in range(pop_size)
            ]
            print("print fitness", len(fitness_scores))

            # Find the minimum fitness score in the current generation
            current_min_index = np.argmin(fitness_scores)
            current_min_fitness = fitness_scores[current_min_index]

            # Update best tracking variables if the current minimum is less than the tracked best
            if current_min_fitness < best_fitness_score:
                best_fitness_score = current_min_fitness
                best_image_tensor = tensor_image2[current_min_index]  # Ensure a deep copy
                best_image_index = current_min_index

            # Perform selection
            selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[-best_left:]
            now_best = np.min(fitness_scores)
            parent_pop = [now_pop[idx] for idx in selected_indices]
            print("now_best " + str(now_best) + " average_best " + str(np.mean(fitness_scores)))
           # wandb.log({"ft_score": now_best})

            if now_best < 0:
                break
            elif now_best == prev_best:
                perturbation_size *= 2
            else:
                perturbation_size = init_perturbation

            k_pop = []
            print("Size of parent_pop:", len(parent_pop))

            # Select k-idx for cross_over genes
            for k_idx in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                print("mom_idx:", mom_idx, "pop_idx:", pop_idx)
                spl_idx = np.random.choice(4, size=1)[0]
                k_gene = torch.cat(
                    [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                    dim=1,
                )  # crossover

                # Mutation
                diffs = (k_gene != original_lv).float()
                k_gene += (
                    perturbation_size * torch.randn(k_gene.size(), device=k_gene.device) * diffs
                )  # random adding noise only to different places

                k_pop.append(k_gene)

            # Combine parent_pop and k_pop for the next generation
            now_pop = parent_pop + k_pop
            prev_best = now_best
            # Apply a final clamp across all now_pop before next iteration starts
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

        # After the loop, save the last image if it exists
        if best_image_tensor is not None:
            tensor_image = best_image_tensor.to(device)
            tensor_image = tensor_image.unsqueeze(0).to(device)
            tensor_image_np = tensor_image.squeeze().detach().cpu().numpy()
            perturb_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
            perturb_label = np.argmax(perturb_logit).item()
            predicted_labels.append(perturb_label)
            all_img_lst.append(tensor_image_np)

            # Save the image as a numpy array
            image_filename = f'image_{saved_images-1}_iteration{g_idx + 1}_X{original_label}_Y{perturb_label}'
            np.save(os.path.join(proj_path, 'generated_images', image_filename), tensor_image_np)
        else:
            print("image is none")

    else:
        # If no match, simply skip to the next iteration (no image is saved or processed)
        print(f"Label mismatch for image {saved_images}: expected {expected_label}, got {original_label}")

# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(proj_path, "bound_imgs_mnist_sd.npy"), all_imgs)



