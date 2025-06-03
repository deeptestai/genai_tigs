#def run_diffusion_tig(gen_num, pop_size, best_left, perturbation_size, initial_perturbation_size,
#                      imgs_to_samp, image_size_selector, prompt):
import torch
import os 
import re
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from tgate import TgateSDDeepCacheLoader
from torch import autocast
import cv2
import random
import zipfile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from PIL import Image
import torch

def tensor_to_pil(tensor_image):
    """
    Safely convert a torch tensor to a PIL image.
    Handles both grayscale (1-channel) and RGB (3-channel) images.
    """
    if not isinstance(tensor_image, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, but got {type(tensor_image)}")
    
    # Remove batch dimension if exists
    if tensor_image.ndim == 4:
        tensor_image = tensor_image.squeeze(0)

    if tensor_image.ndim != 3:
        raise ValueError(f"Tensor must have 3 dimensions (C, H, W) after squeezing, got shape {tensor_image.shape}")

    # Clamp values safely between [0, 1]
    tensor_image = tensor_image.clamp(0, 1)

    # Move to CPU and convert to numpy
    image_np = tensor_image.detach().cpu().numpy()

    # Handle channel arrangement
    if image_np.shape[0] == 1:  # Grayscale
        image_np = image_np[0]  # Remove channel dimension
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np, mode="L")
    elif image_np.shape[0] == 3:  # RGB
        image_np = np.transpose(image_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np, mode="RGB")
    else:
        raise ValueError(f"Unsupported number of channels: {image_np.shape[0]}")
def denormalize(tensor, mean, std):
    """Undo normalization for correct visualization."""
    if tensor.ndim == 4:
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    elif tensor.ndim == 3:
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    else:
        raise ValueError("Expected 3D or 4D tensor")
    return tensor * std + mean
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
#classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).to(device)
#classifier.eval()
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

result_dir = "./imagenet_sd_teddy/"
original_images_dir = os.path.join(result_dir, "original_images")
perturb_images_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(original_images_dir, exist_ok=True)
os.makedirs(perturb_images_dir, exist_ok=True)
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./sd/imagenet_sd/teddy_bear_imagenet-000005.safetensors"
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
gen_num = 250
pop_size = 25
num_inference_steps = 25
height, width = 512, 512
min_val = -5.3243408203125
max_val = 5.55196666717529
predicted_labels = []
all_img_lst = []
def run_diffusion_tig_teddy(gen_num, pop_size, best_left, perturbation_size, initial_perturbation_size,
                      imgs_to_samp, classifier_choice, prompt,classifier_file):
    if classifier_choice == "VGG19bn":
        classifier = torch.hub.load("pytorch/vision:v0.10.0", "vgg19_bn", pretrained=True).to(device)
        classifier.eval()
        print( classifier," Default classifier loaded.")

    elif classifier_choice == "Upload Custom":
        if classifier_file is None:
            raise ValueError("Please upload a .jit TorchScript model file.")

        try:
            classifier = torch.jit.load(classifier_file.name, map_location=device)
            classifier.eval()
            print( " Custom TorchScript classifier loaded.")
        except Exception as e:
            raise ValueError(f" Failed to load custom classifier: {e}")

    else:
        raise ValueError("Unknown classifier selected.")


    all_img_lst = []
    final_iterations = []
    num_misclassified = 0
    num1_misclassified = 0
    total_images = 0
    all_gallery_items = []
    saved_image_paths = []
    status_rows = []
    saved_images = 0

    seed = 0
    torch_generator = torch.Generator(device=device)

    for n in range(imgs_to_samp):
        seedSelect = seed + n
        generator = torch_generator.manual_seed(seedSelect)

        original_lv = torch.randn((1, 4, 64, 64), device=device).to(torch.float16)

        expected_label = 850

        with torch.inference_mode():
            init_img = pipe.tgate(prompt=prompt, guidance_scale=3.5,gate_step =10,
                            num_inference_steps= num_inference_steps, latents=original_lv)["images"][0]

        tensor_image = transform(init_img).unsqueeze(0).to(device)
        original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit).item()
        print("original label of each number",original_label)
        # Check if the predicted label matches the expected label
        if original_label != expected_label:
            num1_misclassified += 1
            continue  # Skip mismatches
        total_images += 1
        # If the label matches, save the image and proceed with the genetic algorithm
        original_image_path  = os.path.join(original_images_dir, f'image_{saved_images}_X{original_label}_prompt_{expected_label}.png')
        init_img.save(original_image_path )

        # Proceed with the genetic algorithm now that the label matches
        init_pop = [
            original_lv + initial_perturbation_size * torch.randn((4, height // 8, width // 8), device=device).to(torch.float16)
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
                perturb_img = pipe([prompt] * pop_size, guidance_scale=1.4, generator=generator, 
                    num_inference_steps=num_inference_steps,
                    latents=indivs_lv,
                )["images"]

            torch.cuda.empty_cache()
            tensor_image2 = torch.stack([transform(image) for image in perturb_img])
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
                perturbation_size = initial_perturbation_size

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



        if best_image_tensor is not None:
              
              final_generation = g_idx + 1
              final_iterations.append(final_generation) 
              # Inference on best perturbed image
              input_tensor = best_image_tensor.unsqueeze(0).to(device)
              with torch.inference_mode():
                  perturbed_logit = classifier(input_tensor).squeeze().detach().cpu().numpy()
              predicted_best_label = np.argmax(perturbed_logit).item()
              print("Predicted label", predicted_best_label)
        
              #pert_pil = Image.fromarray((tensor_image_np * 255).astype(np.uint8))
              denorm_tensor = denormalize(input_tensor,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
              perturbed_image_pil = tensor_to_pil(denorm_tensor)
              # Save perturbed image
              predicted_best_label = torch.argmax(classifier(best_image_tensor.unsqueeze(0).to(device)), dim=1).item()
              perturbed_image_path = os.path.join(perturb_images_dir, f'perturbed_image_{saved_images}_X{predicted_best_label}.png')
              perturbed_image_pil.save(perturbed_image_path)
              # Save row info
              status_rows.append([
                  n + 1,
                  original_label,
                  predicted_best_label,
                  g_idx + 1  # or however you track iterations
              ])
              if predicted_best_label != original_label:
                  num_misclassified += 1


              saved_images += 1

              misclassification_rate = (num_misclassified / saved_images) * 100 if saved_images > 0 else 0
              if final_iterations:
                 Avg_iterations = sum(final_iterations) / len(final_iterations)
              else:
                 Avg_iterations = 0  # or use None if you want to indicate 'no data'
              #total_iterations.append(final_generation)
              all_gallery_items.append((init_img, f"Expected label: {original_label}"))
              all_gallery_items.append((perturbed_image_pil, f"Predicted label: {predicted_best_label}; iterations: {g_idx + 1}"))
              saved_image_paths.append(original_image_path)
              saved_image_paths.append(perturbed_image_path)




        # Yield progress to Gradio
        yield (
          f"Processing image {n + 1} / {imgs_to_samp}|Misclassified seeds {num_misclassified}|% Misclassification: {misclassification_rate} |# total Iterations:{Avg_iterations: .2f}",
          all_gallery_items, None , status_rows
        )
        zip_path = os.path.join(result_dir, "generated_pairs.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in saved_image_paths:
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname=arcname)
        # Final yield
        yield (
          f"Finished! Total saved images: {saved_images} | Misclassified seeds {num_misclassified}| % Misclassification: {misclassification_rate} |Avg Iterations:{Avg_iterations: .2f}",
          all_gallery_items,
          zip_path, status_rows
        )


        

