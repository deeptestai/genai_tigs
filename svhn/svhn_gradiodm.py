#def run_diffusion_tig(gen_num, pop_size, best_left, perturbation_size, initial_perturbation_size,
#                      imgs_to_samp, image_size_selector, prompt):
import torch
import os 
import re
import numpy as np
from PIL import Image
from DeepCache import DeepCacheSDHelper
from tgate import TgateSDDeepCacheLoader
from diffusers.schedulers import DPMSolverMultistepScheduler
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from torch import autocast
from sa.svhn_classifier.model import VGGNet
import cv2
import random
import zipfile
def process_image(image):
    """
    Resize a 3-channel RGB PIL Image to 32x32 pixels and convert it to a PyTorch tensor.

    Parameters:
    - image (PIL.Image): The input RGB image.

    Returns:
    - tensor (torch.Tensor): The processed image as a PyTorch tensor.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided image needs to be a PIL Image.")

    # Convert PIL Image to numpy array (RGB)
    img_np = np.array(image)

    # Resize the image to 32x32 pixels if it's not already that size
    if img_np.shape[0] != 32 or img_np.shape[1] != 32:
        resized_image = cv2.resize(img_np, (32, 32), interpolation=cv2.INTER_NEAREST)
    else:
        resized_image = img_np

    # Convert the numpy array back to PIL Image (to use torchvision transforms)
    img_pil = Image.fromarray(resized_image)

    # Convert PIL Image to PyTorch Tensor
    transform = transforms.ToTensor()
    tensor = transform(img_pil)

    return tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

result_dir = "./svhn_sd/"
os.makedirs(result_dir, exist_ok=True)
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(original_images_dir, exist_ok=True)
perturb_images_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(perturb_images_dir, exist_ok=True)

base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./sd/svhn_sd/svhn_finetune_lorav1.5-000005.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
base_model_id, variant="fp16", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config)                   
pipe = TgateSDDeepCacheLoader(
       pipe,
       cache_interval=3,
       cache_branch_id=0,
).to(device)
gen_num = 250
pop_size = 25
num_inference_steps = 25
height, width = 512, 512
min_val = -5.62070274353027
max_val = 5.27061462402344
predicted_labels = []
all_img_lst = []
def run_diffusion_tig1(gen_num, pop_size, best_left, perturbation_size, initial_perturbation_size,
                      imgs_to_samp, classifier_choice, prompt, classifier_file):
    if classifier_choice == "VGGNET":
        # Load default SVHN classifier
        classifier = VGGNet().to(device)
        classifier.load_state_dict(
            torch.load("./sa/svhn_classifier/model/SVHN_vggnet.pth", map_location=device)
        )
        classifier.eval()
        print(" Default VGGNet SVHN classifier loaded.")

    elif classifier_choice == "Upload Custom":
        if classifier_file is None:
            raise ValueError("Please upload a .jit TorchScript model file for SVHN.")

        try:
            if not classifier_file.name.endswith(".jit"):
                raise gr.Error(" Only .jit TorchScript files are supported.")
            classifier = torch.jit.load(classifier_file.name, map_location=device)
            classifier.eval()
            print(" Custom TorchScript SVHN classifier loaded.")
        except Exception as e:
            raise ValueError(f" Failed to load custom SVHN classifier: {e}")

    else:
        raise ValueError(" Unknown classifier option selected for SVHN.")



    saved_images = 0
    final_iterations = []
    num_misclassified = 0
    num1_misclassified = 0
    total_images = 0
    all_gallery_items = []
    saved_image_paths = []
    status_rows = []

    seed = 0
    torch_generator = torch.Generator(device=device)

    for n in range(imgs_to_samp):
        seedSelect = seed + n
        generator = torch_generator.manual_seed(seedSelect)

        original_lv = torch.randn((1, 4, 64, 64), device=device).to(torch.float16)

        expected_label = int(re.search(r"HouseNo(\d+)", prompt).group(1))
        print("expected label of each number",expected_label)
        with torch.inference_mode():
            init_img = pipe.tgate(prompt = prompt,guidance_scale= 3.5,gate_step =10, num_inference_steps= num_inference_steps,latents=original_lv)["images"][0]

        tensor_image = process_image(init_img).unsqueeze(0).to(device)
        original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit).item()
        print("original label of each number",original_label)
        # Check if the predicted label matches the expected label
        if original_label != expected_label:
            num1_misclassified += 1
            continue  # Skip mismatches
        #total_images += 1
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
                perturb_img = pipe.tgate([prompt]*(pop_size),guidance_scale =1.4,gate_step = 10, generator= generator, 
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
              tensor_image_np = input_tensor.squeeze().detach().cpu().numpy()
              # Convert perturbed image to PIL
              # Prepare for saving (handle shape issues)
              if tensor_image_np.ndim == 3:
                  if tensor_image_np.shape[0] == 3:
                     # (3, H, W) → (H, W, 3)
                     tensor_image_np = np.transpose(tensor_image_np, (1, 2, 0))
              elif tensor_image_np.shape[0] == 1:
                   tensor_image_np = tensor_image_np.squeeze(0)
              else:
                   raise ValueError(f"Unexpected shape: {tensor_image_np.shape}")
        

              # Create PIL Image
              pert_pil = Image.fromarray((tensor_image_np * 255).astype(np.uint8))

              perturbed_image_path = os.path.join(perturb_images_dir, f"image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png")
              pert_pil.save(perturbed_image_path)

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

              all_gallery_items.append((init_img, f"Expected label: {original_label}"))
              all_gallery_items.append((pert_pil, f"Predicted label: {predicted_best_label} ; iterations: {g_idx + 1}"))
              saved_image_paths.append(original_image_path)
              saved_image_paths.append(perturbed_image_path)




        # Yield progress to Gradio
        yield (
          f"Processing image {n + 1} / {imgs_to_samp}| Misclassified seeds {num_misclassified}|% Misclassification: {misclassification_rate}|Avg Iterations{Avg_iterations}",
          all_gallery_items, None, status_rows
        )
       

        zip_path = os.path.join(result_dir, "generated_pairs.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in saved_image_paths:
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname=arcname)
        # Final yield
        # Final yield
        yield (
          f"Finished! Total saved images: {saved_images} | Misclassified seeds {num_misclassified}| % Misclassification: {misclassification_rate}|Avg Iterations: {Avg_iterations}",


          all_gallery_items,
          zip_path, status_rows
        )





