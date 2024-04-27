import os
import re
import random
from diffusers import  StableDiffusionPipeline,AutoencoderKL
from tgate import TgateSDDeepCacheLoader
from diffusers.schedulers import DPMSolverMultistepScheduler
#from libs.benchmark import benchmark
import numpy as np
import torch
import wandb
from torch import autocast
from PIL import Image
from torchvision import transforms
from collections import Counter
from svhn_classifier.model import VGGNet
run =wandb.init(project="sinvadsvhnfitness")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
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
classifier = VGGNet().to(device)
# Load pretrained model
classifier.load_state_dict(
    torch.load(
        "./svhn_classifier/model/SVHN_vggnet.pth",
        map_location=device,
   )
)
#classifier = classifier.to(device='cuda', dtype=torch.float16)
classifier.eval()
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        #transforms.Grayscale(num_output_channels=1),  # This converts to grayscale
        transforms.ToTensor(),
    ]
)
prompts =[ "A photo of HouseNo0 Hnumber0","A photo of HouseNo1 Hnumber1","A photo of HouseNo2 Hnumber2 ","A photo of HouseNo3 Hnumber3","A photo of HouseNo4 Hnumber4","A photo of HouseNo5 Hnumber5","A photo of HouseNo6 Hnumber6","A photo of HouseNo7 Hnumber7 ","A photo of HouseNo8 Hnumber8","A photo of HouseNo9 Hnumber9"]  # prompt to dream about
proj_name = "test"
num_inference_steps = 25
width = 512
height = 512

init_perturbation = 0.00217826347351074
best_left = 10
perturbation_size = 0.00108913173675537
gen_steps = 250
pop_size = 25
min_val = -5.62070274353027
max_val = 5.27061462402344
fitness_scores = []
all_img_lst = []
num_samples = 100
image_info = []
predicted_labels = []
proj_path = "./result_svhn_sdsinvad/"+proj_name+"_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/Newresult', exist_ok=True)
 os.makedirs(os.path.join(proj_path, 'generated_images'), exist_ok=True)
print('Creating init image')
#lms = PNDMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear")
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./svhn_finetune_lorav1.5-000005.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
base_model_id,variant="fp16", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
pipe = pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config) # rescale_betas_zero_snr=True)
pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)
pipe = TgateSDDeepCacheLoader(
       pipe,
       cache_interval=5,
       cache_branch_id=0,
).to(device)
seed =0
for n in range(num_samples):
    seedSelect = seed+n
    generator = generator.manual_seed(seedSelect)
    original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8),generator = generator, device=device).to(torch.float16)
    label_matched = False  # Flag to track when the correct label is matched
    attempt_count = 0  # To prevent infinite loops

    while not label_matched and attempt_count < 10:
         randprompt = random.choice(prompts)
         expected_label = int(re.search(r"HouseNo(\d+)", randprompt).group(1))  # Extract the number following 'HouseNo'
         with torch.inference_mode():
              init_img = pipe.tgate(prompt = randprompt,guidance_scale=1.4,gate_step = 10, num_inference_steps= num_inference_steps,latents=original_lv)["images"][0]
         tensor_image = transform(init_img)
         tensor_image = tensor_image.unsqueeze(0).to(device)
         original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
         original_label = np.argmax(original_logit).item()
         # Check if the predicted label matches the expected label from the prompt
         if original_label == expected_label:
           # Save the image only if the label matches
            init_img_path = os.path.join(proj_path, f'image_{n}_X{original_label}_prompt_{expected_label}.png')
            init_img.save(init_img_path)
            print(f"Image {n} with matching label saved at {init_img_path}")
            label_matched = True
         else:
            print(f"Attempt {attempt_count + 1}: Label mismatch for image {n}: expected {expected_label}, got {original_label}")
            attempt_count += 1
    if not label_matched:
        print(f"No matching prompt found after 10 attempts for image {n}.")    

    init_pop = [
      original_lv + init_perturbation * torch.randn((pipe.unet.config.in_channels, height // 8, width // 8), device=device).to(torch.float16)
       for _ in range(pop_size)
    ]

    best_fitness_score = float('inf')
    best_image_tensor = None
    best_image_index = -1
    now_pop = init_pop
    prev_best = np.inf
    for i in range(gen_steps):
       # Flatten all tensors in now_pop into a single tensor for min/max evaluation
        indivs_lv = torch.cat(now_pop, dim=0).view(-1, 4, height // 8, width // 8).to(torch.float16)
        print(indivs_lv.shape)
        #per
        with torch.inference_mode():                         # torch.no_grad(),torch.cuda.amp.autocast(dtype =torch.bfloat16):
             perturb_img = pipe([randprompt]*(pop_size),guidance_scale = 1.4,generator= generator, 
                num_inference_steps=num_inference_steps,
                latents=indivs_lv,
             )["images"]
        # all_img_lst.append(perturb_img)
        torch.cuda.empty_cache()

        tensor_image2 =torch.stack([transform(image) for image in perturb_img])
        tensor_image2 = tensor_image2.to(device)
        all_logits = classifier(tensor_image2).detach().cpu().numpy()
        perturb_label1 = np.argmax(all_logits).item()
        print(all_logits.shape)
        print(tensor_image2.shape)
  
        fitness_scores = [
            calculate_fitness(all_logits[k_idx], original_label)
            for k_idx in range(pop_size)
        ]
        print("print fitness",len(fitness_scores))

        # Find the minimum fitness score in the current generation
        current_min_index = np.argmin(fitness_scores)
        current_min_fitness = fitness_scores[current_min_index]
    
        # Update best tracking variables if the current minimum is less than the tracked best
        if current_min_fitness < best_fitness_score:
           best_fitness_score = current_min_fitness
           best_image_tensor  = tensor_image2[current_min_index]  # Ensure a deep copy
           best_image_index = current_min_index
        # Perform selection
        selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True,
           )[-best_left:]
        now_best = np.min(fitness_scores)
        parent_pop = [now_pop[idx] for idx in selected_indices]
        print(parent_pop[-1].shape)
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

        # Combine parent_pop and k_pop for the next generation
        now_pop = parent_pop + k_pop
        prev_best = now_best
        # Apply a final clamp across all now_pop before next iteration starts
        now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]
      

   # mod      # .view(1, pipe.unet.config.in_channels, height // 8, width // 8).to(torch.float1
   # with torch.inference_mode():

      # last_image_list = pipe.tgate(prompt=randprompt,guidance_scale= 1.4, gate_step = 10,  num_inference_steps= 10, latents=mod_best)["images"]
    # After the loop, save the last image if it exists
    if best_image_tensor is not None:
       tensor_image = best_image_tensor.to(device)
       tensor_image = tensor_image.unsqueeze(0).to(device)
       tensor_image_np= tensor_image.squeeze().detach().cpu().numpy()
       perturb_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
       perturb_label = np.argmax(perturb_logit).item()
       predicted_labels.append(perturb_label)
       all_img_lst.append(tensor_image_np)
       # Save the image as a numpy array
       image_filename = f'image_{n}_iteration{i}_X{original_label}_Y{perturb_label}.png'
       # Save the image as a numpy array
       np.save(os.path.join(proj_path,'generated_images', image_filename), tensor_image_np)
    else: 
       print("image is none")
    #For saving image for myself
    if best_image_tensor is not None:
       # Ensure tensor is on CPU and squeeze out the batch dimension if necessary
       if best_image_tensor.dim() == 4 and best_image_tensor.shape[0] == 1:  # Checking for single item in batch
            tensor_image = best_image_tensor.squeeze(0).cpu()
       else:
            tensor_image = best_image_tensor.cpu()
    best_image_pil = transforms.ToPILImage()(tensor_image.cpu())
    image_filename = f'image_{n}_iteration{i}_X{original_label}_Y{perturb_label}.png'
    last_image_path = os.path.join(proj_path, 'Newresult', image_filename)
    best_image_pil.save(last_image_path)
    image_info.append((n, i, original_label, perturb_label))
# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(proj_path, "bound_imgs_svhn_sd.npy"), all_imgs)
