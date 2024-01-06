import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import numpy as np
import torch
from torch import autocast
from PIL import Image
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).to(device)
classifier.eval()
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
prompt = "A photo of 1pizza" # prompt to dream about
proj_name = "test"
num_inference_steps = 25
width = 512
height = 512
init_perturbation = 0.02
best_left = 5
gen_steps = 250
perturbation_size=0.01
pop_size = 10
fitness_scores = []
all_img_lst = []
num_samples = 100
proj_path = "./evolution_sdsinvad_imgnt/"+proj_name+"_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/Newresultjump', exist_ok=True)

print('Creating init image')
lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./img_sd/fine_tune_imgnet-000005.safetensors"
pipe = StableDiffusionPipeline.from_pretrained(
base_model_id, scheduler=lms, safety_checker=None).to(device)
pipe.load_lora_weights(weights_path)

for n in range(num_samples):
    original_lv = torch.randn((1, 4, height // 8, width // 8), device=device)
    with autocast("cuda"):
         init_img = pipe([prompt], num_inference_steps=25, latents=original_lv, width=width, height=height)["images"][0]
         init_img_path = os.path.join(proj_path, f'_origin_{n}.png')
         init_img.save(init_img_path)
         print(f"Original image {n} saved at {init_img_path}")
    # Assuming 'image' is your PIL Image or similar
    # Assuming 'image' is your PIL Image or similar
   # model_image_pil = Image.fromarray(np.uint8(init_img))
   # resized_image = model_image_pil.resize((64, 64), Image.Resampling.LANCZOS)
    tensor_image = transform(init_img)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    init_pop = [
        original_lv + init_perturbation * torch.randn((4, height // 8, width // 8), device=device)
        for _ in range(pop_size)
    ]
    now_pop = init_pop
    prev_best = np.inf
    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5 * torch.ones(original_lv.size(), device=device)
    )
    for i in range(gen_steps):
        indivs_lv = torch.cat(now_pop, dim=0).view(-1, 4, height // 8, width // 8)
        print(indivs_lv.shape)
        with autocast("cuda"):
            perturb_img = pipe([prompt]*10,
                num_inference_steps=num_inference_steps,
                latents=indivs_lv,
            )["images"]
            all_img_lst.append(perturb_img)
        #if isinstance(perturb_img, list) and perturb_img:
           # last_image = perturb_img[0]
        tensor_images = [transform(img) for img in perturb_img]
        # Assuming 'image' is your PIL Image or similar
       # model_image_pil = Image.fromarray(np.uint8(tensor_img))
       # resized_image = model_image_pil.resize((64, 64), Image.Resampling.LANCZOS)
       # tensor_image = transform(tensor_images)

        tensor_image_batch = torch.stack(tensor_images).to(device)
        all_logits = classifier(tensor_image_batch).squeeze().detach().cpu().numpy()
        #perturb_label = np.argmax(perturb_logits).item()
        fitness_scores = [
            calculate_fitness(all_logits[k_idx], original_label)
            for k_idx in range(pop_size)
        ]
        # Perform selection
        selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
            )[-best_left:]
        parent_pop = [now_pop[idx] for idx in selected_indices]
        now_best = np.min(fitness_scores)
        print("now_best " + str(now_best) + " average_best " + str(np.mean(fitness_scores)))
        if now_best < 0:
           break
        elif now_best == prev_best:
            perturbation_size *= 2
        else:
            perturbation_size = init_perturbation
        k_pop = []
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
            # random matching to latent_images[i]
            interp_mask = binom_sampler.sample()
            k_gene = interp_mask * original_lv + (1 - interp_mask) * k_gene
            k_pop.append(k_gene)


        now_pop = parent_pop + k_pop
        prev_best = now_best
    mod_best = parent_pop[-1]
    print("Shape of mod_best:", mod_best.shape)
    print("Type of mod_best:", mod_best.dtype)
    last_image = pipe([prompt]*1,
                       num_inference_steps=num_inference_steps,
                       latents=mod_best,
                       )["images"]
    if isinstance(perturb_img, list) and perturb_img:
            last_image = perturb_img[0]
    tensor_image = transform(last_image)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    perturb_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    perturb_label = np.argmax(original_logit).item()
   

    image_filename = f'image_{n}_X{original_label}_Y{perturb_label}.png'
    last_image_path = os.path.join(proj_path, 'Newresultjump', image_filename)
    last_image.save(last_image_path)
    print(f"Last image saved at {last_image_path}")


