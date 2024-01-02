import os
from diffusers import DDPMScheduler, DDPMPipeline
import numpy as np
import torch
from torch import autocast
from PIL import Image
from torchvision import transforms
from mnist_classifier.model import MnistClassifier
#from huggingface_hub import notebook_login
#notebook_login()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 28*28

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
        "./mnist_classifier/MNIST_conv_classifier.pth",
        map_location=device,
   )
)
classifier.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
])

proj_name = "test"
num_inference_steps = 25
width = 512
height = 512
init_perturbation = 0.02
best_left = 5
perturbation_size = 0.01
frame_index = 0
gen_steps = 250
pop_size = 10
all_img_lst = []
num_samples = 100
proj_path = "./evolution2/"+proj_name+"_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/Newresultjump', exist_ok=True)

print('Creating init image')
scheduler = DDPMScheduler.from_pretrained("Maryamm/ddpm_finetune_mnist")
model = DDPMPipeline.from_pretrained("Maryamm/ddpm_finetune_mnist").to("cuda")
scheduler.set_timesteps(1000)
generator = torch.Generator(device='cuda:0').manual_seed(0)
for n in range(num_samples):
    original_lv = torch.randn((1, 1, 28, 28), device=device)
    for t in scheduler.timesteps:
      with torch.no_grad():
          noisy_residual = model.unet(original_lv, t).sample
          prev_noisy_sample = scheduler.step(noisy_residual, t, original_lv).prev_sample
          original_lv = prev_noisy_sample
      image = ( original_lv/ 2 + 0.5).clamp(0, 1)
      image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
      # Check if the image is grayscale (i.e., only one channel)
      if image.shape[-1] == 1:
      # Squeeze out the channel dimension
        image = image.squeeze(-1)
    image = (image * 255).round().astype("uint8")
    init_img = Image.fromarray(image)
    init_img_path = os.path.join(proj_path, f'_origin_{n}.png')
    init_img.save(init_img_path)
    print(f"Original image {n} saved at {init_img_path}")
    # Assuming 'image' is your PIL Image or similar
    # model_image_pil = Image.fromarray(np.uint8(init_img))
    # resized_image = model_image_pil.resize((28, 28), Image.Resampling.LANCZOS)
    tensor_image = transform(init_img)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    original_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    init_pop = [
        original_lv + init_perturbation * torch.randn((1, 28, 28), device=device)
        for _ in range(pop_size)
    ]
    now_pop = init_pop
    prev_best = np.inf
    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5 * torch.ones(original_lv.size(), device=device)
    )
   # original_latent = original_lv
   # prev_best = np.inf
    for i in range(gen_steps):
        indivs_lv = torch.cat(now_pop, dim=0).view(-1, 1, 28, 28)
        print(indivs_lv.shape)
        all_perturb_imgs = []  # List to store all perturbed images

        for t in scheduler.timesteps:
            with torch.no_grad():
                 noisy_residual = model.unet(indivs_lv, t).sample
                 prev_noisy_sample = scheduler.step(noisy_residual, t, indivs_lv).prev_sample
                 indivs_lv = prev_noisy_sample

        # Normalize and clamp the image
        images = (indivs_lv / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()

        for img_array in images:
             # Check if the image is grayscale (i.e., only one channel)
            if img_array.shape[-1] == 1:
                 img_array = img_array.squeeze(-1)  # Squeeze out the channel dimension
            img_array = (img_array * 255).round().astype("uint8")
            perturb_img = Image.fromarray(img_array)
            all_perturb_imgs.append(perturb_img)

        #if isinstance(perturb_img, list) and perturb_img:
           # last_image = perturb_img[0]
        print("perturb_images_length",len(all_perturb_imgs))
        tensor_image2 = torch.stack([transform(image) for image in all_perturb_imgs])
        # Handle a single image
        # model_image_pil = Image.fromarray(np.uint8(last_image))
        # resized_image2 = model_image_pil.resize((28, 28), Image.Resampling.LANCZOS)
        # tensor_image2 =torch.stack([transform(image) for image in perturb_img])
        # tensor_image2 = transform(last_image)
        tensor_image2 = tensor_image2.to(device)
        all_logits = classifier(tensor_image2).detach().cpu().numpy()
        perturb_label = np.argmax(all_logits).item()
        print(all_logits.shape)
        print(tensor_image2.shape)
        fitness_scores = [
            calculate_fitness(all_logits[k_idx], original_label)
            for k_idx in range(pop_size)
        ]
        print("print fitness",len(fitness_scores))
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
        print("Size of parent_pop:", len(parent_pop))
        
        for k_idx in range(pop_size - best_left):
            mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
            print("mom_idx:", mom_idx, "pop_idx:", pop_idx)
            spl_idx = np.random.choice(1, size=1)[0]
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
    mod_best = parent_pop[-1].view(1, 1, 28,28)
    for t in scheduler.timesteps:
       with torch.no_grad():
          noisy_residual = model.unet(mod_best, t).sample
          prev_noisy_sample = scheduler.step(noisy_residual, t, mod_best).prev_sample
          mod_best = prev_noisy_sample
    image = ( indivs_lv/ 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    # Check if the image is grayscale (i.e., only one channel)
    if image.shape[-1] == 1:
    # Squeeze out the channel dimension
      image = image.squeeze(-1)
    image = (image * 255).round().astype("uint8")
    last_img = Image.fromarray(image) 
    tensor_image = transform(last_img)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    perturb_logit = classifier(tensor_image).squeeze().detach().cpu().numpy()
    perturb_label = np.argmax(original_logit).item()


    image_filename = f'image_{n}_X{original_label}_Y{perturb_label}.png'
    last_image_path = os.path.join(proj_path, 'Newresultjump', image_filename)
    last_img.save(last_image_path)
    print(f"Last image saved at {last_image_path}")

     
