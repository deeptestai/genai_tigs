import torch
import os
#import wandb
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
from vae.svhn_vae.svhn_vae.model import ConvVAE
from unconditional_gan.classifier.model import VGGNet

#run = wandb.init(project="Sinvad_latent_beased_SVHN")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 32 * 32 * 3
# Load the trained models
vae = ConvVAE(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800).to(device)
classifier = VGGNet().to(device)
vae.load_state_dict(
    torch.load(
        "./vae/svhn_vae/svhn_vae/svhn_convend.pth",    #path to vae ckpt
        map_location=device,
    )
)
classifier.load_state_dict(
    torch.load(
        "./classifier/SVHN_vggnet.pth",             #adjust your own path to classifier
        map_location=device,
    )
)
# Set models to evaluation mode
vae.eval()
classifier.eval()
result_dir = "./result_vae_svhn"  # Directory to save the images
# Subdirectory for original images
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "original_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(prtb_img_dir, exist_ok=True)
test_dataset = torchvision.datasets.SVHN(root='./data', split="test", transform=transforms.ToTensor(), download=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
print("Data loader ready...")



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


image_info = []
predictions = []
num_samples = 100
gen_num = 250
pop_size = 25
best_left = 10
min_val=-5.26724052429199
max_val=4.72675895690918
imgs_to_samp = 100
perturbation_size = 0.00999399948120117 #  perturbation size
initial_perturbation_size = 0.0199879989624023  # Initial perturbation size

all_img_lst = []
saved_images = 0  # Initialize saved image counter

### multi-image sample loop ###
for img_idx in trange(imgs_to_samp):  # Loop will run for 100 iterations
    for i, (x, x_class) in enumerate(test_data_loader):
        samp_img = x[0:1]
        samp_class = x_class[0].item()

    img_enc, _ = vae.encode(samp_img.to(device))
    original_lv = img_enc
    original_image = vae.decode(original_lv).view(-1, 3, 32, 32)
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()

    # Only save and proceed with GA if the original label matches the sample class
    if original_label == samp_class:
        print(f"True class: {samp_class}, Predicted class: {original_label}")

        # Save original image when the class matches
        original_image_np = original_image.squeeze().detach().cpu().numpy()

        # Define the filename that includes the label
        filename = f"original_image_{saved_images}_X_{original_label}.npy"

        # Define the full path for saving the file
        file_path = os.path.join(original_images_dir, filename)

        # Save the image data as a NumPy file
        np.save(file_path, original_image_np)

        # Increment the saved images counter when a match is found and saved
        saved_images += 1

        ### Initialize genetic algorithm (GA) ###
        init_pop = [
            original_lv + initial_perturbation_size * torch.randn(1, 800).to(device)
            for _ in range(pop_size)
        ]
        now_pop = init_pop
        prev_best = np.inf
        best_fitness_score = np.inf
        best_image_tensor = None
        best_image_index = -1

        ### GA ###
        for g_idx in range(gen_num):
            indivs = torch.cat(now_pop, dim=0)
            dec_imgs = vae.decode(indivs).view(-1, 3, 32, 32)
            all_logits = classifier(dec_imgs).squeeze().detach().cpu().numpy()

            fitness_scores = [
                calculate_fitness(all_logits[k_idx], original_label)
                for k_idx in range(pop_size)
            ]
            # Finding the minimum fitness score in this generation
            current_min_index = np.argmin(fitness_scores)
            current_min_fitness = fitness_scores[current_min_index]

            # Update global minimum if the current score is lower
            if current_min_fitness < best_fitness_score:
                best_fitness_score = current_min_fitness
                best_image_tensor = dec_imgs[current_min_index].cpu().detach()
                best_image_index = current_min_index

            # Perform selection
            selected_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True,
            )[-best_left:]

            now_best = np.min(fitness_scores)
            parent_pop = [now_pop[idx] for idx in selected_indices]

            # Perform crossover and mutation
            print(
                "now_best "
                + str(now_best)
                + ", Average_score "
                + str(np.mean(fitness_scores))
            )

            if now_best < 0:
                break
            elif now_best == prev_best:
                perturbation_size *= 2
            else:
                perturbation_size = initial_perturbation_size

            k_pop = []
            for k_idx in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(800, size=1)[0]
                k_gene = torch.cat(
                    [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                    dim=1,
                )  # crossover

                # Mutation
                diffs = (k_gene != original_lv).float()
                k_gene += (
                    perturbation_size * torch.randn(k_gene.size()).to(device) * diffs
                )  # random adding noise only to diff places
               
                k_pop.append(k_gene)

            # Current population is obtained combining the parent pop and its offspring
            now_pop = parent_pop + k_pop
            prev_best = now_best
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

        # Assuming best_image_tensor is the tensor that needs to be processed
        mod_best_image_tensor = best_image_tensor.to(device)  # Move tensor to the appropriate device if not already
        mod_best_image_np = best_image_tensor.cpu().detach().numpy()  # Convert tensor to numpy array after moving to CPU

        # Assuming classifier is already defined and appropriate for the tensor as is
        final_bound_logits = classifier(mod_best_image_tensor.unsqueeze(0))  # Ensure tensor is in correct shape for classifier
        predicted_best_label = torch.argmax(final_bound_logits, dim=1).item()

        # Define the path for saving the numpy file with detailed filename
        image_path_np = os.path.join(
            result_dir,
            f"image_{saved_images}_iteration{g_idx}_X{original_label}_Y{predicted_best_label}.npy"
        )

        # Save the numpy array to a file
        np.save(image_path_np, mod_best_image_np)

        all_img_lst.append(mod_best_image_np)

    else:
        # Skip if no match found
        print(f"Mismatch: True class: {samp_class}, Predicted class: {original_label}")

# Save all generated images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, "bound_imgs_svhn_vae.npy"), all_imgs)

print(f"Total matching images saved: {saved_images}")
