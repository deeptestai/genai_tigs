import torch
from tqdm import trange
# import wandb
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from sa.model import MnistClassifier
from cdcgan.cdcgan_mnist import Generator

# run = wandb.init(project="Sinvad_fitness_CDCGAN_Mnist2")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28 * 28 * 1
# Load the trained models
G = Generator(ngpu=1, nz=100, nc=1).to(device)  # Modify the Generator architecture
classifier = MnistClassifier(img_size=img_size).to(device)

# Load the weights of the models
G.load_state_dict(
    torch.load("./cdcgan/weights/netG_epoch_10cdcgan.pth", map_location=device)
)
classifier.load_state_dict(
    torch.load(
        "./sa/models/MNIST_conv_classifier.pth",
        map_location=device,
    )
)
# Set models to evaluation mode
G.eval()
classifier.eval()
result_dir = "./result23_cdcgan_mnist"  # Directory to save the images
# Subdirectory for original images
original_images_dir = os.path.join(result_dir, "original_images")
# Subdirectory for perturbed images
prtb_img_dir = os.path.join(result_dir, "perturb_images")

# Create the directories if they do not exist
os.makedirs(original_images_dir, exist_ok=True)
os.makedirs(prtb_img_dir, exist_ok=True)


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
# Directory to save images for analysis
image_save_dir = './generated_images'
os.makedirs(image_save_dir, exist_ok=True)

image_info = []
predictions = []
imgs_to_samp = 100
gen_num = 250
pop_size = 25
best_left = 10
min_val = -4.78276300430298
max_val = 4.08758640289307
perturbation_size = 0.000887034940719605  # Default perturbation size (low-mutation_extent)
initial_perturbation_size = 0.00177406988143921 # Initial perturbation size (low-mutation_extent)
# Generate a random latent vector
latent_space = torch.randn(imgs_to_samp, 100, 1, 1).to(device)
random_labels = torch.randint(0, 10, (imgs_to_samp,)).to(device)
all_img_lst = []
saved_images =0
for img_idx in trange(imgs_to_samp):
    # Generate the non-perturbed image
    original_latent = latent_space[img_idx].unsqueeze(0)
    original_labelr = random_labels[img_idx]
    expected_label  = original_labelr.item()
    # Generate the non-perturbed image using the generator
    original_image = G(original_latent, original_labelr.view(1))
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    # Compare the predicted label with the label used for generation
    if original_label == expected_label:
        print(f"Match found: True label {original_labelr.item()}, Predicted label {original_label}")
       # Calculate fitness for the original image
        original_image_path = os.path.join(original_images_dir,f'original_image{saved_images}_X{original_label}.png')
        # original_image_pil.save(original_image_path)
        save_image(original_image,original_image_path)
        # Convert the tensor to a NumPy array
        original_image_np = original_image.squeeze().detach().cpu().numpy()

        # Define the filename that includes the label
        filename = f"original_image_{saved_images}_X_{original_label}.npy"

        # Define the full path for saving the file
        file_path = os.path.join(original_images_dir, filename)

        # Save the image data as a NumPy file
        np.save(file_path, original_image_np)
    
        saved_images +=1

        ### Initialize optimization ###
        init_pop = [
            torch.clamp(original_latent.unsqueeze(0)
            + initial_perturbation_size * torch.randn_like(original_latent), min=min_val, max=max_val)
            for _ in range(pop_size)
        ]
        now_pop = init_pop
        prev_best = np.inf
        best_fitness_score = np.inf
        best_image_tensor = None
        best_image_index = -1
        init_label = torch.tensor([original_label] * pop_size)
        ###GA_algorithm###
        for g_idx in range(gen_num):
            indivs_lV = torch.cat(now_pop, dim=0).view(-1, 100, 1, 1).to(device)
            indivs_labels = init_label.to(device)

            # Generate images using the generator
            Gen_imgs = G(indivs_lV, indivs_labels)
            predicted_labels = torch.argmax(classifier(Gen_imgs), dim=1).cpu()
            # Calculate fitness scores for the current generation
            all_logits = classifier(Gen_imgs).squeeze().detach().cpu().numpy()
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
                best_image_tensor = Gen_imgs[current_min_index].cpu().detach()
                best_image_index = current_min_index

            # Perform selection
            selected_indices = sorted(
                range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
            )[-best_left:]
            now_best = np.min(fitness_scores)

            parent_pop = [now_pop[i] for i in selected_indices]

            # Perform crossover and mutation
            print("now_best " + str(now_best) + " now_best " + str(np.mean(fitness_scores)))
            #wandb.log({"ft_scores": now_best})

            if now_best < 0:
               break
            elif now_best == prev_best:
                perturbation_size *= 2
            else:
                perturbation_size = initial_perturbation_size

            k_pop = []
            for k_idx in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(100, size=1)[0]
                k_gene = torch.cat(
                    [
                        parent_pop[mom_idx][:, :spl_idx],
                        parent_pop[pop_idx][:, spl_idx:],
                    ],
                    dim=1,
                )  # Apply crossover to latent vector

                # Mutation for latent vector
                diffs = (k_gene != original_latent).float()
                k_gene += perturbation_size * torch.randn_like(k_gene).to(device) * diffs

                k_pop.append(k_gene)

            now_pop = parent_pop + k_pop
            prev_best = now_best
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

   
        # Assuming best_image_tensor is a tensor that needs to be processed
        mod_best_image_tensor = best_image_tensor.to(device)  # Move tensor to the appropriate device if not already
        mod_best_image_np = best_image_tensor.cpu().detach().numpy()  # Convert tensor to numpy array after moving to CPU

        # Assuming classifier is already defined and appropriate for the tensor as is
        final_bound_logits = classifier(mod_best_image_tensor.unsqueeze(0))  # Ensure tensor is in correct shape for classifier
        predicted_best_label = torch.argmax(final_bound_logits, dim=1).item()

       # Define the path for saving the numpy file with detailed filename
        image_path_np = os.path.join(
            result_dir,
            f"image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.npy"
        )

        # Save the numpy array to a file
        np.save(image_path_np, mod_best_image_np)
    
        all_img_lst.append(mod_best_image_np)

        image_path = os.path.join(prtb_img_dir,f'image{saved_images}_iteration_{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png',)
        # perturbed_image_pil.save(image_path)
        save_image(mod_best_image_tensor,image_path)
    else:
        # Skip if no match found
        print(f"Mismatch: True class: {expected_label}, Predicted class: {original_label}")

    
# Save all generated images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, "bound_imgs_mnist_cdcgan.npy"), all_imgs)
