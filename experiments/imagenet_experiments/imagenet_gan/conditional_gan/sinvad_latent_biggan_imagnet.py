import torch
import torch.nn.functional as F
import wandb
from tqdm import trange
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torch.cuda.amp import autocast
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names,truncated_noise_sample)
def save_image(image_tensor, filename):
    """Converts a tensor to an image and saves it."""
    img = image_tensor.clone().detach()
    img = img.to('cpu').float().numpy()
   # print(f"Image shape before transpose: {img.shape}")  # Debugging print
    if img.shape[0] == 1:
        img = np.squeeze(img, axis=0)  # Remove channel dimension if it is 1
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)  # Expand grayscale to 3D
    img = np.transpose(img, (1, 2, 0))
   # print(f"Image shape after transpose: {img.shape}")  # Debugging print
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename)


# Preprocess transform definition
preprocess_for_classifier = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.ToTensor(),  # Convert back to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Preprocess transform definition for saving images
def preprocess_for_saving(image_tensor):
    image = (image_tensor + 1) / 2  # Scale to [0, 1]
    image = image.squeeze(0)  # Remove batch dimension if it exists
    image = transforms.ToPILImage()(image.cpu())  # Convert to PIL Image
    return image

# Function to preprocess an image for VGG19
def preprocess_image(image_tensor):
    image = preprocess_for_saving(image_tensor)  # Convert tensor to PIL Image
    image = preprocess_for_classifier(image)  # Apply resizing and normalization for the classifier
    return image.unsqueeze(0).to(device)  # Add batch dimension


#run = wandb.init(project="Sinvad_fitness_BIGGAN_Imagenet1K")
# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained models
# 'biggan-deep-128' for 128x128 images
G = BigGAN.from_pretrained('biggan-deep-256').to(device)

# Load pretrained model
classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).to(device)
classifier.eval()

# Set models to evaluation mode
G.eval()
result_dir = "./result23_biggan_imagnet" # Directory to save the images
original_images_dir = os.path.join(result_dir,'original_images')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "or_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
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

# Generate the image
image_info = []
predictions = []
imgs_to_samp = 10
num_classes = 1000 
gen_num = 250
saved_images = 0
#max_attempts_per_image = 500  # Maximum attempts to generate a correct pizza image
pop_size = 25
best_left = 10
min_val = -0.799978375434876
max_val = 0.799991071224213
perturbation_size = 0.000159996944665909   # perturbation size
initial_perturbation_size = 0.000319993889331818  # Initial perturbation size
# Generate a random latent vector
latent_space = torch.randn(imgs_to_samp, 128).to(device)
# Create one-hot encoded class vectors for pizza class only
expected_label = 963  # Assuming 963 for pizza/ replace with 850 for teddy_bear class
class_vectors = torch.zeros(imgs_to_samp, num_classes).to(device)
class_vectors[:, expected_label] = 1
all_img_lst = []
truncation = 1
# Lists to store successful latents and seeds
for img_idx in trange(imgs_to_samp):
   # success = False  # Flag to check if correct image was generated
   # for attempt in range(max_attempts_per_image):
        seed = torch.randint(0, 10000, (1,)).item()
        torch.manual_seed(seed)  # Set the seed for reproducibility
        noise_vector = truncated_noise_sample(batch_size=1, dim_z=128, truncation=truncation, seed=seed)
        original_latent = torch.from_numpy(noise_vector).to(device)
        #original_latent = torch.randn(1,128).to(device)
        original_latent = torch.clamp(original_latent, min = min_val, max =max_val)
        # Create a new one-hot encoded class vector for pizza class only
        original_label = torch.zeros(1, num_classes).to(device)
        original_label[:, expected_label] = 1
        with torch.no_grad():
           # Generate image with the generator model
           original_image = G(original_latent, original_label, truncation).to(device)  # Adjust truncation as needed
           original_image = preprocess_image(original_image)
           # Classify generated image
        original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit).item()

        # Check if the generated image is classified as pizza
        if original_label == expected_label:
           # success = True
           # successful_latent = original_latent
           # successful_seed = seed
           # successful_label = original_label
            original_image_path = os.path.join(original_images_dir,f'original_image{saved_images}_X{original_label}.png')
            save_image(original_image.squeeze(0), original_image_path)
            original_image_np = original_image.detach().cpu().numpy() 
            # Define the filename that includes the label
            filename = f"original_image_{saved_images}_X_{original_label}.npy"

            # Define the full path for saving the file
            file_path = os.path.join(original_images_dir, filename)

            # Save the image data as a NumPy file
            np.save(file_path, original_image_np)
            saved_images +=1
            print(f'Success: Image {saved_images} classified correctly as expected class {original_label}.')

            ### Initialize optimization ###
            init_pop = [
                 original_latent.unsqueeze(0)
                 + initial_perturbation_size * torch.randn_like(original_latent)
                 for _ in range(pop_size)
            ]
            # Generate labels for the perturbed latent vectors
            pop_labels = torch.tensor([original_label] * pop_size)
            # Convert the labels to one-hot encoded class vectors
            pop_class_vectors = torch.nn.functional.one_hot(pop_labels, num_classes).float().to(device)
            best_fitness_score = np.inf
            best_image_tensor = None
            best_image_index = -1
            now_pop = init_pop
            prev_best = np.inf
            ###GA_algorithm###
            for g_idx in range(gen_num):

                indivs_lv = torch.cat(now_pop, dim=0).to(device)  # Concatenate to form a batch
                indivs_lv = indivs_lv.view(pop_size, -1)
                indivs_labels = pop_class_vectors.view(pop_size,-1).to(device)
                with torch.no_grad():
                    # Generate images
                    Gen_imgs = G(indivs_lv, indivs_labels, truncation= truncation).to(device) 
                    # Preprocess the generated images individually and stack them
                    Gen_imgs = torch.cat([preprocess_image(img) for img in Gen_imgs])


                # Compute fitness
                all_logits = classifier(Gen_imgs).squeeze().detach().cpu().numpy()
                fitness_scores = [
                    calculate_fitness(all_logits[k_idx], original_label)
                    for k_idx in range(pop_size)
                ]

                # Perform selection
                selected_indices = sorted(
                    range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
                )[-best_left:]
                # Finding the minimum fitness score in this generation
                current_min_index = np.argmin(fitness_scores)
                current_min_fitness = fitness_scores[current_min_index]

                # Update global minimum if the current score is lower
                if current_min_fitness < best_fitness_score:
                     best_fitness_score = current_min_fitness
                     best_image_tensor = Gen_imgs[current_min_index]
                     best_image_index = current_min_index

                now_best = np.min(fitness_scores)

                parent_pop = [now_pop[i] for i in selected_indices]

                # Perform crossover and mutation
                print("now_best " + str(now_best) + " average_best " + str(np.mean(fitness_scores)))
                # wandb.log({"ft_scores": now_best})

                if now_best < 0:
                   break
                elif now_best == prev_best:
                    perturbation_size *= 2
                else:
                    perturbation_size = initial_perturbation_size

                k_pop = []
                for k_idx in range(pop_size - best_left):
                      mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                      spl_idx = np.random.choice(128, size=1)[0]
                      k_gene = torch.cat(
                        [
                           parent_pop[mom_idx][:, :spl_idx],
                           parent_pop[pop_idx][:, spl_idx:],
                        ],
                        dim=1,
                       )  # Apply crossover to latent vector

                      # Mutation for latent vector
                      diffs = (k_gene != original_latent).float()
                      k_gene += perturbation_size * torch.randn_like(k_gene) * diffs

                      k_pop.append(k_gene)

                now_pop = parent_pop + k_pop
                prev_best = now_best
                now_pop = [torch.clamp(tensor, min= min_val, max= max_val) for tensor in now_pop]
            # After the loop, save the last image if it exists
            if best_image_tensor is not None:
                  # Assuming best_image_tensor is directly from the output (model or any processing)
                  best_image_tensor = best_image_tensor.to(device)  # Ensure tensor is on CPU
                  best_image_np = best_image_tensor.detach().cpu().numpy()  # Normalize to [0, 1] if not already
                  with torch.no_grad():
 
                      final_bound_logits = classifier(best_image_tensor.unsqueeze(0)).squeeze().detach().cpu().numpy()
                  predicted_best_label = np.argmax(final_bound_logits)

                  # Append the NumPy array to the list
                  all_img_lst.append(best_image_np)
                  image_path_np = os.path.join(result_dir,f'image{saved_images}_iteration_{g_idx + 1}_X{original_label}_Y{predicted_best_label}.npy',)
                  # Save the numpy array to a file
                  np.save(image_path_np, best_image_np)
                  best_image_pil_path = os.path.join(prtb_img_dir, f'image{saved_images}_iteration_{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png')
                  save_image(best_image_tensor.squeeze(0), best_image_pil_path)

            else:
                print("image is none")
        else: 
           # IF no match, simply skip to the next iteration (no image is saved or processed)
           print(f"Label mismatch for image {saved_images}:expected {expected_label},got {original_label}")
           

# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, 'bound_imgs_imagnet_cdcgan.npy'), all_imgs)

