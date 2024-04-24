import torch

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
    torch.load("./cdcgan/weights/netG_epoch_59cdcgan.pth", map_location=device)
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
result_dir = "./result_cdcgan_mnist"  # Directory to save the images
# Subdirectory for original images
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)


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
num_samples = 100
gen_num = 250
pop_size = 25
best_left = 10
min_val = -4.78276300430298
max_val = 4.08758640289307


perturbation_size = 0.00887034940719605  # Default perturbation size
initial_perturbation_size = 0.0177406988143921 # Initial perturbation size
# Generate a random latent vector
latent_space = torch.randn(num_samples, 100, 1, 1).to(device)
random_labels = torch.randint(0, 10, (num_samples,)).to(device)
all_img_lst = []
for i in range(num_samples):
    # Generate the non-perturbed image
    original_latent = latent_space[i]
    original_labelr = random_labels[i]
    # Generate the non-perturbed image using the generator
    original_image = G(original_latent.view(1, 100, 1, 1), original_labelr.view(1))
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    
    # Save the original image
    original_image_pil.save(original_image_path)
    # Convert the tensor to a NumPy array
    original_image_np = original_image.squeeze().detach().cpu().numpy()

    # Define the filename that includes the label
    filename = f"original_image_{i}_X_{original_label}.npy"

    # Define the full path for saving the file
    file_path = os.path.join(original_images_dir, filename)

    # Save the image data as a NumPy file
    np.save(file_path, original_image_np)

    ### Initialize optimization ###
    init_pop = [
            original_latent.unsqueeze(0)
            + initial_perturbation_size * torch.randn_like(original_latent)
        for _ in range(pop_size)
    ]

    now_pop = init_pop
    prev_best = np.inf
    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5 * torch.ones(original_latent.size(), device=device)
    )
    best_fitness_score = np.inf
    best_image_tensor = None
    best_image_index = -1
    init_label = [original_label] * (pop_size)
    ###GA_algorithm###
    for g_idx in range(gen_num):
        # Combine current latent vectors and labels
        indivs_latent = torch.cat(now_pop, dim=0).view(-1, 100, 1, 1).to(device)
        indivs_labels = torch.tensor(init_label).to(device)

        # Generate images using the generator
        Gen_imgs = G(indivs_latent, indivs_labels)
        # Gen_imgs1 = Gen_imgs.detach().cpu().numpy()
        predicted_labels = torch.argmax(classifier(Gen_imgs), dim=1).to(device)
        # all_img_lst1.append(Gen_imgs1)
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
        # Selection
        selected_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
        )[-best_left:]
        
        # Correctly finding the minimum fitness score among selected individuals
       # selected_fitness_scores = [fitness_scores[i] for i in selected_indices]
        now_best = np.min(fitness_scores)

        parent_pop = [now_pop[i] for i in selected_indices]
        # Perform crossover and mutation
        print("now_best " + str(now_best) + " now_best " + str(np.mean(fitness_scores)))
        # wandb.log({"ft_scores": now_best})

        # Check condition to save image
        if now_best < 0:
            break
        elif now_best == prev_best:
            perturbation_size *= 2
        else:
            perturbation_size = initial_perturbation_size

        # Crossover and Mutation for latent vectors
        k_pop = []
        k_labels = []  # Initialize to store new labels
        for k_idx in range(pop_size - best_left):
            mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
            spl_idx = np.random.choice(100, size=1)[0]
            k_gene = torch.cat(
                [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:],],
                dim=1,
            )  # Apply crossover to latent vector

            # Mutation for latent vector
            diffs = (k_gene != original_latent).float()
            k_gene += perturbation_size * torch.randn_like(k_gene).to(device) * diffs
            # Perform random matching to latent_images[i]
           # interp_mask = binom_sampler.sample().to(device)
           # k_gene = interp_mask * original_latent + (1 - interp_mask) * k_gene
            k_pop.append(k_gene)

        now_pop = parent_pop + k_pop
        prev_best = now_best
        now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

    mod_best_image_tensor = best_image_tensor.to(device)
    mod_best_image_np = best_image_tensor.cpu().detach().numpy()

    final_bound_logits = classifier(mod_best_image_tensor.unsqueeze(0))
    predicted_best_label = torch.argmax(final_bound_logits, dim=1).item()
     # Define the path for saving the numpy file with detailed filename
    image_path_np = os.path.join(
        result_dir,
        f"image_{i}_iteration{g_idx}_X{original_label}_Y{predicted_best_label}.npy"
    )

    # Save the numpy array to a file
    np.save(image_path_np, mod_best_image_np)
    all_img_lst.append(mod_best_image_np)

# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, "bound_imgs_mnist_cdcgan.npy"), all_imgs)
