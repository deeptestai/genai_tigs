import torch

import wandb
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from sa.model import MnistClassifier
from cdcgan.cdcgan_mnist import Generator

run = wandb.init(project="Sinvad_fitness_CDCGAN_Mnist2")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28 * 28 * 1
# Load the trained models
G = Generator(ngpu=1, nz=100, nc=1).to(device)  # Modify the Generator architecture
classifier = MnistClassifier(img_size=img_size).to(device)

# Load the weights of the models
G.load_state_dict(torch.load("./cdcgan/netG_epoch_59cdcgan.pth", map_location=device))
classifier.load_state_dict(
    torch.load(
        "/home/maryam/Documents/SEDL/SINVAD/sa/models/MNIST_conv_classifier.pth",
        map_location=device,
    )
)
# Set models to evaluation mode
G.eval()
classifier.eval()
result_dir = "./result_mn"


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
num_samples = 1
gen_num = 500
pop_size = 25
best_left = 10
perturbation_size = 0.02  # Default perturbation size
initial_perturbation_size = 0.01  # Initial perturbation size
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
    # Calculate fitness for the original image
    fitness_original = calculate_fitness(original_logit, original_label)

    # Initialize optimization for both latent vectors and labels
    init_pop_latent = [
        original_latent.unsqueeze(0)
        + initial_perturbation_size * torch.randn_like(original_latent)
        for _ in range(pop_size)
    ]
    init_pop_labels = [
        torch.randint(0, 10, (num_samples,)).to(device) for _ in range(pop_size)
    ]  # Random labels

    now_pop_latent = init_pop_latent
    now_pop_labels = init_pop_labels

    prev_best = np.inf
    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5 * torch.ones(original_latent.size(), device=device)
    )

    ###GA_algorithm###
    for g_idx in range(gen_num):
        # Combine current latent vectors and labels
        indivs_latent = torch.cat(now_pop_latent, dim=0).view(-1, 100, 1, 1)
        indivs_labels = torch.cat(now_pop_labels, dim=0).squeeze()

        # Generate images using the generator
        Gen_imgs = G(indivs_latent, indivs_labels)
        Gen_imgs1 = Gen_imgs.detach().cpu().numpy()
        predicted_labels = torch.argmax(classifier(Gen_imgs), dim=1).cpu()

        all_img_lst.append(Gen_imgs1)
        # Calculate fitness scores for the current generation
        all_logits = classifier(Gen_imgs).squeeze().detach().cpu().numpy()
        fitness_scores = [
            calculate_fitness(all_logits[k_idx], original_label)
            for k_idx in range(pop_size)
        ]
        # Selection
        selected_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
        )[-best_left:]
        now_best = np.min(fitness_scores)

        parent_pop_latent = [now_pop_latent[i] for i in selected_indices]
        parent_pop_labels = [now_pop_labels[i] for i in selected_indices]

        # Perform crossover and mutation
        print("now_best " + str(now_best) + " now_best " + str(np.mean(fitness_scores)))
        wandb.log({"ft_scores": now_best})

        if now_best < 0:
            break
        elif now_best == prev_best:
            perturbation_size *= 2
        else:
            perturbation_size = initial_perturbation_size

        # Crossover and Mutation for both latent vectors and labels
        k_pop_latent = []
        k_pop_labels = []
        for k_idx in range(pop_size - best_left):
            mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)

            # Crossover for latent vectors
            spl_idx = np.random.choice(100, size=1)[0]
            k_gene_latent = torch.cat(
                [
                    parent_pop_latent[mom_idx][:, :spl_idx],
                    parent_pop_latent[pop_idx][:, spl_idx:],
                ],
                dim=1,
            )

            # Mutation for latent vectors
            diffs = (k_gene_latent != original_latent).float()
            k_gene_latent += perturbation_size * torch.randn_like(k_gene_latent) * diffs

            # Crossover for labels (simple averaging and rounding)
            k_gene_label = (
                parent_pop_labels[mom_idx] + parent_pop_labels[pop_idx]
            ) // 2

            # Mutation for labels
            if torch.rand(1).item() < 0.1:  # 10% mutation chance for labels
                k_gene_label = torch.randint(0, 10, (num_samples,)).to(device)

            k_pop_latent.append(k_gene_latent)
            k_pop_labels.append(k_gene_label)

        now_pop_latent = parent_pop_latent + k_pop_latent
        now_pop_labels = parent_pop_labels + k_pop_labels

        prev_best = now_best

    mod_best_latent = parent_pop_latent[-1].unsqueeze(0)
    mod_best_label = parent_pop_labels[-1].unsqueeze(0)
    # Adjusted expansion

    final_bound_img = G(mod_best_latent, mod_best_label).cpu().detach().numpy()
    all_img_lst.append(final_bound_img)

    # Convert the image to a PyTorch tensor
    final_bound_img_tensor = torch.from_numpy(final_bound_img).to(device)
    prediction = torch.argmax(classifier(final_bound_img_tensor)).item()
    predictions.append(prediction)

    # Save the image as PNG
    image = Image.fromarray((final_bound_img[0][0] * 255).astype(np.uint8), mode="L")
    wandb_image = wandb.Image(image)

    # Log the image along with other relevant information
    wandb.log(
        {
            "Generated Image": wandb_image,
            "Expected Label X": original_label,
            "Predicted Label Y": predictions,
            "Fitness Score": now_best,
        }
    )

    # Save all generated images as a numpy array
    all_imgs = np.vstack(all_img_lst)
    np.save("generated_images.npy", all_imgs)
