import torch

import wandb
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from Svhn_classifier.model import VGGNet
from cdcgan_svhn_train import Generator


run = wandb.init(project="Sinvad_fitness_CDCGAN_svhn")
# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28 * 28 * 1
# Load the trained models
G = Generator(ngpu=1, nz=100, nc=1).to(device)  # Modify the Generator architecture
classifier = VGGNet().to(device)

# Load the weights of the models
G.load_state_dict(
    torch.load(
        "/home/maryam/Documents/SEDL/SINVAD/cdcgan_svhn/cdcgan_result/netG_epoch_99_ganbsvh.pth",
        map_location=device,
    )
)
classifier.load_state_dict(
    torch.load(
        "/home/maryam/Documents/SEDL/SINVAD/cdcgan_svhn/Svhn_classifier/Classifier/SVHN_vggnet.pth",
        map_location=device,
    )
)
# Set models to evaluation mode
G.eval()
classifier.eval()
result_dir = "./result_sh"


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
perturbation_size = 0.7  # Default perturbation size
initial_perturbation_size = 0.1  # Initial perturbation size
# mutation_perturbation_size = 0.1  # Mutation perturbation size
# Generate a random latent vector
latent_space = torch.randn(num_samples, 100, 1, 1).to(device)
random_labels = torch.randint(0, 10, (num_samples,)).to(device)
all_img_lst = []
for i in range(num_samples):
    # Generate the non-perturbed image
    original_latent = latent_space[i].unsqueeze(0)
    original_labelr = random_labels[i]
    # Generate the non-perturbed image using the generator
    original_image = G(original_latent, original_labelr.view(1))
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    # Calculate fitness for the original image
    fitness_original = calculate_fitness(original_logit, original_label)
    ### Initialize optimization ###

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
    init_label = torch.tensor([original_labelr] * pop_size)
    ###GA_algorithm###
    for g_idx in range(gen_num):
        indivs_lV = torch.cat(now_pop, dim=0).view(-1, 100, 1, 1)
        indivs_labels = init_label

        # Generate images using the generator
        Gen_imgs = G(indivs_lV, indivs_labels)
        Gen_imgs1 = Gen_imgs.detach().cpu().numpy()
        predicted_labels = torch.argmax(classifier(Gen_imgs), dim=1).cpu()

        all_img_lst.append(Gen_imgs1)
        # Calculate fitness scores for the current generation
        all_logits = classifier(Gen_imgs).squeeze().detach().cpu().numpy()
        fitness_scores = [
            calculate_fitness(all_logits[k_idx], original_label)
            for k_idx in range(pop_size)
        ]

        # Perform selection
        selected_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
        )[-best_left:]
        now_best = np.min(fitness_scores)

        parent_pop = [now_pop[i] for i in selected_indices]

        # Perform crossover and mutation
        print("now_best " + str(now_best) + " now_best " + str(np.mean(fitness_scores)))
        wandb.log({"ft_scores": now_best})

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
            k_gene += perturbation_size * torch.randn_like(k_gene) * diffs

            # Perform random matching to latent_images[i]
            interp_mask = binom_sampler.sample()
            k_gene = interp_mask * original_latent + (1 - interp_mask) * k_gene

            k_pop.append(k_gene)

        now_pop = parent_pop + k_pop
        prev_best = now_best
        init_label = predicted_labels

    mod_best_latent = parent_pop[-1].squeeze(0)
    # Adjusted expansion

    mod_best_image = G(mod_best_latent.unsqueeze(0), original_labelr.view(1))

    mod_best_label = torch.argmax(classifier(mod_best_image)).cpu().item()
    final_bound_img = (
        G(
            mod_best_latent.unsqueeze(0),
            torch.tensor([mod_best_label], dtype=torch.long, device=device),
        )
        .detach()
        .numpy()
    )
    all_img_lst.append(final_bound_img)

    # Convert the image to a PyTorch tensor
    final_bound_img_tensor = torch.from_numpy(final_bound_img).float().to(device)
    prediction = torch.argmax(classifier(final_bound_img_tensor)).item()
    predictions.append(prediction)

    # Convert the image data to grayscale (if it's not already)
    if final_bound_img.shape[1] == 1:
        final_bound_img = np.repeat(
            final_bound_img, 3, axis=1
        )  # Repeat grayscale values for R, G, and B

    # Transpose the dimensions to (32, 32, 3)
    transposed_img_data = final_bound_img[0].transpose(1, 2, 0)

    # Scale the image data from [0, 1] to [0, 255]
    scaled_img_data = (transposed_img_data * 255).astype(np.uint8)

    # Create the PIL Image
    image = Image.fromarray(scaled_img_data, mode="RGB")
    wandb_image = wandb.Image(image)

    # Log the image along with other relevant information
    wandb.log(
        {
            "Generated Image": wandb_image,
            "Expected Label X": original_label,
            "Predicted Label Y": predictions,
        }
    )

# Save all generated images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save("generated_images.npy", all_imgs)
