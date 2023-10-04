import torch
import os
import wandb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dcgan.cifar10gan import Generator
from cifar10_classifier.model import CifarClassifier

run = wandb.init(project="Sinvad_latent_beased_Cifar10DCGAN")

# Mapping of CIFAR-10 numerical labels to English labels
cifar10_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32 * 32 * 1
# Load the trained models
G = Generator(ngpu=1, nz=100, nc=3).to(device)  # Modify the Generator architecture
classifier = CifarClassifier(img_size=img_size).to(device)

# Load the weights of the models
G.load_state_dict(
    torch.load(
        "./cifar10_dcgan/net1G_epochcifar10_999.pth",
        map_location=device,
    )
)
classifier.load_state_dict(
    torch.load(
        "./cifar_10_classifier/CIFAR10_classifier.pth",
        map_location=device,
    )
)
# Set models to evaluation mode
G.eval()
classifier.eval()
result_dir = "./result_mn"
os.makedirs(result_dir, exist_ok=True)


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
perturbation_size = 0.05  # Default perturbation size
initial_perturbation_size = 0.07  # Initial perturbation size
# mutation_perturbation_size = 0.1  # Mutation perturbation size

# Generate a random latent vector
latent_space = torch.randn(num_samples, 100, 1, 1).to(device)
all_img_lst = []
for i in range(num_samples):
    # Generate the non-perturbed image
    original_lv = latent_space[i].unsqueeze(0)
    original_image = G(original_lv)
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()
    # Calculate fitness for the original image
    fitness_original = calculate_fitness(original_logit, original_label)
    ### Initialize optimization ###
    init_pop = [
        original_lv.unsqueeze(0)
        + initial_perturbation_size * torch.randn_like(original_lv)
        for _ in range(pop_size)
    ]
    now_pop = init_pop
    prev_best = np.inf
    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5 * torch.ones(original_lv.size(), device=device)
    )
    now_best_values = []
    ###GA_algorithm###
    for g_idx in range(gen_num):
        indivs_lv = torch.cat(now_pop, dim=0).view(-1, 100, 1, 1)
        gen_imgs = G(indivs_lv)
        gen_imgs1 = gen_imgs.detach().numpy()
        # Add the first generated image to the results
        all_img_lst.append(gen_imgs1)

        all_logits = classifier(gen_imgs).squeeze().detach().cpu().numpy()
        fitness_scores = [
            calculate_fitness(all_logits[k_idx], original_label)
            for k_idx in range(pop_size)
        ]

        # Perform selection
        best_idxs = sorted(
            range(len(fitness_scores)),
            key=lambda i_x: fitness_scores[i_x],
            reverse=True,
        )[-best_left:]

        # Consider the lowest fitness score
        now_best = np.min(fitness_scores)

        parent_pop = [now_pop[idx] for idx in best_idxs]

        # Perform crossover and mutation
        print(
            "now_best "
            + str(now_best)
            + ", Average_score "
            + str(np.mean(fitness_scores))
        )
        # Log fitness scores
        wandb.log({"Fitness Score": now_best, "Sample Index": i, "Iteration": g_idx})

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
                [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                dim=1,
            )  # crossover

            # Mutation
            diffs = (k_gene != original_lv).float()
            k_gene += (
                perturbation_size * torch.randn(k_gene.size()) * diffs
            )  # random adding noise only to diff places
            # random matching to latent_images[i]
            interp_mask = binom_sampler.sample()
            k_gene = interp_mask * original_lv + (1 - interp_mask) * k_gene
            k_pop.append(k_gene)

        # Current population is obtained combining the parent pop and its offspring
        now_pop = parent_pop + k_pop
        prev_best = now_best

    mod_best = parent_pop[-1].squeeze(0)
    final_bound_img = G(mod_best).detach().numpy()
    all_img_lst.append(final_bound_img)
    # Convert the image to a PyTorch tensor
    final_bound_img = final_bound_img.reshape(3, 32, 32)
    final_bound_img_tensor = torch.from_numpy(final_bound_img).float()
    prediction = torch.argmax(classifier(final_bound_img_tensor)).item()
    predictions.append(prediction)

    # Save the image as PNG
    # Convert a PyTorch tensor to a numpy array
    image_array = final_bound_img_tensor.cpu().numpy().transpose(1, 2, 0)
    image_array = ((image_array + 1) * 127.5).astype(np.uint8)
    image = Image.fromarray(image_array)
    image_path = os.path.join(
        result_dir, f"image_{g_idx}_X{original_label}_Y{prediction}.png"
    )
    image.save(image_path)

    # Store the image info
    image_info.append((g_idx, original_label, prediction))

# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, "bound_imgs_MNIST.npy"), all_imgs)

# Save the image info
with open(os.path.join(result_dir, "image_info.txt"), "w") as f:
    f.write("Image Index, Expected Label X, Predicted Label Y\n")
    for img_info in image_info:
        f.write(f"{img_info[0]}, {img_info[1]}, {img_info[2]}\n")
