import torch

#import wandb
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int)


#run = wandb.init(project="Sinvad_fitness_BIGGAN_Imagenet1K")
# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 128 * 128 * 1
# Load the trained models
# 'biggan-deep-128' for 128x128 images
G = BigGAN.from_pretrained('biggan-deep-128').to(device)

# Load pretrained model
classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
classifier.eval()

# Set models to evaluation mode
G.eval()
classifier.eval()
result_dir = "./result_biggan_imagnet" # Directory to save the images
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
# Generate the image

image_info = []
predictions = []
num_samples = 100
gen_num = 500
pop_size = 25
best_left = 10
perturbation_size = 0.01 # Default perturbation size
initial_perturbation_size = 0.02 # Initial perturbation size
# mutation_perturbation_size = 0.1  # Mutation perturbation size
# Generate a random latent vector
latent_space = torch.randn(num_samples, 128, 1, 1).to(device)
class_vector = torch.zeros(1, 1000)
class_vector[0][963] = 1.0
class_vector = class_vector.to(device)
all_img_lst = []
for i in range(num_samples):
    # Generate the non-perturbed image
    class_vector = one_hot_from_int([963], batch_size=1) # Convert labels to one-hot encoded vectors
    class_vector = torch.tensor(class_vector, device=device).float()
    original_latent = latent_space[i].squeeze(-1).squeeze(-1)
    original_latent = original_latent.view(1, -1)  # This will reshape it to [1, 128]
    class_vector = class_vector.view(1, -1)  # This will reshape it to [1, 1000] just in case it's not
    #original_labelr = random_labels[i]
    # Generate the non-perturbed image using the generator
    #original_image = G(original_latent, original_labelr.view(1))
    # Generate the image
    with torch.no_grad():
     original_image = G(original_latent, class_vector, 1)  # 0.7 is a truncation value, which can be tuned
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

    # Generate initial class vectors randomly for the population
    init_labels_pop = torch.randint(0, 1000, (pop_size,)).to(device)  # 1000 classes
    init_class_vectors_pop = torch.nn.functional.one_hot(init_labels_pop, 1000).float().to(device)

    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5 * torch.ones(original_latent.size(), device=device)
    )
    now_pop = init_pop
    prev_best = np.inf
    #init_label = torch.tensor([original_labelr] * pop_size)
    ###GA_algorithm###
     # reshape the class vector

    for g_idx in range(gen_num):
        fitness_scores = []

        for idx, (latent, class_vector) in enumerate(zip(now_pop, init_class_vectors_pop)):
            # Generate images using the generator
            generated_image = G(latent.view(1, 128), class_vector.view(1, -1), 1)

            # Compute fitness
            all_logits = classifier(generated_image).squeeze().detach().cpu().numpy()
            fitness = calculate_fitness(all_logits, original_label)
            fitness_scores.append(fitness)

        # Perform selection
        selected_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
        )[-best_left:]
        now_best = np.min(fitness_scores)

        parent_pop = [now_pop[i] for i in selected_indices]
        # Use the same selected_indices to select the class vectors
        selected_class_vectors = [init_class_vectors_pop[i] for i in selected_indices]

        # Update the class vectors during crossover/mutation in a similar manner to the latents.

        # Perform crossover and mutation
        print("now_best " + str(now_best) + " average_best " + str(np.mean(fitness_scores)))
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

            # Perform random matching to latent_images[i]
            interp_mask = binom_sampler.sample()
            k_gene = interp_mask * original_latent + (1 - interp_mask) * k_gene

            k_pop.append(k_gene)

        now_pop = parent_pop + k_pop
        init_class_vectors_pop = selected_class_vectors
        prev_best = now_best

    mod_best_latent = parent_pop[-1].squeeze(0)
    # Adjusted expansion

    mod_best_class_vector = init_class_vectors_pop[-1].view(1,1000)

    # Generate the image with the best latent vector and its associated class vector
    mod_best_image = G(mod_best_latent, mod_best_class_vector, 1).detach()
    mod_best_label = torch.argmax(classifier(mod_best_image)).cpu().item()

    # Generate a new image using the best latent and the classifier's predicted class
    one_hot_class_vector = torch.nn.functional.one_hot(torch.tensor([mod_best_label], dtype=torch.long, device=device),
                                                       1000).float()

    final_bound_img_tensor = G(
        mod_best_latent,
        one_hot_class_vector,
        1  # truncation value
    ).detach()

    # Add the image to the list
    all_img_lst.append(final_bound_img_tensor.cpu().numpy())

    # Get the classifier's prediction on the new image
    prediction = torch.argmax(classifier(final_bound_img_tensor)).item()
    predictions.append(prediction)

    # Save the image as PNG
    # Convert the tensor to a numpy array and then to an image format
    image_array = final_bound_img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    image_array = ((image_array + 1) * 127.5).astype(np.uint8)
    image = Image.fromarray(image_array)

    #wandb_image = wandb.Image(image)

    # Log the image along with other relevant information
   # wandb.log(
   #    {
   #        "Generated Image": wandb_image,
   #        "Expected Label X": original_label,
   #        "Predicted Label Y": predictions,  #   "Fitness Score": now_best,
   #    }
   #)
    image_path = os.path.join(result_dir, f'image_{i}_X{original_label}_Y{prediction}.png')
    image.save(image_path)

    # Store the image info
    image_info.append((i, original_label, prediction))

# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, 'bound_imgs_imagnet_cdcgan.npy'), all_imgs)

# Save the image info
with open(os.path.join(result_dir, 'image_info.txt'), 'w') as f:
    f.write("Image Index, Expected Label X, Predicted Label Y\n")
    for img_info in image_info:
        f.write(f"{img_info[0]}, {img_info[1]}, {img_info[2]}\n")
misclassified_count = 0

# Iterate over the image info list
for img_info in image_info:
    expected_label = img_info[1]
    predicted_label = img_info[2]
    if predicted_label != expected_label:
        misclassified_count += 1

misclassification_percentage = (misclassified_count / len(image_info)) * 100

print(f"Misclassification Percentage: {misclassification_percentage:.2f}%")

