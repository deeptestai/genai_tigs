import torch
# import wandb
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import trange
from torchvision import datasets
import torchvision
import os

from sa.model import MnistClassifier
from vae.model import VAE

# run = wandb.init(project="sinvad_fitness_sd_mnist")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28 * 28 * 1
torch.no_grad()  # since nothing is trained here

### Prep (e.g. Load Models) ###
vae = VAE(img_size=28 * 28, h_dim=1600, z_dim=400).to(device)
classifier = MnistClassifier(img_size=img_size).to(device)
vae.load_state_dict(torch.load("./vae/models/MNIST_EnD.pth", map_location=device,))
vae.eval()
classifier.load_state_dict(
    torch.load(
        "./sa/models/MNIST_conv_classifier.pth",
        map_location=device,
    )
)
classifier.eval()

result_dir = "./result23_vae_mnist"  # Directory to save the images
# Subdirectory for original images
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)

print("models loaded...")

# Transforms: Convert image to tensor and normalize it
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=True
)
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


### GA Params ###
gen_num = 250
pop_size = 25
best_left = 10
min_val = -3.86494088172913
max_val = 3.45633792877197
imgs_to_samp = 100
perturbation_size = 0.0073212788105011  # Default perturbation size (high_mutation value)
initial_perturbation_size = 0.0146425576210022  # Initial perturbation size (high mutation value)
predictions = []
all_img_lst = []
image_info = []
saved_images = 0  # Counter for saved images

for img_idx in trange(imgs_to_samp):
    ### Sample image ###
    for i, (x, x_class) in enumerate(test_data_loader):
        samp_img = x[0:1]
        samp_class = x_class[0].item()

    img_enc, _ = vae.encode(samp_img.view(-1, img_size).to(device))
    original_lv = img_enc
    original_image = vae.decode(original_lv).view(-1, 1, 28, 28)
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()

    # Only save and proceed with GA if the original label matches the sample class
    if original_label == samp_class:
        print(f"True class: {samp_class}, Predicted class: {original_label}")

        # Convert the tensor to a PIL Image
        original_image_pil = transforms.ToPILImage()(original_image.squeeze().cpu())

        # Define the path for saving the original image
        original_image_path = os.path.join(
            original_images_dir, f"original_image_{saved_images}_X{original_label}.png"
        )

        # Save the original image
        original_image_pil.save(original_image_path)

        # Convert the tensor to a NumPy array
        original_image_np = original_image.squeeze().detach().cpu().numpy()

        # Define the filename that includes the label
        filename = f"original_image_{saved_images}_X_{original_label}.npy"

        # Define the full path for saving the file
        file_path = os.path.join(original_images_dir, filename)

        # Save the image data as a NumPy file
        np.save(file_path, original_image_np)

        # Increment the saved_images counter when an image is saved
        saved_images += 1

        ### Initialize optimization (GA) ###
        init_pop = [
            original_lv + initial_perturbation_size * torch.randn(1, 400).to(device)
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
            dec_imgs = vae.decode(indivs).view(-1, 1, 28, 28)
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
            # Consider the lowest fitness score
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
                spl_idx = np.random.choice(400, size=1)[0]
                k_gene = torch.cat(
                    [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                    dim=1,
                )  # crossover

                # mutation
                diffs = (k_gene != original_lv).float()
                k_gene += (
                    perturbation_size * torch.randn(k_gene.size()).to(device) * diffs
                )  # random adding noise only to diff places

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
            f"image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.npy"
        )

        # Save the numpy array to a file
        np.save(image_path_np, mod_best_image_np)

        all_img_lst.append(mod_best_image_np)

    else:
        print(f"Mismatch: True class: {samp_class}, Predicted class: {original_label}")

# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, "bound_imgs_mnist_vae.npy"), all_imgs)

print(f"Total saved images: {saved_images}")
