import torch
import os
import torchvision
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from torchvision import transforms
from tqdm import trange
from vae_model import ConvVAE
from cifar10_classifier.model import VGGNet
from torchvision.utils import save_image as torch_save_image  # Import and alias to avoid naming conflicts

def save_image(tensor, filename):
    """
    Save a tensor as an image.
    Assumes tensor is normalized in the format [channels, height, width].
    """
    img = vutils.make_grid(tensor, normalize=True)  # Normalize ensures [0,1] range
    img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    # Now, transfer the tensor to CPU and convert it to numpy array for saving
    img = img.to('cpu', torch.uint8).numpy()  # Convert tensor to numpy array in [0,255] range
    img = Image.fromarray(img)
    img.save(filename)

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
image_size = 32 * 32 * 3
# Load the trained models
vae = ConvVAE(c_num=3, h_dim=4000, z_dim=1024).to(device)
classifier = VGGNet().to(device)
vae.load_state_dict(
    torch.load(
        "./cifar10_vae/weights/cifar10_convend.pth",
        map_location=device,
    )
)
classifier.load_state_dict(
    torch.load(
        "./cifar10_classifier/CIFAR10_cifar10_train.pynet.pth",
        map_location=device,
    )
)
# Set models to evaluation mode
vae.eval()
classifier.eval()
result_dir = "./result_vae_cifar10"  # Directory to save the images
# Subdirectory for original images
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "or_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(prtb_img_dir, exist_ok=True)

# Transforms: Convert image to tensor and normalize it
test_dataset = torchvision.datasets.CIFAR10(root='./datat', train=False, transform=transforms.ToTensor(), download=True)
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


gen_num = 250
pop_size = 25
best_left = 10
min_val = -4.15868806838989
max_val = 3.85224413871765
imgs_to_samp = 100
perturbation_size = 0.00801093220710754  # Default perturbation size
initial_perturbation_size = 0.0160218644142151  # Initial perturbation size
all_img_lst = []
saved_images = 0  # Counter for saved images

### Multi-image sample loop ###
for img_idx in trange(imgs_to_samp):  # Loop runs for 100 iterations
    # Load the image and class inside the loop
    for i, (x, x_class) in enumerate(test_data_loader):
        samp_img = x[0:1]  # Load one sample image
        samp_class = x_class[0].item()  # Get the class label

    # Perform VAE encoding and decoding after label match
    img_enc, _ = vae.encode(samp_img.to(device))
    original_lv = img_enc
    original_image = vae.decode(original_lv).view(-1, 3, 32, 32)

    # Predict the class for the original image using the classifier
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()

    # Only proceed if the predicted label matches the sample class
    if original_label == samp_class:
        print(f"True class: {samp_class}, Predicted class: {original_label}")

        # Save the original image
        original_image_np = original_image.squeeze().detach().cpu().numpy()
        filename = f"original_image_{saved_images}_X_{original_label}.npy"
        file_path = os.path.join(original_images_dir, filename)
        np.save(file_path, original_image_np)

        # Increment the saved_images counter
        saved_images += 1

        # Initialize genetic algorithm (GA)
        init_pop = [
            original_lv + initial_perturbation_size * torch.randn(1, 1024).to(device)
            for _ in range(pop_size)
        ]
        now_pop = init_pop
        prev_best = np.inf
        best_fitness_score = np.inf
        best_image_tensor = None
        best_image_index = -1

        # Genetic Algorithm (GA) Loop
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
            print(f"now_best {now_best}, Average_score {np.mean(fitness_scores)}")

            if now_best < 0:
                break
            elif now_best == prev_best:
                perturbation_size *= 2
            else:
                perturbation_size = initial_perturbation_size

            k_pop = []
            for k_idx in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(1024, size=1)[0]
                k_gene = torch.cat(
                    [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                    dim=1,
                )  # crossover

                # Mutation
                diffs = (k_gene != original_lv).float()
                k_gene += (
                    perturbation_size * torch.randn(k_gene.size()).to(device) * diffs
                )  # random adding noise only to different places

                k_pop.append(k_gene)

            # Current population is obtained by combining the parent pop and its offspring
            now_pop = parent_pop + k_pop
            prev_best = now_best
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

        # Save best image tensor
        mod_best_image_tensor = best_image_tensor.to(device)
        mod_best_image_np = best_image_tensor.cpu().detach().numpy()

        final_bound_logits = classifier(mod_best_image_tensor.unsqueeze(0))
        predicted_best_label = torch.argmax(final_bound_logits, dim=1).item()

        # Save the numpy file
        image_path_np = os.path.join(result_dir, f"image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.npy")
        np.save(image_path_np, mod_best_image_np)

        save_image(mod_best_image_tensor, os.path.join(prtb_img_dir, f'image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png'))

    else:
        # If no match is found
        print(f"No match for image {img_idx}: True class {samp_class}, Predicted class {original_label}")

# Save all generated images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, "bound_imgs_cifar10_vae.npy"), all_imgs)

print(f"Total matching images saved: {saved_images}") 
