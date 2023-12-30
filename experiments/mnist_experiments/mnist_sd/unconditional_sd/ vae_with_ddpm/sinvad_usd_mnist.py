import torch
#import wandb
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import trange
from torchvision import datasets
import os

from sa.model import MnistClassifier
from vae.model import VAE

#run = wandb.init(project="sinvad_fitness_sd_mnist")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28 * 28 * 1
torch.no_grad()  # since nothing is trained here

### Prep (e.g. Load Models) ###
vae = VAE(img_size=28 * 28, h_dim=1600, z_dim=400).to(device)
classifier = MnistClassifier(img_size=img_size).to(device)
vae.load_state_dict(
    torch.load(
        "./vae/models/MNIST_EnD.pth",
        map_location=device,
    )
)
vae.eval()
classifier.load_state_dict(
    torch.load(
        "/home/maryam/Documents/SEDL/SINVAD/sa/models/MNIST_conv_classifier.pth",
        map_location=device,
    )
)
classifier.eval()
result_dir = "./result_usd_mnist" # Directory to save the images
os.makedirs(result_dir, exist_ok=True)
print("models loaded...")
# Transforms: Convert image to tensor and normalize it
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),  # This converts to grayscale
        transforms.ToTensor(),
    ]
)

dataset = datasets.ImageFolder(
    root="./mnist_ddpm_gen_img",
    transform=transform,
)

loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=1, shuffle=True
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
gen_num = 500
pop_size = 25
best_left = 10
imgs_to_samp = 100
perturbation_size = 0.1  # Default perturbation size
initial_perturbation_size = 0.7 # Initial perturbation size
predictions = []
all_img_lst = []
image_info = []
random_indices = np.random.permutation(len(dataset))

for img_idx in trange(imgs_to_samp):
    samp_img, samp_class = dataset[random_indices[img_idx]]

    img_enc, _ = vae.encode(samp_img.view(-1, img_size).to(device))
    original_lv = img_enc
    original_image = vae.decode(original_lv).view(-1, 1, 28, 28)
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()
    # Calculate fitness for the original image
    fitness_original = calculate_fitness(original_logit, original_label)

    ### Initialize optimization ###
    init_pop = [
        original_lv + initial_perturbation_size * torch.randn(1, 400).to(device)
        for _ in range(pop_size)
    ]
    now_pop = init_pop
    prev_best = np.inf
    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5 * torch.ones(original_lv.size())
    )

    ### GA ###
    for g_idx in range(gen_num):
        indivs = torch.cat(now_pop, dim=0)
        dec_imgs = vae.decode(indivs).view(-1, 1, 28, 28)
        all_logits = classifier(dec_imgs).squeeze().detach().cpu().numpy()

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
        #wandb.log({"Fitness Score": now_best, "Sample Index": i, "Iteration": g_idx})
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
            diffs = (k_gene != img_enc).float()
            k_gene += (
                perturbation_size * torch.randn(k_gene.size()).to(device) * diffs
            )  # random adding noise only to diff places
            # random matching to img_enc
            interp_mask = binom_sampler.sample().to(device)
            k_gene = interp_mask * img_enc + (1 - interp_mask) * k_gene

            k_pop.append(k_gene)
        now_pop = parent_pop + k_pop
        prev_best = now_best

    mod_best = parent_pop[-1].clone()
    final_bound_img = vae.decode(parent_pop[-1]).detach().cpu().numpy()
    all_img_lst.append(final_bound_img)
    # Convert the image to a PyTorch tensor
    final_bound_img = final_bound_img.reshape(1, 28, 28)
    final_bound_img_tensor = torch.from_numpy(final_bound_img).float().to(device)
    #final_bound_img_tensor = final_bound_img_tensor.view(1, 1, 28, 28)
    prediction = torch.argmax(classifier(final_bound_img_tensor)).item()
    predictions.append(prediction)

    # Save the image as PNG
    #final_bound_img_2d = final_bound_img.reshape(1, 28, 28)
    #image = Image.fromarray((final_bound_img_2d[0] * 255).astype(np.uint8), mode="L")
    #wandb_image = wandb.Image(image)

    # Log the image along with other relevant information
    #wandb.log(
    #    {
    #        "Generated Image": wandb_image,
    #       "Expected Label X": original_label,
    #        "Predicted Label Y": predictions,  # "Fitness Score": now_best,
    #   }
   # )
    # Save the image as PNG
    image = Image.fromarray((final_bound_img[0] * 255).astype(np.uint8), mode='L')
    image_path = os.path.join(result_dir, f'image_{img_idx}_X{original_label}_Y{prediction}.png')
    image.save(image_path)

    # Store the image info
    image_info.append((img_idx, original_label, prediction))

# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, 'output_vae_ddpm_img.npy'), all_imgs)

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

