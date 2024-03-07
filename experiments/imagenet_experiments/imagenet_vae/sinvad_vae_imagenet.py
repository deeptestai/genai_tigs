import torch
import os
import json
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter
import wandb
import numpy as np  #### working on it
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import trange
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vae_model import  VAE, Encoder,  Decoder


def save_image(tensor, filename):
    """
    Save a tensor as an image
    """
    img = vutils.make_grid(tensor, normalize=True)  # Normalize ensures [0,1] range
    img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    # Now, transfer the tensor to CPU and convert it to numpy array for saving
    img = img.to(
        "cpu", torch.uint8
    ).numpy()  # Convert tensor to numpy array in [0,255] range
    img = Image.fromarray(img)
    img.save(filename)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model parameters
latent_dim = 1024
img_dim = 128
img_chn = 3

# Initialize the VAE model
encoder = Encoder(img_dim, img_chn, latent_dim)
decoder = Decoder(latent_dim, img_dim)
vae_model = VAE(encoder, decoder).to(device)
encoder.load_state_dict(
    torch.load(
        "./checkpoints/encoder_epoch_60.pth",
        map_location=device,
    )
)
decoder.load_state_dict(
    torch.load(
        "./checkpoints/decoder_epoch_60.pth",
        map_location=device,
    )
)
# Load pretrained model
classifier = torch.hub.load("pytorch/vision:v0.10.0", "vgg19_bn", pretrained=True).to(
    device
)
classifier.eval()
vae_model.eval()  # Set the model to evaluation mode
result_dir = "./result_vae_Imagenetsd"
os.makedirs(result_dir, exist_ok=True)
# Dataset and DataLoader setup
VAL_DIR = "/home/vincenzo.riccio/ddpm_diffuser/validation_dataset"  # Update this path
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ]
)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
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


image_info = []
predictions = []
gen_num = 500
pop_size = 25
best_left = 10
num_samples = 8
perturbation_size = 0.5  # Default perturbation size
initial_perturbation_size = 0.1  # Initial perturbation size
#latent_space = torch.randn(num_samples, 1024, 1, 1).to(device)
all_img_lst = []
for i in range(num_samples):
    for k, (x, x_class) in enumerate(val_loader):
        samp_img = x[0:1]
        samp_class = x_class[0].item()
        samp_img = F.resize(samp_img, [128,128])
    # Generate the non-perturbed image
    _, _, img_enc = vae_model.encoder(samp_img.to(device))
    # decoder
    original_image =vae_model.decoder(img_enc).detach()
    original_image1 = F.resize(original_image, [256, 256])
    original_logit = classifier(original_image1).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()
    # Assuming original_image is a single image tensor, add batch dimension if not present
    if original_image.dim() == 3:
            original_image = original_image.unsqueeze(0)
   
    # filename format to save image
    file_path = os.path.join(result_dir, f'org_label_{original_label}_{i}.png')

    # Save the image
    save_image(original_image, file_path)
    # Calculate fitness for the original image
    fitness_original = calculate_fitness(original_logit, original_label)
    all_img_lst = []

    ### Initialize optimization ###
    init_pop = [
        img_enc + initial_perturbation_size * torch.randn(1, 1024).to(device)
        for _ in range(pop_size)
    ]
    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5 * torch.ones(img_enc.size())
    )
    now_pop = init_pop
    prev_best = np.inf
    ### GA ###
    for g_idx in range(gen_num):
        indivs = torch.cat(now_pop, dim=0)
        dec_imgs = decoder(indivs)
       # decoded_img_resized 
        dec_imgs1 = torch.stack([F.resize(img, [256, 256]) for img in dec_imgs])
        all_logits = classifier(dec_imgs1).squeeze().detach().cpu().numpy()

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
        # wandb.log({"Fitness Score": now_best, "Sample Index": i, "Iteration": g_idx})
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
            diffs = (k_gene != img_enc).float()
            k_gene += (
                perturbation_size * torch.randn(k_gene.size()).to(device) * diffs
            )  # random adding noise only to diff places
            # random matching to latent_images[i]
            interp_mask = binom_sampler.sample()
            interp_mask = interp_mask.to(device)  # Move interp_mask to the same device as img_enc

            k_gene = interp_mask * img_enc + (1 - interp_mask) * k_gene
            k_pop.append(k_gene)

        # Current population is obtained combining the parent pop and its offspring
        now_pop = parent_pop + k_pop
        prev_best = now_best

    mod_best = parent_pop[-1].clone()
    final_bound_img =vae_model.decoder(parent_pop[-1]).detach().cpu().numpy()
    all_img_lst.append(final_bound_img)
    # Convert the image to a PyTorch tensor
    final_bound_img_tensor = torch.from_numpy(final_bound_img).float().to(device)
   # final_bound_img_tensor = F.interpolate(final_bound_img_tensor, size=(224, 224), mode='bicubic', align_corners=False)

   # final_bound_img_tensor =F.resize(final_bound_img_tensor, [128, 128])
    final_bound_img_tensor1 =F.resize(final_bound_img_tensor, [256, 256])
    prediction = torch.argmax(classifier(final_bound_img_tensor1)).item()
    predictions.append(prediction)
    save_image(
        final_bound_img_tensor,
        os.path.join(result_dir, f"image_{i}_X{original_label}_Y{prediction}.png"),
    )
    # Store the image info
    image_info.append((i, original_label, prediction))

# Save all generated images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save("generated_images_vae_Imagenet.npy", all_imgs)
# Save the image info
with open(os.path.join(result_dir, "image_info.txt"), "w") as f:
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
