import torch
import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from vae_model_traink import Model
import torchvision.utils as vutils
from PIL import Image
from tqdm import trange
def save_image(tensor, filename):
    img = vutils.make_grid(tensor, normalize=True)
    img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img = img.to("cpu", torch.uint8).numpy()
    img = Image.fromarray(img)
    img.save(filename)
    print(f"Saved image: {filename}")

def load_model(checkpoint_path):
    model = Model(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 512
img_dim = 128
img_chn = 3

vae = Model(latent_dim=latent_dim).to(device)
checkpoint_path = "./imagenet_vae_model_epoch_41.pth"
vae = load_model(checkpoint_path)
vae.eval()

classifier = torch.hub.load("pytorch/vision:v0.10.0", "vgg19_bn", pretrained=True).to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((128, 128))
])

result_dir = "./Imagenet_img"
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "or_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(prtb_img_dir, exist_ok=True)

TEST_DIR = "./test1_pizza"
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
print("Data loader ready...")

def calculate_fitness(logit, label):
    expected_logit = logit[label]
    best_indices = np.argsort(-logit)[:2]
    best_index1, best_index2 = best_indices
    if best_index1 == label:
        best_but_not_expected = best_index2
    else:
        best_but_not_expected = best_index1
    new_logit = logit[best_but_not_expected]
    fitness = expected_logit - new_logit
    return fitness

image_info = []
gen_num = 250
pop_size = 25
best_left = 10
min_val = -5.22431135177612
max_val = 5.62481260299683
imgs_to_samp = 100
expected_label = 963       # 963 for pizza class /add 850 for teddy_class when teddy class is required
perturbation_size = 0.0010849123954773
initial_perturbation_size =0.00216982479095459 
all_img_lst = []
saved_images = 0  # Counter for saved images
### Multi-image sample loop ###
for img_idx in trange(imgs_to_samp):  # Loop runs for 100 iterations
    # Load the image and class inside the loop
    for i, (x, _) in enumerate(test_loader):
        # Move image to device
        samp_img = x.to(device)

    # VAE encoding and decoding
    mean, logvar = vae.encode(samp_img)
    img_enc = vae.reparamatrize(mean, torch.exp(0.5 * logvar))
    original_lv = img_enc
    original_image = vae.decode(original_lv)

    # Get the logits and label from classifier
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = original_logit.argmax().item()

    # Check if the label matches the expected class
    if original_label == expected_class:
        # Save the image if it matches
        save_image(original_image, os.path.join(img_pil_dir, f'original_image_{saved_images}_X{original_label}.png'))
        saved_images +=1
        # Process the image with Genetic Algorithm
        ### Initialize optimization ###
        init_pop = [
            original_lv + initial_perturbation_size * torch.randn_like(original_lv).to(device)
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
            dec_imgs = vae.decode(indivs)
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
                best_image_tensor = dec_imgs[current_min_index]
                best_image_index = current_min_index

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

            if now_best < 0:
                break
            elif now_best == prev_best:
                perturbation_size *= 2
            else:
                perturbation_size = initial_perturbation_size

            k_pop = []
            for k_idx in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(512, size=1)[0]
                k_gene = torch.cat(
                    [parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]],
                    dim=1,
                )  # crossover

                # Mutation
                diffs = (k_gene != original_lv).float()
                k_gene += (
                    perturbation_size * torch.randn(k_gene.size()).to(device) * diffs
                )  # random adding noise only to diff places
                k_pop.append(k_gene)

            # Current population is obtained combining the parent pop and its offspring
            now_pop = parent_pop + k_pop
            prev_best = now_best
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

        # Assuming best_image_tensor is tensor that needs to be processed
        mod_best_image_tensor = best_image_tensor.to(device)  # Move tensor to the appropriate device if not already
        mod_best_image_np = best_image_tensor.cpu().detach().numpy()  # Convert tensor to numpy array after moving to CPU

        # Assuming classifier is already defined and appropriate for the tensor as is
        final_bound_logits = classifier(mod_best_image_tensor.unsqueeze(0))  # Ensure tensor is in correct shape for classifier
        predicted_best_label = torch.argmax(final_bound_logits, dim=1).item()

        # Define the path for saving the numpy file with detailed filename
        image_path_np = os.path.join(
            result_dir,
            f"image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.npy"
        )

        # Save the numpy array to a file
        np.save(image_path_np, mod_best_image_np)

        all_img_lst.append(mod_best_image_np)

        save_image(mod_best_image_tensor, os.path.join(prtb_img_dir, f'image_{saved_images}_Iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png'))

    else:
        # If no match is found
        print(f"No match for image {img_idx}: True class {expected_class}, Predicted class {original_label}")

# Save the images as a numpy array
all_imgs = np.vstack(all_img_lst)
np.save(os.path.join(result_dir, "bound_imgs_imgnet_vae.npy"), all_imgs)

# Store the image info
#image_info.append((next_save_index, g_idx + 1, original_label, predicted_best_label))
