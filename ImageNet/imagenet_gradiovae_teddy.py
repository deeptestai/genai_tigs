import torch
import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from vae.imagenet_vae.model import Model
import torchvision.utils as vutils
from PIL import Image
from tqdm import trange
import zipfile
def save_image(tensor, filename):
    img = vutils.make_grid(tensor, normalize=True)
    img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img = img.to("cpu", torch.uint8).numpy()
    img = Image.fromarray(img)
    img.save(filename)

def denormalize_vgg_tensor(t):
    """
    Reverse ImageNet normalization for tensors with shape [3, H, W].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    t = t * std + mean
    return torch.clamp(t, 0, 1)  # Clip to [0,1] range to avoid weird colors



def load_model(checkpoint_path):
    model = Model(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 512

vae = Model(latent_dim=latent_dim).to(device)
checkpoint_path = "./vae/imagenet_vae/imagenet_vae_model_epoch_41.pth"
vae = load_model(checkpoint_path)
vae.eval()

#classifier = torch.hub.load("pytorch/vision:v0.10.0", "vgg16_bn", pretrained=True).to(device)
#classifier.eval()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((128, 128))
])

TEST_DIR = "./test_dataset1"
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

result_dir = "./Imagenet_img_teddy"
original_images_dir = os.path.join(result_dir, "original_images")
perturb_images_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(original_images_dir, exist_ok=True)
os.makedirs(perturb_images_dir, exist_ok=True)

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

def run_vae_tig_teddy(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier_choice,classifier_file):

    if classifier_choice == "VGG19bn":
        classifier = torch.hub.load("pytorch/vision:v0.10.0", "vgg19_bn", pretrained=True).to(device)
        classifier.eval()
        print( classifier," Default classifier loaded.")

    elif classifier_choice == "Upload Custom":
        if classifier_file is None:
            raise ValueError("Please upload a .jit TorchScript model file.")

        try:
            classifier = torch.jit.load(classifier_file.name, map_location=device)
            classifier.eval()
            print( " Custom TorchScript classifier loaded.")
        except Exception as e:
            raise ValueError(f" Failed to load custom classifier: {e}")

    else:
        raise ValueError("Unknown classifier selected.")


    all_img_lst = []
    final_iterations = []
    num_misclassified = 0
    num1_misclassified = 0
    total_images = 0
    all_gallery_items = []
    saved_image_paths = []
    status_rows = []
    saved_images = 0
    min_val = -5.22431135177612
    max_val = 5.62481260299683
    expected_label = 850

    for img_idx in trange(imgs_to_samp):
        for i, (x, _) in enumerate(test_loader):
            samp_img = x.to(device)
            break  # Only take one image

        # VAE encoding/decoding
        mean, logvar = vae.encode(samp_img)
        img_enc = vae.reparamatrize(mean, torch.exp(0.5 * logvar))
        original_lv = img_enc
        original_image = vae.decode(original_lv)

        original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
        original_label = original_logit.argmax().item()
        # Remove batch dimension if needed
        original_image = original_image.squeeze(0)  # -> Tensor [3, H, W]

        # Denormalize (ImageNet-style)
        denorm_orig = denormalize_vgg_tensor(original_image.cpu())

        # Convert to PIL and resize
        original_image_pil = transforms.ToPILImage()(denorm_orig).resize((128, 128))

        if original_label != expected_label:
            num1_misclassified += 1
            continue  # Skip mismatches
        #total_images += 1

        original_image_path = os.path.join(original_images_dir, f"original_image_{saved_images}_X{original_label}.png")
        #mod_image = original_image_pil.resize((image_size_selector, image_size_selector))
        original_image_pil.save(original_image_path)

        # Initialize GA
        init_pop = [original_lv + initial_perturb_size * torch.randn_like(original_lv).to(device) for _ in range(pop_size)]
        now_pop = init_pop
        prev_best = np.inf
        best_fitness_score = np.inf
        best_image_tensor = None

        for g_idx in range(gen_num): 
            indivs = torch.cat(now_pop, dim=0)
            dec_imgs = vae.decode(indivs)
            all_logits = classifier(dec_imgs).squeeze().detach().cpu().numpy()

            fitness_scores = [calculate_fitness(all_logits[k_idx], original_label) for k_idx in range(pop_size)]

            current_min_index = np.argmin(fitness_scores)
            current_min_fitness = fitness_scores[current_min_index]

            if current_min_fitness < best_fitness_score:
                best_fitness_score = current_min_fitness
                best_image_tensor = dec_imgs[current_min_index]

            best_idxs = sorted(range(len(fitness_scores)), key=lambda i_x: fitness_scores[i_x], reverse=True)[-best_left:]
            parent_pop = [now_pop[idx] for idx in best_idxs]

            now_best = np.min(fitness_scores)
            if now_best < 0: 
                break
            elif now_best == prev_best:
                perturb_size *= 2
            else:
                perturb_size = initial_perturb_size

            k_pop = []
            for k_idx in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(512, size=1)[0]
                k_gene = torch.cat([parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]], dim=1)
                diffs = (k_gene != original_lv).float()
                k_gene += perturb_size * torch.randn(k_gene.size()).to(device) * diffs
                k_pop.append(k_gene)

            now_pop = parent_pop + k_pop
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]
            prev_best = now_best


            #mod_image = original_image
            # Save perturbation result
        if best_image_tensor is not None:

           final_generation = g_idx + 1
           final_iterations.append(final_generation) 
           denorm_pert = denormalize_vgg_tensor(best_image_tensor.squeeze().cpu())
           perturbed_image_pil = transforms.ToPILImage()(denorm_pert).resize((128, 128))
               
           # Save perturbed image
           predicted_best_label = torch.argmax(classifier(best_image_tensor.unsqueeze(0).to(device)), dim=1).item()
           perturbed_image_path = os.path.join(perturb_images_dir, f'perturbed_image_{saved_images}_X{predicted_best_label}.png')
           perturbed_image_pil.save(perturbed_image_path)

           # Save row info
           status_rows.append([
               img_idx + 1,
               original_label,
               predicted_best_label,
               g_idx + 1  # or however you track iterations
           ])
           if predicted_best_label != original_label:
               num_misclassified += 1


           saved_images += 1

           misclassification_rate = (num_misclassified / saved_images) * 100 if saved_images > 0 else 0
           if final_iterations:
                 Avg_iterations = sum(final_iterations) / len(final_iterations)
           else:
                 Avg_iterations = 0  # or use None if you want to indicate 'no data'
           all_gallery_items.append((original_image_pil, f"Expected label: {original_label}"))
           all_gallery_items.append((perturbed_image_pil, f"Predicted label: {predicted_best_label}; iterations: {g_idx + 1}"))
           saved_image_paths.append(original_image_path)
           saved_image_paths.append(perturbed_image_path)




        # Yield progress to Gradio
        yield (
          f"Processing image {img_idx + 1} / {imgs_to_samp}|Misclassified seeds {num_misclassified}|% Misclassification: {misclassification_rate} |Avg Iterations:{Avg_iterations: .2f}",
          all_gallery_items, None , status_rows
        )
        
        zip_path = os.path.join(result_dir, "generated_pairs.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in saved_image_paths:
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname=arcname)
        # Final yield
        yield (
          f"Finished! Total saved images: {saved_images} | Misclassified seeds {num_misclassified}| % Misclassification: {misclassification_rate} |Avg Iterations:{Avg_iterations: .2f}",
          all_gallery_items,
          zip_path, status_rows
        )
       

