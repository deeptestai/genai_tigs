import torch
from tqdm import trange
from torchvision.utils import save_image
import numpy as np
import os
from PIL import Image
from sa.cifar10_classifier.model import VGGNet
from cdcgan.cifar10_cdcgan.cdcgan_cifar10 import Generator
import torchvision.utils as vutils
from torchvision import transforms
import zipfile
def save_image(tensor, filename, size=None):
    """
    Save a tensor as an image, properly handling RGB (SVHN) format.
    
    Args:
        tensor (Tensor): Input tensor (C, H, W) with values 0-1.
        filename (str): Path to save the output image.
        size (tuple or int, optional): If given, resize to (size, size).
    """
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Only 3-channel (RGB) images are supported. Got tensor shape: {tensor.shape}")

    tensor = tensor.clamp(0, 1)  # Important! Clamp between [0,1]
    img = tensor.permute(1, 2, 0)  # (C, H, W) → (H, W, C)
    img = (img * 255).to(torch.uint8).cpu().numpy()

    img_pil = Image.fromarray(img, mode="RGB")

    if size is not None:
        img_pil = img_pil.resize((size, size))

    img_pil.save(filename)

#full Min and Max value
min_val = -4.4389796257019
max_val = 4.66634130477905

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Model once globally
img_size = 32 * 32 * 1
G = Generator(ngpu=1, nz=100, nc=3).to(device)   # Modify the Generator architecture
G.load_state_dict(torch.load("./cdcgan/cifar10_cdcgan/weights/netG_epoch_699.pth", map_location=device,weights_only=True))
G.eval()



def calculate_fitness(logit, label):
    expected_logit = logit[label]
    best_indices = np.argsort(-logit)[:2]
    best_index1, best_index2 = best_indices
    best_but_not_expected = best_index2 if best_index1 == label else best_index1
    new_logit = logit[best_but_not_expected]
    return expected_logit - new_logit

def run_gan_tig2(gen_num, pop_size, best_left,perturbation_size,initial_perturbation_size, imgs_to_samp, classifier_choice,classifier_file):
    if classifier_choice == "VGGNET":
        classifier = VGGNet().to(device)
        classifier.load_state_dict(
            torch.load("./sa/cifar10_classifier/model/CIFAR10_cifar10_train.pynet.pth", map_location=device,weights_only=True)
        )
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


    result_dir = "./result23_cdcgan_cifar10"
    os.makedirs(result_dir, exist_ok=True)
    original_images_dir = os.path.join(result_dir, "original_images")
    os.makedirs(original_images_dir, exist_ok=True)
    perturb_images_dir = os.path.join(result_dir, "perturb_images")
    os.makedirs(perturb_images_dir, exist_ok=True)
   # perturbation_size = 0.000887034940719605
   # initial_perturbation_size = 0.00177406988143921

    latent_space = torch.randn(imgs_to_samp, 100, 1, 1).to(device)
    random_labels = torch.randint(0, 10, (imgs_to_samp,)).to(device)

    saved_images = 0
    final_iterations = []
    num_misclassified = 0
    num1_misclassified = 0
    total_images = 0
    all_gallery_items = []
    saved_image_paths = []
    status_rows = []


    for img_idx in trange(imgs_to_samp):
        original_latent = latent_space[img_idx].unsqueeze(0)
        original_label_tensor = random_labels[img_idx]
        expected_label = original_label_tensor.item()

        original_image = G(original_latent, original_label_tensor.view(1))
        original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit).item()

        if original_label != expected_label:
            num1_misclassified += 1
            continue  # Skip mismatches
        total_images += 1


        save_tensor = (original_image.squeeze() + 1) / 2  # [-1,1] → [0,1]
        save_tensor = save_tensor.clamp(0, 1)

        original_image_pil =transforms.ToPILImage()(save_tensor.cpu())
        original_image_path = os.path.join(original_images_dir, f"original_image_{saved_images}_X{original_label}.png")
        #mod_image = original_image_pil.resize((image_size_selector, image_size_selector))
        original_image_pil.save(original_image_path)

        

       #mod_image = mod_image.resize((image_size_selector, image_size_selector))
        # Init population
        init_pop = [
            torch.clamp(
                original_latent.unsqueeze(0) +
                initial_perturbation_size * torch.randn_like(original_latent),
                min=min_val, max=max_val
            ) for _ in range(pop_size)
        ]
        now_pop = init_pop
        prev_best = np.inf
        best_fitness_score = np.inf
        best_image_tensor = None

        for g_idx in range(gen_num):
            indivs_lV = torch.cat(now_pop, dim=0).view(-1, 100, 1, 1).to(device)
            indivs_labels = torch.tensor([original_label] * pop_size).to(device)
            Gen_imgs = G(indivs_lV, indivs_labels)
            all_logits = classifier(Gen_imgs).squeeze().detach().cpu().numpy()
            fitness_scores = [calculate_fitness(all_logits[k], original_label) for k in range(pop_size)]

            min_index = np.argmin(fitness_scores)
            min_fitness = fitness_scores[min_index]
            if min_fitness < best_fitness_score:
                best_fitness_score = min_fitness
                best_image_tensor = Gen_imgs[min_index].cpu().detach()

            selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[-best_left:]
            parent_pop = [now_pop[i] for i in selected_indices]
            now_best = np.min(fitness_scores)

            if now_best < 0:
                final_generation = g_idx + 1
                final_iterations.append(final_generation) 
                break
            elif now_best == prev_best:
                perturbation_size *= 2
            else:
                perturbation_size = initial_perturbation_size

            k_pop = []
            for k_idx in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(100, size=1)[0]
                k_gene = torch.cat([
                    parent_pop[mom_idx][:, :spl_idx],
                    parent_pop[pop_idx][:, spl_idx:]
                ], dim=1)
                diffs = (k_gene != original_latent).float()
                k_gene += perturbation_size * torch.randn_like(k_gene).to(device) * diffs
                k_pop.append(k_gene)

            now_pop = parent_pop + k_pop
            prev_best = now_best
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]

        if best_image_tensor is not None:
            
            final_generation = g_idx + 1
            final_iterations.append(final_generation)

            pert_img_np = best_image_tensor.squeeze().detach().cpu().numpy()
            # (Optional) Denormalize if output was in [-1, 1] range
            pert_img_tensor = (pert_img_np + 1) / 2
            pert_img_np = pert_img_tensor.clip(0, 1)
            if pert_img_np.ndim == 3 and pert_img_np.shape[0] == 3:
               # (3, H, W) → (H, W, 3) for SVHN RGB
               pert_img_np = np.transpose(pert_img_np, (1, 2, 0))
               perturbed_image_pil = Image.fromarray((pert_img_np * 255).astype(np.uint8), mode="RGB")
            else:
               raise ValueError(f"Only 3-channel (RGB) images are supported. Got shape: {pert_img_np.shape}")

            predicted_best_label = torch.argmax(classifier(best_image_tensor.unsqueeze(0).to(device)), dim=1).item()
            perturbed_image_path = os.path.join(perturb_images_dir, f"image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png")
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
          f"Processing image {img_idx + 1} / {imgs_to_samp}| Misclassified seeds {num_misclassified}|% Misclassification: {misclassification_rate}|Avg Iteratins{Avg_iterations}",
          all_gallery_items, None, status_rows
        )
        zip_path = os.path.join(result_dir, "generated_pairs.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in saved_image_paths:
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname=arcname)
        # Final yield
        yield (
          f"Finished! Total saved images: {saved_images} | Misclassified seeds {num_misclassified}| % Misclassification: {misclassification_rate}|Avg Iteratins{Avg_iterations}",

          all_gallery_items,
          zip_path, status_rows
        )


