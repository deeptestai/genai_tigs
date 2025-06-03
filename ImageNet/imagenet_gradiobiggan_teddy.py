import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import trange
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image, resize
#import gradio as gr
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample)
import zipfile
def denormalize_vgg_tensor(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

def save_image(image_tensor, filename):
    img = image_tensor.clone().detach()
    img = img.to('cpu').float().numpy()
    if img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.transpose(img, (1, 2, 0))
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename)
def denormalize_vgg_tensor(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

preprocess_for_classifier = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_for_saving(image_tensor):
    image = (image_tensor + 1) / 2
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image.cpu())
    return image

def preprocess_image(image_tensor):
    image = preprocess_for_saving(image_tensor)
    image = preprocess_for_classifier(image)
    return image.unsqueeze(0).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained models
G = BigGAN.from_pretrained('biggan-deep-256').to(device)
G.eval()
#classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).to(device)
#classifier.eval()

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

def run_biggan_tig_teddy(imgs_to_samp, gen_num, pop_size, best_left, perturb_size, initial_perturb_size,classifier_choice, truncation,classifier_file):
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

    min_val = -1.99990940093994
    max_val = 1.99996650218964
    num_classes = 1000
    image_size_selector = 224
    result_dir = "./result_biggan_gradio"
    original_images_dir = os.path.join(result_dir, "original_images")
    perturb_images_dir = os.path.join(result_dir, "perturb_images")
    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(perturb_images_dir, exist_ok=True)
    gallery_pairs = []
    all_img_lst = []
    final_iterations = []
    num_misclassified = 0
    num1_misclassified = 0
    total_images = 0
    all_gallery_items = []
    saved_image_paths = []
    status_rows = []
    saved_images = 0

    expected_label = 850
    for img_idx in trange(imgs_to_samp):
        seed = torch.randint(0, 10000, (1,)).item()
        torch.manual_seed(seed)
        noise_vector = truncated_noise_sample(batch_size=1, dim_z=128, truncation=truncation, seed=seed)
        original_latent = torch.from_numpy(noise_vector).to(device)
        original_latent = torch.clamp(original_latent, min=min_val, max=max_val)

        label_tensor = torch.zeros(1, num_classes).to(device)
        label_tensor[:, expected_label] = 1

        with torch.no_grad():
            original_image = G(original_latent, label_tensor, truncation).to(device)
            original_image = preprocess_image(original_image)

        original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit)

        if original_label != expected_label:
            num1_misclassified += 1
            continue  # Skip mismatches
        
        denorm_orig = denormalize_vgg_tensor(original_image.squeeze()).cpu()
        original_image_pil = transforms.ToPILImage()(denorm_orig)
        original_image_path = os.path.join(original_images_dir, f"original_image_{saved_images}_X{original_label}.png")
        #mod_image = original_image_pil.resize((image_size_selector, image_size_selector))
        original_image_pil.save(original_image_path)


        # Initialize Genetic Algorithm
        init_pop = [
            original_latent.unsqueeze(0) + initial_perturb_size * torch.randn_like(original_latent)
            for _ in range(pop_size)
        ]
        pop_labels = torch.tensor([original_label] * pop_size)
        pop_class_vectors = torch.nn.functional.one_hot(pop_labels, num_classes).float().to(device)

        now_pop = init_pop
        prev_best = np.inf
        best_fitness_score = np.inf
        best_image_tensor = None

        for g_idx in range(gen_num):
            indivs_lv = torch.cat(now_pop, dim=0).view(pop_size, -1)
            indivs_labels = pop_class_vectors.view(pop_size, -1)
            with torch.no_grad():
                Gen_imgs = G(indivs_lv, indivs_labels, truncation).to(device)
            Gen_imgs = torch.cat([preprocess_image(img) for img in Gen_imgs])
            all_logits = classifier(Gen_imgs).squeeze().detach().cpu().numpy()
            fitness_scores = [calculate_fitness(all_logits[k_idx], original_label) for k_idx in range(pop_size)]

            selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[-int(best_left):]

            current_min_index = np.argmin(fitness_scores)
            current_min_fitness = fitness_scores[current_min_index]

            if current_min_fitness < best_fitness_score:
                best_fitness_score = current_min_fitness
                best_image_tensor = Gen_imgs[current_min_index]

            now_best = np.min(fitness_scores)

            if now_best < 0: 
                break
            elif now_best == prev_best:
                perturb_size *= 2
            else:
                perturb_size = initial_perturb_size

            parent_pop = [now_pop[i] for i in selected_indices]
            k_pop = []

            for k_idx in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(128, size=1)[0]
                k_gene = torch.cat([
                    parent_pop[mom_idx][:, :spl_idx],
                    parent_pop[pop_idx][:, spl_idx:]
                ], dim=1)
                diffs = (k_gene != original_latent).float()
                k_gene += perturb_size * torch.randn_like(k_gene) * diffs
                k_pop.append(k_gene)

            now_pop = parent_pop + k_pop
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop]
            prev_best = now_best

        if best_image_tensor is not None:

           final_generation = g_idx + 1
           final_iterations.append(final_generation) 
           # De-normalize perturbed image
           denorm_pert = denormalize_vgg_tensor(best_image_tensor.squeeze()).cpu()
           perturbed_image_pil = transforms.ToPILImage()(denorm_pert)

           # De-normalize original image
           denorm_orig = denormalize_vgg_tensor(original_image.squeeze()).cpu()
           original_image_pil = transforms.ToPILImage()(denorm_orig)

           predicted_best_label = torch.argmax(classifier(best_image_tensor.unsqueeze(0).to(device)), dim=1).item()
           
           # Save perturbed image
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



