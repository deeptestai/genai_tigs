import torch
import gradio as gr
from torchvision import transforms, datasets
import torchvision
import numpy as np
from tqdm import trange
import os
from PIL import Image
from sa.mnist_classifier.model import MnistClassifier
from vae.mnist_vae.model import VAE
import zipfile
# Full min and max values
min_val = -3.86494088172913
max_val = 3.45633792877197

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once globally
img_size = 28 * 28 * 1
vae = VAE(img_size=img_size, h_dim=1600, z_dim=400).to(device)
classifier = MnistClassifier(img_size=img_size).to(device)
vae.load_state_dict(torch.load("./vae/mnist_vae/models/MNIST_EnD.pth", map_location=device, weights_only=True))
vae.eval()


# Prepare dataset
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

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

def run_vae_tig(gen_num, pop_size, best_left, perturbation_size, initial_perturbation_size, imgs_to_samp, classifier_choice, classifier_file):
    if classifier_choice == "deepconv":
        classifier = MnistClassifier(img_size=28 * 28).to(device)
        classifier.load_state_dict(
            torch.load("./sa/mnist_classifier/models/MNIST_conv_classifier.pth", map_location=device)
        )
        classifier.eval()
        print(" Default classifier loaded.")

    elif classifier_choice == "Upload Custom":
         if classifier_file is None:
            raise ValueError("Please upload a .jit TorchScript model file.")

         try:
            classifier = torch.jit.load(classifier_file.name, map_location=device)
            classifier.eval()
            print(" Custom TorchScript classifier loaded.")
         except Exception as e:
             raise ValueError(f" Failed to load custom classifier: {e}")

    else:
         raise ValueError("Unknown classifier selected.")
    result_dir = "./result23_vae_mnist"
    os.makedirs(result_dir, exist_ok=True)
    original_images_dir = os.path.join(result_dir, "original_images")
    os.makedirs(original_images_dir, exist_ok=True)
    #if image_size_selector is None:
       #print("WARNING: image_size_selector is None. Defaulting to 28.")
       #image_size_selector = 28
    #desired_size = (int(image_size_selector), int(image_size_selector))
    saved_images = 0
    final_iterations = []
    num_misclassified = 0
    num1_misclassified = 0
    total_images = 0
    all_gallery_items = []
    saved_image_paths = []
    status_rows = []


    for img_idx in trange(imgs_to_samp):
        for i, (x, x_class) in enumerate(test_data_loader):
            samp_img = x[0:1]
            samp_class = x_class[0].item()
            break  # take one sample

        img_enc, _ = vae.encode(samp_img.view(-1, img_size).to(device))
        original_lv = img_enc
        original_image = vae.decode(original_lv).view(-1, 1, 28, 28)
        original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
        original_label = original_logit.argmax().item()

        if original_label != samp_class:
            num1_misclassified += 1
            continue  # Skip mismatches
        #total_images += 1

        # Prepare original image
        original_image_pil = transforms.ToPILImage()(original_image.squeeze().cpu())
        original_image_path = os.path.join(original_images_dir, f"original_image_{saved_images}_X{original_label}.png")
        original_image_pil.save(original_image_path)

        # Start genetic algorithm
        init_pop = [original_lv + initial_perturbation_size * torch.randn(1, 400).to(device) for _ in range(pop_size)]
        now_pop = init_pop
        prev_best = np.inf
        best_fitness_score = np.inf
        best_image_tensor = None

        for g_idx in range(gen_num): 
            indivs = torch.cat(now_pop, dim=0)
            dec_imgs = vae.decode(indivs).view(-1, 1, 28, 28)
            all_logits = classifier(dec_imgs).squeeze().detach().cpu().numpy()

            fitness_scores = [calculate_fitness(all_logits[k_idx], original_label) for k_idx in range(pop_size)]
            current_min_index = np.argmin(fitness_scores)
            current_min_fitness = fitness_scores[current_min_index]

            if current_min_fitness < best_fitness_score:
                best_fitness_score = current_min_fitness
                best_image_tensor = dec_imgs[current_min_index].cpu().detach()

            selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[-best_left:]
            now_best = np.min(fitness_scores)
            parent_pop = [now_pop[idx] for idx in selected_indices]

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
                k_gene = torch.cat([parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]], dim=1)
                diffs = (k_gene != original_lv).float()
                k_gene += perturbation_size * torch.randn(k_gene.size()).to(device) * diffs
                k_pop.append(k_gene)
            now_pop = parent_pop + k_pop
            prev_best = now_best
            now_pop = [torch.clamp(tensor, min=min_val, max=max_val) for tensor in now_pop] 

        # Save perturbed image
        final_generation = g_idx + 1
        final_iterations.append(final_generation)
        mod_best_image_tensor = best_image_tensor.to(device)
        predicted_best_label = torch.argmax(classifier(mod_best_image_tensor.unsqueeze(0)), dim=1).item()
        perturbed_image_pil = transforms.ToPILImage()(mod_best_image_tensor.squeeze().cpu())

        perturbed_image_path = os.path.join(result_dir, f"image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png")
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
          f"Processing image {img_idx + 1} / {imgs_to_samp}|Misclassified seeds {num1_misclassified}|% Misclassification: {misclassification_rate} |Avg Iteration{Avg_iterations}",
          all_gallery_items, None , status_rows
        )
       

        zip_path = os.path.join(result_dir, "generated_pairs.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in saved_image_paths:
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname=arcname)
        # Final yield
        yield (
          f"Finished! Total saved images: {saved_images} | Misclassified seeds {num_misclassified}| % Misclassification: {misclassification_rate:.2f} | Avg Iterations: {Avg_iterations:.2f}",
          all_gallery_items,
          zip_path, status_rows
        )



   # yield (
    #    f"Finished! Total saved images: {saved_images}",
     #   [
      #    (resized_original, f"Expected label: {original_label}"),
       #   (resized_perturbed, f"Predicted label: {predicted_best_label}")
       # ]
   # )
#import zipfile

#zip_path = os.path.join(result_dir, "generated_pairs.zip")
#with zipfile.ZipFile(zip_path, 'w') as zipf:
 #   for file_path in saved_image_paths:
  #      arcname = os.path.basename(file_path)
   #     zipf.write(file_path, arcname=arcname)
