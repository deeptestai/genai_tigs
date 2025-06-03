import torch
from tqdm import trange
from torchvision.utils import save_image
import numpy as np
import os
from PIL import Image
import zipfile
from sa.mnist_classifier.model import MnistClassifier
from cdcgan.mnist_cdcgan.cdcgan_mnist import Generator
#full Min and Max value
min_val = -4.78276300430298
max_val = 4.08758640289307

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Model once globally
img_size = 28 * 28 * 1
G = Generator(ngpu=1, nz=100, nc=1).to(device)
classifier = MnistClassifier(img_size=img_size).to(device)
G.load_state_dict(torch.load("./cdcgan/mnist_cdcgan/weights/netG_epoch_10cdcgan.pth", map_location=device))
G.eval()



def calculate_fitness(logit, label):
    expected_logit = logit[label]
    best_indices = np.argsort(-logit)[:2]
    best_index1, best_index2 = best_indices
    best_but_not_expected = best_index2 if best_index1 == label else best_index1
    new_logit = logit[best_but_not_expected]
    return expected_logit - new_logit

def run_gan_tig(gen_num, pop_size, best_left,perturbation_size,initial_perturbation_size, imgs_to_samp, classifier_choice, classifier_file):
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



    result_dir = "./result23_cdcgan_mnist"
    #os.makedirs(os.path.join(result_dir, "original_images"), exist_ok=True)
    #os.makedirs(os.path.join(result_dir, "perturb_images"), exist_ok=True)
    original_images_dir = os.path.join(result_dir, "original_images")
    os.makedirs(original_images_dir, exist_ok=True)
    perturb_images_dir = os.path.join(result_dir, "perturb_images")
    os.makedirs(perturb_images_dir, exist_ok=True)


   # perturbation_size = 0.000887034940719605
   # initial_perturbation_size = 0.00177406988143921

    latent_space = torch.randn(imgs_to_samp, 100, 1, 1).to(device)
    random_labels = torch.randint(0, 10, (imgs_to_samp,)).to(device)
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
        original_latent = latent_space[img_idx].unsqueeze(0)
        original_label_tensor = random_labels[img_idx]
        expected_label = original_label_tensor.item()

        original_image = G(original_latent, original_label_tensor.view(1))
        original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit).item()

        if original_label != expected_label:
            num1_misclassified += 1
            continue  # Skip mismatches
       # total_images += 1


        # Save original image
        original_image_np = original_image.squeeze().detach().cpu().numpy()
        original_image_pil = Image.fromarray((original_image_np * 255).astype(np.uint8))
        original_image_path = os.path.join(original_images_dir, f"original_image_{saved_images}_X{original_label}.png")
        #mod_image = original_image_pil.resize((image_size_selector, image_size_selector))
        original_image_pil.save(original_image_path)
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
            perturbed_image_pil = Image.fromarray((pert_img_np * 255).astype(np.uint8))
           # perturbed_image_pil = perturbed_image_pil.resize((image_size_selector, image_size_selector))

            predicted_best_label = torch.argmax(classifier(best_image_tensor.unsqueeze(0).to(device)), dim=1).item()
            perturbed_image_path = os.path.join(perturb_images_dir, f"image_{saved_images}_iteration{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png")
            perturbed_image_pil.save(perturbed_image_path)
            #resized_original = original_image_pil.resize(desired_size)
            #resized_perturbed = perturbed_image_pil.resize(desired_size)
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
          f"Finished! Total saved images: {saved_images} | Misclassified seeds {num_misclassified}| % Misclassification: {misclassification_rate}|Avg Iterations{Avg_iterations}",
          all_gallery_items,
          zip_path, status_rows
        )

