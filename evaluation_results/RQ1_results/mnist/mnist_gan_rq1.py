import torch
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import trange
from sa.model import MnistClassifier
from cdcgan.cdcgan_mnist import Generator
import time
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28 * 28 * 1

# Load the trained models
G = Generator(ngpu=1, nz=100, nc=1).to(device)
classifier = MnistClassifier(img_size=img_size).to(device)

# Load the weights of the models
G.load_state_dict(
    torch.load("./cdcgan/weights/netG_epoch_10cdcgan.pth", map_location=device)
)
classifier.load_state_dict(
    torch.load(
        "./sa/models/MNIST_conv_classifier.pth",
        map_location=device,
    )
)

# Set models to evaluation mode
G.eval()
classifier.eval()

# Directory to save the images
result_dir = "./mnist_gan_rq1"
original_images_dir = os.path.join(result_dir, "original_images")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)

# Number of samples to generate
imgs_to_samp = 100  # You can set this to any number you want

# Generate random latent vectors and labels
latent_space = torch.randn(imgs_to_samp, 100, 1, 1).to(device)
random_labels = torch.randint(0, 10, (imgs_to_samp,)).to(device)
total_inference_time_ms = 0
# Store predictions and true labels
predictions = []
correct_predictions = 0
total_images = 0

for img_idx in trange(imgs_to_samp):
    # Generate the non-perturbed image
    original_latent = latent_space[img_idx].unsqueeze(0)
    original_labelr = random_labels[img_idx]
    expected_label  = original_labelr.item()

    # Generate the non-perturbed image using the generator
    original_image = G(original_latent, original_labelr.view(1))

    # Get the logit from the classifier
    original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
    original_label = np.argmax(original_logit).item()

    # Compare the predicted label with the label used for generation
    total_images += 1
    if original_label == expected_label:
        correct_predictions += 1
        print(f"Match found: True label {original_labelr.item()}, Predicted label {original_label}")

        # Convert the tensor to a NumPy array
        original_image_np = original_image.squeeze().detach().cpu().numpy()

        # Define the filename that includes the label
        filename = f"original_image_{img_idx}_X_{original_label}.npy"
        file_path = os.path.join(original_images_dir, filename)

        # Save the image data as a NumPy file
        np.save(file_path, original_image_np)


# Calculate accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"Model accuracy based on {total_images} images: {accuracy:.2f}%")

