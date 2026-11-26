import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import torchvision.utils as vutils
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample

def save_image(image_tensor, filename):
    """Converts a tensor to an image and saves it."""
    img = image_tensor.clone().detach()
    img = img.to('cpu').float().numpy()
    if img.shape[0] == 1:
        img = np.squeeze(img, axis=0)  # Remove channel dimension if it is 1
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)  # Expand grayscale to 3D
    img = np.transpose(img, (1, 2, 0))
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename)

# Preprocess transform definition for classifier
preprocess_for_classifier = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.ToTensor(),  # Convert back to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Preprocess transform definition for saving images
def preprocess_for_saving(image_tensor):
    image = (image_tensor + 1) / 2  # Scale to [0, 1]
    image = image.squeeze(0)  # Remove batch dimension if it exists
    image = transforms.ToPILImage()(image.cpu())  # Convert to PIL Image
    return image

# Function to preprocess an image for VGG19
def preprocess_image(image_tensor):
    image = preprocess_for_saving(image_tensor)  # Convert tensor to PIL Image
    image = preprocess_for_classifier(image)  # Apply resizing and normalization for the classifier
    return image.unsqueeze(0).to(device)  # Add batch dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained models
G = BigGAN.from_pretrained('biggan-deep-256').to(device)

# Load pretrained model
classifier = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).to(device)
classifier.eval()

# Set models to evaluation mode
G.eval()

# Directories to save the images
result_dir = "./INPI_gan_rq1"
original_images_dir = os.path.join(result_dir, 'original_images')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(original_images_dir, exist_ok=True)
img_pil_dir = os.path.join(original_images_dir, "or_images")
os.makedirs(img_pil_dir, exist_ok=True)
prtb_img_dir = os.path.join(result_dir, "perturb_images")
os.makedirs(prtb_img_dir, exist_ok=True)

imgs_to_samp = 100  # Number of samples to generate and test
num_classes = 1000  # Number of classes in ImageNet
truncation = 1  # Truncation value
expected_label = 963       # change class label 850 for teddy_bear class
# Track correct predictions
correct_predictions = 0

for img_idx in trange(imgs_to_samp):
    seed = torch.randint(0, 10000, (1,)).item()
    torch.manual_seed(seed)  # Set the seed for reproducibility
    noise_vector = truncated_noise_sample(batch_size=1, dim_z=128, truncation=truncation, seed=seed)
    original_latent = torch.from_numpy(noise_vector).to(device)
    
    # Generate a random class label
    class_label = torch.zeros(1, num_classes).to(device)
    class_label[:, expected_label] = 1  # Set the specific class label

    with torch.no_grad():
        # Generate image with the generator model
        generated_image = G(original_latent, class_label, truncation).to(device)
        preprocessed_image = preprocess_image(generated_image)
        # Classify generated image
        logits = classifier(preprocessed_image).squeeze().detach().cpu().numpy()
        predicted_label = np.argmax(logits).item()

    # Check if the predicted label matches the expected label
    if predicted_label == expected_label:
        correct_predictions += 1

    original_image_path = os.path.join(img_pil_dir, f'original_image_{img_idx}_X{expected_label}_Y{predicted_label}.png')
    save_image(generated_image.squeeze(0), original_image_path)
    
    # Convert the tensor to a NumPy array
    original_image_np = generated_image.detach().cpu().numpy()

    # Define the filename that includes the label
    filename = f"original_image_{img_idx}_X_{expected_label}_Y_{predicted_label}.npy"
    file_path = os.path.join(original_images_dir, filename)

    # Save the image data as a NumPy file
    np.save(file_path, original_image_np)

# Calculate accuracy
accuracy = (correct_predictions / imgs_to_samp) * 100
print(f'Accuracy: {accuracy:.2f}%')
