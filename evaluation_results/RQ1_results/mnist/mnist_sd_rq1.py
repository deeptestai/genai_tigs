import os
import cv2
import torch
import numpy as np
import wandb
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from experiments.mnist_experiments.mnist_gan.conditional_gan.sa.model import MnistClassifier

# Initialize WandB
#run = wandb.init(project="sinvadtestfitness_mnist")
generator = torch.Generator(device = 'cuda')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 28 * 28
height = 512
width = 512
# Define image processing function
def process_image(image):
    """
    Convert a 3-channel RGB PIL Image to grayscale, resize it to 28x28 pixels,
    and convert it to a PyTorch tensor.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided image needs to be a PIL Image.")

    # Convert PIL Image to numpy array (RGB)
    img_np = np.array(image)

    # Convert the image from RGB to grayscale
    gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(gray_image, (28, 28))

    # Convert the numpy array back to PIL Image (to use torchvision transforms)
    img_pil = Image.fromarray(resized_image)

    # Convert PIL Image to PyTorch Tensor
    transform = transforms.ToTensor()
    tensor = transform(img_pil)

    return tensor

# Load the MNIST classifier
classifier = MnistClassifier(img_size=img_size).to(device)
classifier.load_state_dict(
    torch.load(
        "./sa/models/MNIST_conv_classifier.pth",
        map_location=device,
    )
)
classifier.eval()

# Prompts for generating images
prompts = [
    "A photo of Zero Number 0", "A photo of One Number 1", "A photo of Two Number 2",
    "A photo of Three Number 3", "A photo of Four Number 4", "A photo of Five Number 5",
    "A photo of Six Number 6", "A photo of Seven Number 7", "A photo of Eight Number 8",
    "A photo of Nine Number 9"
]

# Stable Diffusion parameters
base_model_id = "runwayml/stable-diffusion-v1-5"
weights_path = "./Mnist_Lora_sdv1.5.safetensors"
num_inference_steps = 15
imgs_to_samp = 100
seed = 0

# Initialize the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id, variant="fp16", torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe.load_lora_weights(weights_path)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True)

# Directories to save generated images
proj_name = "test"
proj_path = "./sd_mnist_rq1gs3.5/" + proj_name + "_"
os.makedirs(proj_path, exist_ok=True)
os.makedirs(os.path.join(proj_path, 'generated_images'), exist_ok=True)

# Track correct predictions
correct_predictions = 0

# Generate and classify images
for img_idx in range(imgs_to_samp):
    prompt = prompts[n % len(prompts)]
    expected_label = n % len(prompts)

    with torch.no_grad():
        seedSelect = seed+img_idx
        generator = generator.manual_seed(seedSelect)
        original_lv = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8),generator=generator, device = device).to(torch.float16)
        # Generate image
        image = pipe(prompt = prompt,guidance_scale=3.5, num_inference_steps= num_inference_steps,latents=original_lv).images[0]
        
        # Process image
        processed_image = process_image(image).unsqueeze(0).to(device)
        
        # Classify image
        logits = classifier(processed_image).squeeze().detach().cpu().numpy()
        predicted_label = np.argmax(logits).item()

        # Check if the predicted label matches the expected label
        if predicted_label == expected_label:
            correct_predictions += 1
        
        # Save generated image
        image.save(os.path.join(proj_path, 'generated_images', f'generated_image_{img_idx}_X{expected_label}_Y{predicted_label}.png'))

# Calculate accuracy
accuracy = (correct_predictions / imgs_to_samp) * 100
print(f'Accuracy: {accuracy:.2f}%')
