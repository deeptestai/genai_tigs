import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def save_images(imgs, folder_path, base_filename):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    for i, img in enumerate(imgs):
        img.save(os.path.join(folder_path, f"{base_filename}_{i}.png"), "PNG")

base_model_id = "runwayml/stable-diffusion-v1-5"
lora_model_file = "/content/drive/MyDrive/fine_tune_imgnet-000005.safetensors"

pipeline = StableDiffusionPipeline.from_pretrained(
    base_model_id, torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.load_lora_weights(lora_model_file)

prompt = "a photo of 1pizza"
images = pipeline(
    prompt=prompt,
    num_inference_steps=25,
    num_images_per_prompt=1,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.8},
    generator=torch.manual_seed(0)
).images

# Specify folder path and base filename for saving images
save_folder_path = "/content/generated_images_imgnt"
base_filename = "generated_image"

# Save images to the specified folder
save_images(images, save_folder_path, base_filename)
