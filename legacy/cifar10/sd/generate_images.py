import os
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image


def save_images(imgs, folder_path, base_filename):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    for i, img in enumerate(imgs):
        img.save(os.path.join(folder_path, f"{base_filename}_{i}.png"), "PNG")


base_model_id = "runwayml/stable-diffusion-v1-5"
lora_model_id = "./models/Lora/cifar10_finetune_lorav1.5.safetensors"

pipeline = StableDiffusionPipeline.from_pretrained(
    base_model_id, torch_dtype=torch.float16, safety_checker=None
)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.to("cuda")
pipeline.load_lora_weights(lora_model_id)

prompt = "A Photo of truck9"
images = pipeline(
    prompt=prompt,
    num_inference_steps=25,
    num_images_per_prompt=10,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 1},
    generator=torch.manual_seed(0)
).images

# Specify folder path and base filename for saving images
save_folder_path = "./generated_images_cifar10"
base_filename = "cifar10_9"

# Save images to the specified folder
save_images(images, save_folder_path, base_filename)

