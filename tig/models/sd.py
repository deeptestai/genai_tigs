"""Stable Diffusion 1.5 with the paper's fine-tuned LoRA weights (one per dataset)."""

from pathlib import Path

import torch

# The legacy scripts load "runwayml/stable-diffusion-v1-5", which is no longer on Hugging Face;
# this is the official mirror of the same weights.
SD_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def stable_diffusion(lora_path: Path, device, precision: str = "fp16", base_model: str = SD_BASE_MODEL):
    """Pipeline set up as in ``legacy/<dataset>/sd/tig_sd.py``: the dataset's LoRA, DDIM scheduler
    with zero-terminal-SNR betas, no safety checker.

    ``precision="fp16"`` is the legacy setting. Some GPUs (GTX 16xx, e.g. the GTX 1650) overflow
    in the fp16 UNet and return NaN, i.e. black images: use ``"fp32"`` there. In fp32 the model
    doesn't fit in 4 GB, so its components are moved to the GPU one at a time (slower)."""
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    on_gpu = torch.device(device).type == "cuda"
    fp32 = precision == "fp32" or not on_gpu
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, variant="fp16", torch_dtype=torch.float32 if fp32 else torch.float16,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.load_lora_weights(str(lora_path))
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True)
    pipe.set_progress_bar_config(disable=True)
    # Lower peak memory so a 4 GB GPU can run it; same images up to float rounding.
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()  # pipe.enable_vae_slicing() in older diffusers, removed in 0.42
    if fp32 and on_gpu:
        pipe.enable_model_cpu_offload(device=device)
        return pipe
    return pipe.to(device)
