import os
from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
import torch
from typing import Optional

# Environment variables with defaults
DEVICE = os.environ.get("DEVICE", "cpu")
TORCH_DTYPE = getattr(torch, os.environ.get("TORCH_DTYPE", "float32"))
MODEL_DIR = os.environ.get("MODEL_DIR", "./model")
BASE_MODEL = os.environ.get("BASE_MODEL", "CompVis/stable-diffusion-v1-4")

# Check if CUDA is available and adjust device accordingly
if DEVICE == "auto":
    if torch.cuda.is_available():
        DEVICE = "cuda"
        TORCH_DTYPE = torch.float16  # Use float16 for GPU
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
        TORCH_DTYPE = torch.float16  # Use float16 for Apple Silicon
    else:
        DEVICE = "cpu"
        TORCH_DTYPE = torch.float32  # Use float32 for CPU

print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")

# 1. Load base model
print("Loading base model...")
try:
    pipe_kwargs = {
        "torch_dtype": TORCH_DTYPE,
        "safety_checker": None
    }
    
    # Only add revision for float16
    if TORCH_DTYPE == torch.float16:
        pipe_kwargs["revision"] = "fp16"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        **pipe_kwargs
    ).to(DEVICE)
    print("Base model loaded successfully.")
except Exception as e:
    print(f"Error loading base model: {e}")
    # Fallback to CPU with float32
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to("cpu")
    DEVICE = "cpu"
    print("Fallback: Using CPU with float32")

# 2. Load LoRA adapter
print("Loading LoRA adapter...")
try:
    unet = pipe.unet
    peft_config = PeftConfig.from_pretrained(MODEL_DIR)
    unet = PeftModel.from_pretrained(unet, MODEL_DIR, adapter_name="default")
    pipe.unet = unet
    print("LoRA adapter loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load LoRA adapter: {e}")
    print("Continuing with base model only...")

# 3. Inference function
def generate_image(
    prompt: str, 
    num_inference_steps: int = 30, 
    guidance_scale: float = 7.5, 
    width: int = 512, 
    height: int = 512,
    seed: Optional[int] = None
):
    """Generates an image using the Stable Diffusion pipeline with specified parameters."""
    print(f"Generating image for prompt: '{prompt}'")
    print(f"Settings: Steps={num_inference_steps}, Scale={guidance_scale}, Size={width}x{height}, Seed={seed}")
    print(f"Device: {DEVICE}, Memory available: {torch.cuda.get_device_properties(0).total_memory if DEVICE == 'cuda' else 'N/A'}")

    # Handle seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("Using random seed.")

    try:
        # Call the pipeline with all parameters
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale, 
            width=width,
            height=height,
            generator=generator
        )
        image = result.images[0]
        print("Image generation successful.")
        return image
    except Exception as e:
        # Log the error for backend debugging
        print(f"Error during pipeline execution: {e}")
        # Re-raise the exception so FastAPI can handle it
        raise e 
