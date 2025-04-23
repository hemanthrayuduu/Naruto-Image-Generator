from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
import torch
from typing import Optional

# 1. 加载基础模型
print("Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16",
    safety_checker=None
).to("mps")
print("Base model loaded.")

# 2. 注入微调的 LoRA adapter
print("Loading LoRA adapter...")
unet = pipe.unet
peft_config = PeftConfig.from_pretrained("./model")
unet = PeftModel.from_pretrained(unet, "./model", adapter_name="default")
pipe.unet = unet
print("LoRA adapter loaded.")

# 3. 推理函数
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

    # Handle seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
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
        # Re-raise the exception so FastAPI can handle it (optional, could return None)
        # Depending on desired behavior, you might want to return a specific error indicator
        raise e 
