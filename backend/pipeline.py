import os
import torch
from typing import Optional
import warnings
import huggingface_hub
warnings.filterwarnings("ignore")

# Try to import diffusers with fallback handling
try:
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
    print("Diffusers imported successfully")
except ImportError as e:
    print(f"Warning: Could not import diffusers: {e}")
    DIFFUSERS_AVAILABLE = False

# Environment variables with defaults
DEVICE = os.environ.get("DEVICE", "cpu")
TORCH_DTYPE = getattr(torch, os.environ.get("TORCH_DTYPE", "float32"))
MODEL_DIR = os.environ.get("MODEL_DIR", "./model")

# Use a much smaller model for memory-constrained environments
SMALL_MODEL = "runwayml/stable-diffusion-v1-5"
ENABLE_MEMORY_EFFICIENT_ATTENTION = True

print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")
print(f"Diffusers available: {DIFFUSERS_AVAILABLE}")
print("huggingface_hub version:", huggingface_hub.__version__)
print("split_torch_state_dict_into_shards in huggingface_hub:", hasattr(huggingface_hub, "split_torch_state_dict_into_shards"))

# Initialize pipeline as None
pipe = None

def create_dummy_pipeline():
    """Create a dummy pipeline that generates simple colored images"""
    from PIL import Image, ImageDraw
    import random
    
    class DummyPipeline:
        def __call__(self, prompt, **kwargs):
            # Create a simple colored image with text
            width = kwargs.get('width', 256)
            height = kwargs.get('height', 256)
            
            # Generate random color based on prompt hash
            color_hash = hash(prompt) % 16777215
            color = f"#{color_hash:06x}"
            
            img = Image.new('RGB', (width, height), color)
            draw = ImageDraw.Draw(img)
            
            # Add prompt text (simplified)
            try:
                # Use default font
                draw.text((10, 10), f"Generated: {prompt[:30]}...", fill="white")
                draw.text((10, height-30), "Demo Mode (Diffusers not available)", fill="white")
            except:
                pass
            
            # Mock return format
            class MockResult:
                def __init__(self, img):
                    self.images = [img]
            
            return MockResult(img)
    
    print("Using dummy pipeline (fallback mode)")
    return DummyPipeline()

def initialize_pipeline():
    """Initialize the pipeline with memory optimizations"""
    global pipe
    
    if pipe is not None:
        return pipe
    
    if not DIFFUSERS_AVAILABLE:
        print("Diffusers not available, using dummy pipeline")
        return create_dummy_pipeline()
    
    print("Loading optimized Stable Diffusion model...")
    
    try:
        # Try to load with maximum memory efficiency
        pipe = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch",  # Tiny model for testing
            torch_dtype=TORCH_DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        # Enable memory efficient attention if available
        if hasattr(pipe.unet, 'set_use_memory_efficient_attention_xformers'):
            try:
                pipe.unet.set_use_memory_efficient_attention_xformers(True)
            except:
                pass
        
        # Move to device
        pipe = pipe.to(DEVICE)
        
        # Enable CPU offload for even more memory efficiency
        if DEVICE == "cpu":
            try:
                pipe.enable_sequential_cpu_offload()
            except:
                pass
        
        print("Tiny Stable Diffusion model loaded successfully (memory optimized)")
        return pipe
        
    except Exception as e:
        print(f"Error loading tiny model: {e}")
        # Ultimate fallback - create a dummy pipeline that returns a simple image
        return create_dummy_pipeline()

# 3. Inference function
def generate_image(
    prompt: str, 
    num_inference_steps: int = 10,  # Reduced for memory efficiency
    guidance_scale: float = 7.5, 
    width: int = 256,  # Reduced size for memory efficiency
    height: int = 256, # Reduced size for memory efficiency
    seed: Optional[int] = None
):
    """Generates an image using the memory-optimized pipeline"""
    global pipe
    
    # Initialize pipeline if not already done
    if pipe is None:
        pipe = initialize_pipeline()
    
    print(f"Generating image for prompt: '{prompt[:50]}...'")
    print(f"Settings: Steps={num_inference_steps}, Scale={guidance_scale}, Size={width}x{height}")
    
    # Limit parameters for memory efficiency
    num_inference_steps = min(num_inference_steps, 20)  # Cap at 20 steps
    width = min(width, 512)   # Cap at 512px
    height = min(height, 512) # Cap at 512px

    # Handle seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        print(f"Using seed: {seed}")

    try:
        # Call the pipeline with memory-efficient parameters
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
        print(f"Error during generation: {e}")
        # Return a simple error image
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (width, height), 'red')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Error: Generation failed", fill="white")
        draw.text((10, 30), str(e)[:40], fill="white")
        return img 
