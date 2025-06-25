import os
import torch
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Try to import diffusers with fallback handling
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
    print("Diffusers imported successfully")
except ImportError as e:
    print(f"Warning: Could not import diffusers: {e}")
    DIFFUSERS_AVAILABLE = False

# Detect device - prioritize GPU on HF Spaces
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_DIR = "./model"

# Use your fine-tuned LoRA model
BASE_MODEL = "CompVis/stable-diffusion-v1-4"
LORA_MODEL_PATH = MODEL_DIR

print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")
print(f"Diffusers available: {DIFFUSERS_AVAILABLE}")

# Initialize pipeline as None
pipe = None

def initialize_pipeline():
    """Initialize the pipeline optimized for HF Spaces"""
    global pipe
    
    if pipe is not None:
        return pipe
    
    if not DIFFUSERS_AVAILABLE:
        print("Diffusers not available")
        return None
    
    print("Loading Stable Diffusion model for HF Spaces...")
    
    try:
        # Load base model with optimizations for HF Spaces
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=TORCH_DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        # Move to device
        pipe = pipe.to(DEVICE)
        
        # Try to load LoRA weights if available
        if os.path.exists(LORA_MODEL_PATH):
            try:
                pipe.load_lora_weights(LORA_MODEL_PATH)
                print("✅ LoRA weights loaded successfully")
            except Exception as e:
                print(f"⚠️ LoRA loading failed: {e}")
                print("Continuing with base model...")
        
        # Enable optimizations
        if DEVICE == "cuda":
            # GPU optimizations
            pipe.enable_model_cpu_offload()
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers memory optimization enabled")
            except:
                print("XFormers not available")
        else:
            # CPU optimizations
            pipe.enable_sequential_cpu_offload()
        
        print(f"✅ Pipeline loaded successfully on {DEVICE}")
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def generate_image(
    prompt: str, 
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5, 
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None
):
    """Generate image optimized for HF Spaces"""
    global pipe
    
    # Initialize pipeline if not already done
    if pipe is None:
        pipe = initialize_pipeline()
        
    if pipe is None:
        raise Exception("Pipeline failed to initialize")
    
    print(f"🎨 Generating: '{prompt[:50]}...'")
    
    # Handle seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Generate image
    with torch.autocast(DEVICE):
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
    
    image = result.images[0]
    print("✅ Generation complete!")
    return image 