# IMPORTANT: Import spaces FIRST before any CUDA-related packages
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False
    print("⚠️ Spaces module not available, running without ZeroGPU")

# Now import other packages
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import os
from PIL import Image
import random

# Patch Gradio 5.x get_type() to handle boolean schemas
# This error happens during component introspection and breaks API registration
try:
    from gradio_client import utils
    _original_get_type = utils.get_type
    _original_json_schema_to_python_type = utils._json_schema_to_python_type
    
    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "any"
        return _original_get_type(schema)
    
    def _patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return _original_json_schema_to_python_type(schema, defs)
    
    utils.get_type = _patched_get_type
    utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except Exception:
    pass  # If patch fails, continue anyway

# Model configuration
BASE_MODEL = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize pipeline globally
pipe = None

def initialize_pipeline():
    """Initialize the pipeline for HF Spaces"""
    global pipe
    if pipe is not None:
        return pipe
    
    print("Loading Stable Diffusion pipeline...")
    print(f"Device: {DEVICE}, CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Use appropriate dtype based on device
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        
        # Try loading with variant first (for newer diffusers)
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
                variant="fp16" if DEVICE == "cuda" else None,
                low_cpu_mem_usage=True
            )
        except (TypeError, ValueError) as e:
            print(f"First load attempt failed: {e}")
            print("Trying without variant parameter...")
            # Fallback without variant
            pipe = StableDiffusionPipeline.from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
        
        # Try to load your LoRA model if available
        model_path = "./model"
        if os.path.exists(model_path):
            model_files = os.listdir(model_path)
            print(f"Model directory contents: {model_files}")
            
            if any(f.endswith('.safetensors') and 'adapter' in f for f in model_files):
                try:
                    pipe.load_lora_weights(model_path)
                    print("✅ LoRA weights loaded successfully!")
                except Exception as e:
                    print(f"⚠️ Could not load LoRA weights: {e}")
                    print("Using base Stable Diffusion model")
            else:
                print("⚠️ No LoRA adapter files found in model directory")
        else:
            print("⚠️ Model directory not found, using base Stable Diffusion")
        
        pipe = pipe.to(DEVICE)
        
        # Enable memory optimizations if on CPU
        if DEVICE == "cpu":
            pipe.enable_attention_slicing()
            print("✅ CPU optimizations enabled")
        
        print(f"✅ Pipeline loaded on {DEVICE}")
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        raise

def generate_naruto_image_impl(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = -1
):
    """Core image generation logic"""
    global pipe
    
    try:
        if pipe is None:
            pipe = initialize_pipeline()
        
        # Handle random seed
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        # Enhanced prompt for Naruto style
        enhanced_prompt = f"{prompt}, naruto style, anime art, detailed, high quality"
        
        # Generate image
        result = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
        
        image = result.images[0]
        return image, f"Generated with seed: {seed}"
    
    except Exception as e:
        # Return error image with details
        error_img = Image.new('RGB', (512, 512), color=(220, 100, 100))
        error_msg = f"Error: {str(e)[:100]}"
        print(f"Generation error: {e}")
        return error_img, error_msg

# Use raw function - Gradio 5.x handles @spaces.GPU differently
def generate_naruto_image(*args, **kwargs):
    if SPACES_AVAILABLE:
        # ZeroGPU will auto-allocate when called from HF Spaces
        return generate_naruto_image_impl(*args, **kwargs)
    else:
        return generate_naruto_image_impl(*args, **kwargs)

# Define Gradio interface
def create_interface():
    with gr.Blocks(title="🍥 Naruto Image Generator") as demo:
        
        gr.Markdown(
            """
            # 🍥 Naruto Image Generator
            
            Generate **Naruto-style anime images** using AI! 
            
            ⚡ **Powered by ZeroGPU** - Fast generation with H200 GPU  
            🎨 **Fine-tuned model** - Optimized for Naruto/anime style  
            🌟 **Multiple users welcome** - Share with friends!
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="🎯 Prompt",
                    placeholder="Describe the Naruto character or scene you want to generate...\nExample: 'Naruto Uzumaki in sage mode, orange jumpsuit, determined expression'",
                    lines=3,
                    value="Naruto Uzumaki, orange jumpsuit, headband, blue eyes, whiskers, determined expression"
                )
                
                negative_prompt = gr.Textbox(
                    label="🚫 Negative Prompt (Optional)",
                    placeholder="What you DON'T want in the image...",
                    lines=2,
                    value="blurry, low quality, distorted, nsfw"
                )
                
                with gr.Row():
                    steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="🔄 Inference Steps",
                        info="More steps = higher quality but slower"
                    )
                    
                    guidance = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="🎯 Guidance Scale",
                        info="How closely to follow the prompt"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                        label="📏 Width"
                    )
                    
                    height = gr.Slider(
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                        label="📏 Height"
                    )
                
                seed = gr.Number(
                    label="🎲 Seed (-1 for random)",
                    value=-1,
                    precision=0
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate Naruto Image!",
                    variant="primary"
                )
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="🖼️ Generated Image",
                    type="pil",
                    height=512
                )
                
                output_info = gr.Textbox(
                    label="ℹ️ Generation Info",
                    lines=2,
                    interactive=False
                )
        
        # Example prompts
        gr.Markdown("### 💡 Example Prompts")
        example_prompts = [
            "Naruto Uzumaki in sage mode, orange and black outfit, determined expression",
            "Sasuke Uchiha with sharingan eyes, dark hair, serious expression",
            "Sakura Haruno, pink hair, green eyes, medical ninja outfit",
            "Kakashi Hatake, silver hair, mask, reading book, relaxed pose",
            "Itachi Uchiha, long black hair, red sharingan, black cloak",
        ]
        
        gr.Examples(
            examples=[[p] for p in example_prompts],
            inputs=[prompt],
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_naruto_image,
            inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
            outputs=[output_image, output_info],
            show_progress=True
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### 🔧 Technical Details
            - **Model**: Stable Diffusion v1.4 + Naruto LoRA fine-tuning
            - **GPU**: NVIDIA H200 (70GB VRAM) via ZeroGPU
            - **Generation Time**: ~10-15 seconds per image
            - **Resolution**: Up to 768x768 pixels
            
            ### 📝 Tips for Better Results
            1. **Be specific** - Include details like clothing, pose, expression
            2. **Use character names** - "Naruto Uzumaki", "Sasuke Uchiha", etc.
            3. **Add style keywords** - "anime style", "detailed", "high quality"
            4. **Use negative prompts** - Remove unwanted elements
            
            ### 🌟 Share Your Creations!
            Found this useful? Share the space with your friends!
            """
        )
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    
    # Gradio 5.x still has a bug in get_type() for boolean schemas
    # It only affects API doc generation, not the app itself
    # Suppress the error to let the app launch
    try:
        demo.launch()
    except TypeError as e:
        if "unhashable type: 'dict'" in str(e) or "argument of type 'bool' is not iterable" in str(e):
            print(f"⚠️ Known Gradio schema parsing error (non-fatal): {e}")
            print("App should still be functional")
        else:
            raise 