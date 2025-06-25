import gradio as gr
import torch
import spaces
from diffusers import StableDiffusionPipeline
import os
from PIL import Image
import random

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
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True
    )
    
    # Try to load your LoRA model if available
    if os.path.exists("./model") and any(f.endswith('.safetensors') for f in os.listdir("./model")):
        try:
            pipe.load_lora_weights("./model")
            print("✅ LoRA weights loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load LoRA weights: {e}")
            print("Using base Stable Diffusion model")
    
    pipe = pipe.to(DEVICE)
    print(f"✅ Pipeline loaded on {DEVICE}")
    return pipe

@spaces.GPU
def generate_naruto_image(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = -1
):
    """Generate Naruto-style image using ZeroGPU"""
    global pipe
    
    if pipe is None:
        pipe = initialize_pipeline()
    
    # Handle random seed
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Enhanced prompt for Naruto style
    enhanced_prompt = f"{prompt}, naruto style, anime art, detailed, high quality"
    
    # Generate image
    try:
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
        # Return error image
        error_img = Image.new('RGB', (512, 512), color='red')
        return error_img, f"Error: {str(e)}"

# Define Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="🍥 Naruto Image Generator",
        css="""
        .generate-btn {
            background: linear-gradient(45deg, #ff6b35, #f7931e) !important;
            border: none !important;
            color: white !important;
            font-weight: bold !important;
        }
        .generate-btn:hover {
            transform: scale(1.05) !important;
        }
        """
    ) as demo:
        
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
                    variant="primary",
                    elem_classes=["generate-btn"]
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
        
        examples = gr.Examples(
            examples=[[prompt, "", 25, 7.5, 512, 512, -1] for prompt in example_prompts],
            inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
            outputs=[output_image, output_info],
            fn=generate_naruto_image,
            cache_examples=False
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
    demo.launch(
        share=False,  # HF Spaces handles sharing
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 