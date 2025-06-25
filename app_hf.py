import gradio as gr
import torch
from pipeline_hf import generate_image
import os
from PIL import Image

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["DEVICE"] = device

# Gradio interface for Hugging Face Spaces
def generate_naruto_image(
    prompt: str,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = None
):
    """Generate Naruto-style images using the fine-tuned model"""
    try:
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        # Generate image
        image = generate_image(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed if seed != 0 else None
        )
        
        return image, f"Generated successfully on {device.upper()}"
        
    except Exception as e:
        error_img = Image.new('RGB', (512, 512), 'red')
        return error_img, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Naruto Image Generator") as demo:
    gr.Markdown("# 🍃 Naruto Image Generator")
    gr.Markdown("Generate Naruto-style anime images using Stable Diffusion")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="Describe the image you want to generate...",
                lines=3
            )
            
            with gr.Row():
                steps = gr.Slider(10, 50, value=25, label="Inference Steps")
                guidance = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
            
            with gr.Row():
                width = gr.Slider(256, 768, value=512, step=64, label="Width")
                height = gr.Slider(256, 768, value=512, step=64, label="Height")
            
            seed = gr.Number(label="Seed (0 for random)", value=0, precision=0)
            
            generate_btn = gr.Button("🎨 Generate Image", variant="primary")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")
            status = gr.Textbox(label="Status", interactive=False)
    
    # Event handlers
    generate_btn.click(
        fn=generate_naruto_image,
        inputs=[prompt, steps, guidance, width, height, seed],
        outputs=[output_image, status]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["Naruto Uzumaki in sage mode, anime style", 25, 7.5, 512, 512, 42],
            ["Sasuke Uchiha with sharingan, dark aesthetic", 30, 8.0, 512, 512, 123],
            ["Sakura Haruno in a cherry blossom field", 25, 7.0, 512, 512, 456],
        ],
        inputs=[prompt, steps, guidance, width, height, seed]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 