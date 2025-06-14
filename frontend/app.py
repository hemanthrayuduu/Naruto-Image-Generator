"""Home page for the Naruto Image Generator application."""
import streamlit as st
import requests
from PIL import Image
import io
from typing import Optional, Dict, Any
import os
from utils import save_generated_image
from config import (
    BACKEND_API_URL,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
)

def generate_image(
    prompt: str,
    steps: int,
    guidance_scale: float,
    dimensions: str,
    seed: Optional[int],
    model_type: str = "finetuned"  # or "base"
) -> Optional[Image.Image]:
    """Generate image using the specified model."""
    try:
        width, height = map(int, dimensions.split("x"))
        
        # Prepare parameters
        params = {
            "prompt": prompt,
            "num_inference_steps": steps,  # Changed to match backend parameter name
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "model_type": model_type
        }
        
        # Make API request
        response = requests.post(
            f"{BACKEND_API_URL}/generate",
            json=params,
            timeout=180,
        )
        response.raise_for_status()
        
        # Convert response to image
        image = Image.open(io.BytesIO(response.content))
        
        # Save image and metadata
        save_generated_image(
            image=image,
            prompt=prompt,
            model_type=model_type,
            parameters={
                "steps": steps,
                "guidance_scale": guidance_scale,
                "dimensions": dimensions,
                "seed": seed
            }
        )
        
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def main():
    # Set page config
    st.set_page_config(
        page_title="Naruto Image Generator",
        page_icon="🍥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Customize sidebar title
    st.sidebar.title("Naruto Image Generator")
    
    # Title and introduction
    st.title("Welcome to Naruto Image Generator! 🍥")
    
    # Hero section with description
    st.markdown("""
    Transform your ideas into stunning Naruto-style artwork using our advanced AI image generator!
    This application uses state-of-the-art machine learning models fine-tuned on Naruto artwork
    to create unique and authentic-looking images in the iconic Naruto style.
    """)
    
    # Generation Form
    st.header("🎨 Generate Images")
    
    with st.form("generation_form"):
        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="Example: A ninja in Naruto style with blue eyes and blonde hair",
            help="Describe the image you want to generate"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            steps = st.slider(
                "Number of inference steps",
                min_value=0,
                max_value=100,
                value=DEFAULT_STEPS,
                help="More steps generally means better quality but slower generation"
            )
            
            dimensions = st.selectbox(
                "Image dimensions",
                options=["256x256", "512x512", "768x768"],
                index=1,
                help="Choose the size of the generated image"
            )
        
        with col2:
            guidance_scale = st.slider(
                "Guidance scale",
                min_value=1.0,
                max_value=20.0,
                value=DEFAULT_GUIDANCE_SCALE,
                step=0.5,
                help="Higher values make the image more closely match the prompt"
            )
            
            seed = st.number_input(
                "Seed (optional)",
                value=DEFAULT_SEED if DEFAULT_SEED is not None else None,
                help="Set a seed for reproducible results",
                step=1
            )

        # Model comparison info
        st.markdown("### 🔄 Model Comparison")
        st.info("Images will be generated using both the Naruto fine-tuned model and the Stable Diffusion 2.1 base model for comparison.")
        
        submitted = st.form_submit_button("Generate Images")
    
    if submitted and prompt:
        with st.spinner("Generating images from both models..."):
            # Enhanced prompt only for fine-tuned model
            enhanced_prompt = prompt + ", detailed high quality anime style"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Fine-tuned Naruto Model")
                finetuned_image = generate_image(
                    enhanced_prompt, steps, guidance_scale, dimensions, seed, "finetuned"
                )
                if finetuned_image:
                    st.image(finetuned_image, use_container_width=True)
                    # Create download data
                    buf = io.BytesIO()
                    finetuned_image.save(buf, format='PNG')
                    st.download_button(
                        "⬇️ Download Fine-tuned Image",
                        buf.getvalue(),
                        file_name=f"finetuned_naruto_{seed if seed else 'random'}.png",
                        mime="image/png"
                    )
                    st.markdown("*Fine-tuned model specializes in Naruto-style artwork with high-quality details*")
            
            with col2:
                st.subheader("🎨 Stable Diffusion 2.1 Base Model")
                # Use original prompt for base model
                base_image = generate_image(
                    prompt, steps, guidance_scale, dimensions, seed, "base"
                )
                if base_image:
                    st.image(base_image, use_container_width=True)
                    # Create download data
                    buf = io.BytesIO()
                    base_image.save(buf, format='PNG')
                    st.download_button(
                        "⬇️ Download Base Image",
                        buf.getvalue(),
                        file_name=f"base_naruto_{seed if seed else 'random'}.png",
                        mime="image/png"
                    )
                    st.markdown("*Stable Diffusion 2.1 Base model for general artistic interpretation*")
            
            # Add comparison analysis
            if finetuned_image and base_image:
                st.markdown("---")
                st.subheader("📊 Comparison Analysis")
                st.markdown("""
                Compare the images above to see how the fine-tuned Naruto model differs from the base model:
                - Look for Naruto-specific art style elements
                - Notice differences in character design
                - Compare color palettes and shading
                - Observe background details
                """)
    
    # Tips section at the bottom
    st.markdown("---")
    st.header("💡 Pro Tips")
    st.markdown("""
    - Be specific in your prompts for better results
    - Try comparing base and fine-tuned models to see the difference
    - Use the gallery to get inspiration from previous generations
    - Experiment with different guidance scales for varying levels of creativity
    """)

if __name__ == "__main__":
    main() 
