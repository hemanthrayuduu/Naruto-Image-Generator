# frontend/app.py
import streamlit as st
import requests
from typing import Optional, Dict, Any, Tuple
import io
import base64
import time # Keep for download filename, maybe remove later if not needed
from pathlib import Path # Import Path
from config import BACKEND_API_URL

# --- Configuration ---
# TODO: Replace with actual backend URL from Ram or load from secrets/env var
# BACKEND_URL = "http://localhost:8000/generate" 

# --- Helper Functions ---

def local_css(file_name):
    """Loads a local CSS file into the Streamlit app using a path relative to the script."""
    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent
    # Construct the full path to the CSS file
    css_path = script_dir / file_name
    try:
        # Open using the absolute path
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at expected location: {css_path}. Please ensure {file_name} is in the same directory as app.py.")
    except Exception as e:
        st.error(f"Error loading CSS file {css_path}: {e}")

def generate_image_from_backend(
    prompt: str,
    # Note: Backend currently only uses the prompt. These are kept for future potential expansion.
    steps: int,
    scale: float,
    width: int,
    height: int,
    seed: Optional[int] = None,
) -> Optional[bytes]:
    """
    Sends the generation request (currently only prompt) to the backend 
    and returns the image bytes.

    Args:
        prompt: The text prompt for image generation.
        steps: Number of inference steps (currently ignored by backend).
        scale: Guidance scale (currently ignored by backend).
        width: Image width (currently ignored by backend).
        height: Image height (currently ignored by backend).
        seed: Optional random seed (currently ignored by backend).

    Returns:
        Image data as bytes if successful, None otherwise.
    """
    # Prepare the payload - include all settings now
    payload = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": scale,
        "width": width,
        "height": height,
    }
    # Only include seed if it's provided (not None)
    if seed is not None:
        payload["seed"] = seed
        
    
    

    try:
        # Use st.secrets or environment variables for the URL in production/deployment
        # backend_url = st.secrets.get("BACKEND_URL", BACKEND_URL) 
        backend_url = BACKEND_API_URL # Using hardcoded for now, as per initial setup

        # Send POST request, expect raw bytes in response
        response = requests.post(
            f"{backend_url}/generate",
            json=payload,
            timeout=180,
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check content type to ensure it's an image
        content_type = response.headers.get('content-type')
        if content_type and 'image' in content_type:
            # Read the raw image bytes directly from the response content
            image_bytes = response.content
            st.success("Your Naruto image is ready!")
            return image_bytes
        else:
             st.error(f"Backend returned unexpected content type: {content_type}. Expected an image.")
             # Optionally log the first few bytes/text of the response for debugging
             # try:
             #    st.error(f"Backend response text (first 100 chars): {response.text[:100]}")
             # except Exception:
             #    st.error("Could not decode backend response text.")
             return None

    except requests.exceptions.Timeout:
        st.error(f"Connection to backend timed out ({BACKEND_API_URL}). The image generation might be taking too long or the backend might be down.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the backend at {backend_url}. Is it running? Please ensure the backend service is started.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Backend request failed: {e}")
        # Log detailed error from backend if available
        if e.response is not None:
            try:
                backend_error_details = e.response.json() # Assuming backend sends JSON on error
                st.error(f"Backend error details: {backend_error_details}")
            except Exception: # If response is not JSON or decoding fails
                 st.error(f"Backend response (status {e.response.status_code}): {e.response.text[:500]}") # Show raw text
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during backend communication: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="Naruto Image Generator", layout="wide")

# Load custom CSS - Pass only the filename
local_css("style.css") 

# Initialize session state for gallery if it doesn't exist
if 'gallery' not in st.session_state:
    st.session_state.gallery = [] # List to store image bytes

# Optional: Add a header image or logo
# st.image("path/to/naruto_logo.png", width=200) 

st.title("🍥 Naruto Style Image Generator 🍥")
# Removed warning about unused settings
st.markdown("Enter a prompt and adjust settings to generate a Naruto-style image!")

# Use columns for layout - Adjusted Ratio
col1, col2 = st.columns([1.2, 1]) # Give settings slightly more weight

with col1:
    st.subheader("🎨 Generation Settings")

    # Prompt Input (Feature #2)
    prompt = st.text_area(
        "Enter your prompt:",
        height=100,
        placeholder="e.g., Naruto using Rasengan in the Hidden Leaf Village, anime style"
    )

    # Generation Settings in expander (Feature #3)
    # Updated expander label
    with st.expander("Advanced Settings", expanded=False):
        steps = st.slider("Inference Steps:", min_value=15, max_value=50, value=25, step=1, help="More steps can improve quality but take longer.")
        scale = st.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=7.5, step=0.5, help="How closely the image follows the prompt (higher value = stricter).")

        # Image Dimensions - Added 256x256
        dimensions_options: Dict[str, Tuple[int, int]] = {
            "256 x 256": (256, 256),
            "512 x 512": (512, 512),
            "768 x 768": (768, 768),
            "512 x 768": (512, 768),
            "768 x 512": (768, 512),
            # Add more options if backend supports them
        }
        # Set default to 512x512
        default_dimension_index = list(dimensions_options.keys()).index("512 x 512") 
        selected_dimension = st.selectbox(
            "Image Dimensions:", 
            options=list(dimensions_options.keys()), 
            index=default_dimension_index
        )
        width, height = dimensions_options[selected_dimension]

        # Seed Input
        seed_input = st.number_input(
            "Seed (optional):", 
            min_value=0, 
            max_value=2**32 - 1, # Max for many generators
            value=None, 
            step=1, 
            placeholder="Leave blank for random",
            help="Use the same seed and prompt for reproducible results."
        )

    # Generate Button (Feature #4)
    generate_button = st.button("Generate Image ✨", type="primary", use_container_width=True)

with col2:
    st.subheader("🖼️ Generated Image")

    # Placeholders for image and download button (Feature #6 & #7)
    image_placeholder = st.container() # Use container for better control
    download_placeholder = st.container()

# --- Generation Logic ---
if generate_button:
    if not prompt:
        st.warning("Please enter a prompt to generate an image.")
    else:
        # Clear previous results visually
        image_placeholder.empty()
        download_placeholder.empty()
        
        # Show loading indicator (Feature #5) - Updated text
        with st.spinner("🍥 Summoning Jutsu... Generating image... 🍥"):
            # Prepare seed value (use -1 or None if blank, depending on backend)
            # Assuming backend uses None for random seed - currently backend ignores it
            seed_to_send = seed_input if seed_input is not None else None
            
            # Call the updated backend function (passes other args, but they are ignored in the function's payload)
            image_data = generate_image_from_backend(
                prompt=prompt,
                steps=steps, # Passed but ignored by backend currently
                scale=scale, # Passed but ignored by backend currently
                width=width, # Passed but ignored by backend currently
                height=height,# Passed but ignored by backend currently
                seed=seed_to_send, # Passed but ignored by backend currently
            )

        # Display result or error (Feature #6 & #9)
        if image_data:
            # Add the new image to the beginning of the gallery list
            # Keep only the last 4 images
            st.session_state.gallery.insert(0, image_data) # Add to front
            st.session_state.gallery = st.session_state.gallery[:4] # Keep only top 4
            
            try:
                # Display image within the placeholder container
                with image_placeholder:
                    st.image(image_data, caption="Generated Image") 
                
                # --- Create Descriptive Filename --- 
                # Basic sanitization function for filenames
                def sanitize_filename(name):
                    # Remove invalid chars, replace spaces
                    name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).rstrip()
                    name = name.replace(" ", "_")
                    return name[:50] # Limit length
                
                # Extract keywords from prompt
                prompt_keywords = sanitize_filename(prompt)
                dims = f"{width}x{height}"
                seed_str = f"_seed{seed_to_send}" if seed_to_send is not None else ""
                
                # Construct filename
                download_filename = f"naruto_{prompt_keywords}_{dims}_{steps}s_{scale}g{seed_str}.png"
                # --- End Filename Creation ---

                # Add download button within its placeholder container (Feature #7)
                with download_placeholder:
                    st.download_button(
                        label="Download Image 💾",
                        data=image_data,
                        file_name=download_filename, # Use the new descriptive filename
                        mime="image/png",
                        type="secondary", 
                        help="Download the generated image as a PNG file.", 
                        use_container_width=True
                    )
            except Exception as display_error:
                st.error(f"Error displaying the generated image: {display_error}")
        else:
            # Error messages are now styled by CSS
            with image_placeholder:
                 st.error("Image generation failed. See messages above or check backend logs for details.")


# --- Recent Generations Gallery --- (Feature #8 Implemented)

st.divider()
st.subheader("🖼️ Recent Generations")

# Add a container with a specific class for styling
with st.container(border=False):
    # st.markdown('<div class="gallery-placeholder">', unsafe_allow_html=True) 

    if not st.session_state.gallery: # Check if gallery is empty
        st.caption("Your recent creations will appear here!")
    else:
        # Display gallery images inside expanders (vertical layout)
        for i, img_bytes in enumerate(st.session_state.gallery):
            generation_number = len(st.session_state.gallery) - i # Numbering: Newest is #1
            with st.expander(f"Generation {generation_number}", expanded=False):
                st.image(
                    img_bytes, 
                    caption=f"Generation {generation_number}"
                    # REMOVED use_container_width=True to display at native size
                )

    # Close the custom class div
    # st.markdown('</div>', unsafe_allow_html=True)

# --- Styling Notes --- (Feature #1 & #10)
# Custom styling applied via style.css
# Further styling can be added using st.markdown with unsafe_allow_html=True
# or by creating a frontend/style.css file and loading it.
# Consider adding a Naruto-themed background or color scheme.
# Example: Add custom CSS
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# local_css("frontend/style.css") # If you create style.css 