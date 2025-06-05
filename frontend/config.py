"""Configuration settings for the Naruto Image Generator frontend."""

import os

# Backend API Configuration
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "https://naruto-image-generator.onrender.com")

# Default Generation Parameters
DEFAULT_STEPS = 10
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_IMAGE_SIZE = "512x512"
DEFAULT_SEED = None

# UI Configuration
APP_TITLE = "Naruto Image Generator"
APP_ICON = "🍥"
MAX_TIMEOUT = 180  # seconds

# Model Information
MODELS = {
    "finetuned": {
        "name": "Fine-tuned Naruto Model",
        "description": "Specialized model trained on Naruto artwork for authentic style generation"
    },
    "base": {
        "name": "Stable Diffusion Base Model", 
        "description": "General-purpose model for artistic interpretation"
    }
} 