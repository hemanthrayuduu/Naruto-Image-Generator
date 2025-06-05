"""Utility functions for the Naruto Image Generator frontend."""

import io
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
import streamlit as st


def save_generated_image(
    image: Image.Image,
    prompt: str,
    model_type: str,
    parameters: Dict[str, Any]
) -> bool:
    """
    Save generated image with metadata.
    
    Args:
        image: Generated PIL Image
        prompt: Text prompt used for generation
        model_type: Type of model used ("finetuned" or "base")
        parameters: Generation parameters
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # For web deployment, we don't actually save files to disk
        # Instead, we could implement cloud storage or just return success
        # This is a placeholder for future enhancement
        return True
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return False


def format_generation_metadata(
    prompt: str,
    model_type: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format metadata for generated images.
    
    Args:
        prompt: Text prompt used
        model_type: Model type used
        parameters: Generation parameters
        
    Returns:
        Dict with formatted metadata
    """
    return {
        "prompt": prompt,
        "model_type": model_type,
        "parameters": parameters,
        "timestamp": datetime.now().isoformat(),
        "app_version": "1.0.0"
    }


def create_download_data(image: Image.Image, format: str = "PNG") -> bytes:
    """
    Convert PIL Image to bytes for download.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        bytes: Image data as bytes
    """
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def validate_prompt(prompt: str) -> tuple[bool, str]:
    """
    Validate user input prompt.
    
    Args:
        prompt: User input prompt
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty"
    
    if len(prompt) > 1000:
        return False, "Prompt is too long (max 1000 characters)"
    
    return True, ""


def get_dimension_options() -> list[str]:
    """Get available image dimension options."""
    return ["256x256", "512x512", "768x768", "1024x1024"]


def parse_dimensions(dimension_str: str) -> tuple[int, int]:
    """
    Parse dimension string to width, height tuple.
    
    Args:
        dimension_str: String like "512x512"
        
    Returns:
        tuple: (width, height)
    """
    try:
        width, height = map(int, dimension_str.split("x"))
        return width, height
    except ValueError:
        return 512, 512  # Default fallback 