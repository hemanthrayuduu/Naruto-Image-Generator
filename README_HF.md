---
title: Naruto Image Generator
emoji: 🍃
colorFrom: orange
colorTo: blue
sdk: gradio
sdk_version: 4.15.0
app_file: app_hf.py
pinned: false
license: mit
hardware: t4-small
---

# Naruto Image Generator 🍃

Generate Naruto-style anime images using a fine-tuned Stable Diffusion model.

## Features

- **GPU-Accelerated**: Fast image generation (10-30 seconds)
- **Fine-tuned LoRA**: Specialized for Naruto-style characters
- **Customizable**: Adjust steps, guidance, size, and seed
- **User-friendly**: Simple Gradio interface

## Usage

1. Enter a prompt describing the image you want
2. Adjust generation parameters if needed  
3. Click "Generate Image"
4. Wait 10-30 seconds for your Naruto-style image!

## Examples

- "Naruto Uzumaki in sage mode, anime style"
- "Sasuke Uchiha with sharingan, dark aesthetic"  
- "Sakura Haruno in a cherry blossom field"

## Model Details

- **Base Model**: CompVis/stable-diffusion-v1-4
- **Fine-tuning**: LoRA adapters trained on Naruto dataset
- **Framework**: Diffusers, PEFT, PyTorch

## Deployment

This Space runs on T4 GPU for fast inference. The model automatically falls back to base Stable Diffusion if LoRA weights can't be loaded. 