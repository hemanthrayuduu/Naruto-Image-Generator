import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import generate_image
from fastapi.responses import StreamingResponse
from io import BytesIO
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Naruto Image Generator API",
    description="Generate Naruto-style images using fine-tuned Stable Diffusion",
    version="1.0.0"
)

# CORS configuration from environment variables
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = int(os.environ.get("DEFAULT_STEPS", 20))
    guidance_scale: Optional[float] = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", 7.5))
    width: Optional[int] = int(os.environ.get("DEFAULT_WIDTH", 512))
    height: Optional[int] = int(os.environ.get("DEFAULT_HEIGHT", 512))
    seed: Optional[int] = None
    model_type: Optional[str] = "finetuned"  # "finetuned" or "base"

@app.get("/")
def read_root():
    return {
        "message": "Naruto Image Generator API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "naruto-image-generator"}

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        logger.info(f"Generating image with prompt: {request.prompt[:50]}...")
        
        # Validate parameters
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.num_inference_steps < 1 or request.num_inference_steps > 100:
            raise HTTPException(status_code=400, detail="Steps must be between 1 and 100")
        
        if request.guidance_scale < 1.0 or request.guidance_scale > 20.0:
            raise HTTPException(status_code=400, detail="Guidance scale must be between 1.0 and 20.0")
        
        # Generate image
        image = generate_image(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed
        )
        
        # Convert to response
        buf = BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        
        logger.info("Image generated successfully")
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
