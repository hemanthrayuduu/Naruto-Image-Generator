from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import generate_image
from fastapi.responses import StreamingResponse
from io import BytesIO
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None

@app.post("/generate")
def generate(request: GenerationRequest):
    image = generate_image(
        prompt=request.prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        width=request.width,
        height=request.height,
        seed=request.seed
    )
    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
