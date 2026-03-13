import os
import io
import base64
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen Image Edit 2509 API - FP16")

pipeline = None

class EditRequest(BaseModel):
    image: str
    prompt: str
    negative_prompt: str = " "
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: int = 0

class EditResponse(BaseModel):
    image: str
    processing_time: float
    gpu_memory_gb: float

@app.on_event("startup")
async def load_model():
    global pipeline
    
    logger.info("🚀 Cargando Qwen-Image-Edit-2509 en FP16...")
    logger.info("⚡ Calidad máxima - VRAM requerida: ~40-50GB")
    
    start = time.time()
    
    try:
        from diffusers import QwenImageEditPlusPipeline
        
        model_id = "Qwen/Qwen-Image-Edit-2509"
        
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="balanced",
        )
        
        pipeline = pipeline.to("cuda")
        pipeline.enable_attention_slicing(1)
        
        load_time = time.time() - start
        logger.info(f"✅ Pipeline listo en {load_time:.1f}s")
        
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 VRAM usada: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        raise

@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(503, "Modelo no cargado")
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    return {
        "status": "healthy",
        "model": "Qwen-Image-Edit-2509",
        "precision": "FP16",
        "quality": "maximum",
        "gpu_memory_gb": round(gpu_mem, 2),
    }

@app.post("/edit", response_model=EditResponse)
async def edit_image(req: EditRequest):
    global pipeline
    
    if pipeline is None:
        raise HTTPException(503, "Pipeline no inicializado")
    
    start_time = time.time()
    
    try:
        img_bytes = base64.b64decode(req.image)
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        generator = torch.manual_seed(req.seed)
        
        with torch.inference_mode():
            result = pipeline(
                image=[input_image],
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                true_cfg_scale=req.true_cfg_scale,
                guidance_scale=req.guidance_scale,
                generator=generator,
            ).images[0]
        
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        process_time = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        
        logger.info(f"✨ Edición completada en {process_time:.2f}s")
        
        return EditResponse(
            image=result_b64,
            processing_time=process_time,
            gpu_memory_gb=round(gpu_mem, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Error en edición: {e}")
        raise HTTPException(500, str(e))

@app.get("/")
def root():
    return {
        "service": "Qwen-Image-Edit-2509 API",
        "version": "FP16-MAX",
        "quality": "Maximum",
        "vram_required": "40-50GB",
    }