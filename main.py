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

app = FastAPI(title="Qwen Image Edit 2509 API")
pipeline = None

class EditRequest(BaseModel):
    image: str  # Base64 encoded
    prompt: str
    negative_prompt: str = " "
    num_inference_steps: int = 25  # 25 es rápido, 50 es máxima calidad
    true_cfg_scale: float = 4.0    # Parámetro específico de Qwen
    guidance_scale: float = 1.0    # Diferente a Stable Diffusion
    seed: int = 0

class EditResponse(BaseModel):
    image: str
    processing_time: float
    gpu_memory_gb: float

@app.on_event("startup")
async def load_model():
    """
    Carga el modelo Qwen-Image-Edit-2509.
    Basado en: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
    """
    global pipeline
    
    logger.info("🚀 Iniciando carga de Qwen-Image-Edit-2509...")
    start = time.time()
    
    try:
        from diffusers import QwenImageEditPipeline
        
        model_id = "Qwen/Qwen-Image-Edit-2509"
        
        # Cargar con device_map="auto" para distribución automática entre CPUs y GPUs
        logger.info("📥 Descargando modelo...")
        pipeline = QwenImageEditPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Usar bfloat16 para mejor estabilidad que float16
            device_map="auto",            # Distribución automática (no hacer .to("cuda") después)
        )
        
        # Optimizaciones para VRAM
        logger.info("⚙️ Aplicando optimizaciones...")
        pipeline.enable_attention_slicing(1)
        
        load_time = time.time() - start
        logger.info(f"✅ Modelo listo en {load_time:.1f}s")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU: {gpu_name}")
            logger.info(f"💾 VRAM: {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB")
        
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo: {e}")
        logger.error(f"   Este error impide que el servicio funcione.")
        raise

@app.get("/")
def root():
    """Endpoint raíz con info del servicio"""
    return {
        "service": "Qwen-Image-Edit-2509 API",
        "model": "Qwen/Qwen-Image-Edit-2509",
        "precision": "bfloat16",
        "status": "ready" if pipeline is not None else "loading",
        "endpoints": {
            "health": "/health",
            "edit": "/edit",
        }
    }

@app.get("/health")
def health():
    """
    Health check para SaladCloud Readiness Probe.
    Retorna 200 solo si el modelo está cargado y listo.
    """
    if pipeline is None:
        logger.error("❌ Health check fallido: pipeline no está cargado")
        raise HTTPException(status_code=503, detail="Modelo aún cargando")
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    return {
        "status": "healthy",
        "model": "Qwen-Image-Edit-2509",
        "precision": "bfloat16",
        "gpu_memory_gb": round(gpu_mem, 2),
        "cuda_available": torch.cuda.is_available(),
    }

@app.post("/edit", response_model=EditResponse)
async def edit_image(req: EditRequest):
    """
    Endpoint principal para editar imágenes.
    
    Parámetros específicos de Qwen-Image-Edit-2509:
    - true_cfg_scale: Controla qué tanto seguir el prompt (recomendado: 4.0)
    - guidance_scale: NO use el valor estándar de Stable Diffusion (7.5)
    - num_inference_steps: 25 (rápido) o 50 (máxima calidad)
    
    Basado en: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
    """
    global pipeline
    
    if pipeline is None:
        logger.error("❌ Edit request fallido: pipeline no inicializado")
        raise HTTPException(status_code=503, detail="Modelo aún cargando")
    
    start_time = time.time()
    
    try:
        # 1. Decodificar imagen base64
        logger.info(f"📸 Procesando imagen con prompt: '{req.prompt}'")
        
        try:
            # Limpiar prefijo data:image/... si existe
            image_data = req.image
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            img_bytes = base64.b64decode(image_data)
            input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            logger.info(f"   Tamaño de imagen: {input_image.size}")
        except Exception as e:
            logger.error(f"❌ Error decodificando imagen: {e}")
            raise HTTPException(status_code=400, detail=f"Imagen inválida: {str(e)}")
        
        # 2. Generar con parámetros correctos para Qwen
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(req.seed)
        
        logger.info(f"   Steps: {req.num_inference_steps}, true_cfg: {req.true_cfg_scale}")
        
        # Usar inference_mode para optimizar VRAM
        with torch.inference_mode():
            output = pipeline(
                image=input_image,  # Qwen acepta una imagen (no lista)
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                true_cfg_scale=req.true_cfg_scale,      # IMPORTANTE: específico de Qwen
                guidance_scale=req.guidance_scale,       # NO cambiar a 7.5
                generator=generator,
            )
        
        result_image = output.images[0]
        
        # 3. Codificar resultado a base64
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        process_time = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        logger.info(f"✅ Edición exitosa en {process_time:.2f}s")
        logger.info(f"   VRAM usado: {gpu_mem:.2f}GB")
        
        return EditResponse(
            image=result_b64,
            processing_time=process_time,
            gpu_memory_gb=round(gpu_mem, 2)
        )
        
    except HTTPException:
        raise  # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"❌ Error durante edición: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
