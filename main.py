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
    image: str  # Base64 encoded (sin prefijo data:image/...)
    prompt: str
    negative_prompt: str = " "
    num_inference_steps: int = 28
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: int = 0


class EditResponse(BaseModel):
    image: str
    processing_time: float
    gpu_memory_gb: float


@app.on_event("startup")
async def load_model():
    """
    Carga el modelo Qwen-Image-Edit-2509.
    Usa enable_model_cpu_offload() para caber en 24GB de VRAM.
    Ref: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
    """
    global pipeline

    logger.info("🚀 Iniciando carga de Qwen-Image-Edit-2509...")
    start = time.time()

    try:
        from diffusers import QwenImageEditPlusPipeline

        model_id = "Qwen/Qwen-Image-Edit-2509"

        logger.info("📥 Descargando modelo (puede tardar varios minutos la primera vez)...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

        # SOLUCIÓN: En vez de .to("cuda") que carga TODO en VRAM,
        # usar enable_model_cpu_offload() que mueve componentes a GPU
        # solo cuando se necesitan. Esto permite correr en 24GB.
        logger.info("⚙️ Configurando CPU offload para caber en 24GB VRAM...")
        pipeline.enable_model_cpu_offload()
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
        raise


@app.get("/")
def root():
    return {
        "service": "Qwen-Image-Edit-2509 API",
        "model": "Qwen/Qwen-Image-Edit-2509",
        "pipeline": "QwenImageEditPlusPipeline",
        "precision": "bfloat16",
        "offload": "cpu_offload (24GB mode)",
        "status": "ready" if pipeline is not None else "loading",
        "endpoints": {
            "health": "/health",
            "edit": "/edit",
        },
    }


@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo aún cargando")

    gpu_mem = (
        torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    )
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_memory_gb": round(gpu_mem, 2),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/edit", response_model=EditResponse)
async def edit_image(req: EditRequest):
    """
    Endpoint principal para editar imágenes.
    Usa CPU offload — cada componente se mueve a GPU solo cuando se usa.
    """
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo aún cargando")

    start_time = time.time()

    try:
        logger.info(f"📸 Procesando: '{req.prompt}'")

        try:
            image_data = req.image
            if "," in image_data:
                image_data = image_data.split(",")[1]

            img_bytes = base64.b64decode(image_data)
            input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            logger.info(f"   Tamaño: {input_image.size}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Imagen inválida: {str(e)}")

        generator = torch.Generator(device="cpu").manual_seed(req.seed)

        logger.info(
            f"   Steps: {req.num_inference_steps}, true_cfg: {req.true_cfg_scale}"
        )

        with torch.inference_mode():
            output = pipeline(
                image=input_image,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                true_cfg_scale=req.true_cfg_scale,
                guidance_scale=req.guidance_scale,
                generator=generator,
            )

        result_image = output.images[0]

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        result_b64 = base64.b64encode(buffer.getvalue()).decode()

        process_time = time.time() - start_time
        gpu_mem = (
            torch.cuda.memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0
        )

        logger.info(f"✅ Listo en {process_time:.2f}s | VRAM: {gpu_mem:.2f}GB")

        return EditResponse(
            image=result_b64,
            processing_time=process_time,
            gpu_memory_gb=round(gpu_mem, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
