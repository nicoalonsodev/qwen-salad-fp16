import os
import io
import base64
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from typing import Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen Image Edit 2511 API")
pipeline = None

# Resolución máxima que procesa el modelo (balance VRAM/calidad)
MODEL_MAX_SIZE = 768


class EditRequest(BaseModel):
    image: Union[str, list[str]]  # Un base64 o lista de base64
    prompt: str
    negative_prompt: str = " "
    num_inference_steps: int = 25
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: int = 0


class EditResponse(BaseModel):
    image: str
    processing_time: float
    gpu_memory_gb: float


def decode_image(img_b64: str) -> Image.Image:
    """Decodifica un base64 a PIL Image, limpiando prefijo si existe."""
    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def resize_for_model(img: Image.Image, max_size: int = MODEL_MAX_SIZE) -> Image.Image:
    """Redimensiona al max_size manteniendo aspect ratio. Múltiplos de 8."""
    w, h = img.size
    if max(w, h) <= max_size:
        # Igual aseguramos múltiplos de 8
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w == w and new_h == h:
            return img
        return img.resize((new_w, new_h), Image.LANCZOS)
    ratio = max_size / max(w, h)
    new_w = (int(w * ratio) // 8) * 8
    new_h = (int(h * ratio) // 8) * 8
    logger.info(f"   Resize: {w}x{h} → {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)


@app.on_event("startup")
async def load_model():
    global pipeline

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logger.info("🚀 Iniciando carga de Qwen-Image-Edit-2511 (full precision)...")
    start = time.time()

    try:
        from diffusers import QwenImageEditPlusPipeline

        model_id = "Qwen/Qwen-Image-Edit-2511"

        logger.info("📥 Descargando modelo...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

        logger.info("⚙️ Cargando modelo en GPU (sin offload)...")
        pipeline.to("cuda")
        pipeline.enable_attention_slicing(1)

        load_time = time.time() - start
        logger.info(f"✅ Modelo listo en {load_time:.1f}s")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"🎮 GPU: {gpu_name} ({gpu_mem_total:.1f}GB)")
            logger.info(f"💾 VRAM usada: {gpu_mem_used:.1f}GB / {gpu_mem_total:.1f}GB")

    except Exception as e:
        logger.error(f"❌ Error al cargar modelo: {e}")
        raise


@app.get("/")
def root():
    return {
        "service": "Qwen-Image-Edit-2511 API (full precision)",
        "status": "ready" if pipeline is not None else "loading",
    }


@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo aún cargando")
    return {"status": "healthy", "model_loaded": True}


@app.post("/edit", response_model=EditResponse)
async def edit_image(req: EditRequest):
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo aún cargando")

    start_time = time.time()

    try:
        logger.info(f"📸 Procesando: '{req.prompt[:80]}...'")

        # Decodificar imagen(es)
        try:
            if isinstance(req.image, list):
                images_original = [decode_image(img) for img in req.image]
                logger.info(f"   {len(images_original)} imágenes: {[img.size for img in images_original]}")
                # Guardar tamaño original de la primera imagen para upscale final
                original_size = images_original[0].size
                images = [resize_for_model(img) for img in images_original]
            else:
                image_original = decode_image(req.image)
                original_size = image_original.size
                logger.info(f"   Tamaño original: {original_size}")
                images = resize_for_model(image_original)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Imagen inválida: {str(e)}")

        generator = torch.Generator(device="cuda").manual_seed(req.seed)

        logger.info(f"   Steps: {req.num_inference_steps}, true_cfg: {req.true_cfg_scale}")

        with torch.inference_mode():
            output = pipeline(
                image=images,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                true_cfg_scale=req.true_cfg_scale,
                guidance_scale=req.guidance_scale,
                generator=generator,
            )

        result_image = output.images[0]

        # Upscale al tamaño original del input
        if result_image.size != original_size:
            logger.info(f"   Upscale: {result_image.size} → {original_size}")
            result_image = result_image.resize(original_size, Image.LANCZOS)

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        result_b64 = base64.b64encode(buffer.getvalue()).decode()

        process_time = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

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
