FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Forzar instalación de torch + torchvision + torchaudio compatibles ANTES del resto
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Instalar el resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Reinstalar torchvision AL FINAL — por si requirements.txt lo sobreescribió
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Verificar — si falla, el build falla (no llega una imagen rota a RunPod)
RUN python -c "import torch; import torchvision; print('torch:', torch.__version__, '| torchvision:', torchvision.__version__, '| CUDA:', torch.version.cuda)"

COPY main.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
