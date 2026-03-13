import requests
import base64

url = "http://localhost:8000"

# Health check
print("Health:", requests.get(f"{url}/health").json())

# Test con imagen
with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(f"{url}/edit", json={
    "image": img_b64,
    "prompt": "make it cyberpunk style, neon lights, futuristic",
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
})

result = response.json()
print(f"Tiempo: {result['processing_time']:.2f}s")
print(f"VRAM: {result['gpu_memory_gb']:.2f} GB")

# Guardar resultado
with open("result.png", "wb") as f:
    f.write(base64.b64decode(result["image"]))
print("✅ Resultado guardado en result.png")