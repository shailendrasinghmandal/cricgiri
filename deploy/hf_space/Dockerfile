# CricGiri Delivery API — Hugging Face Space (free, always-on, CPU).
# HF Spaces build this Dockerfile and expose the port named in README.md (app_port: 7860).
FROM python:3.11-slim-bookworm

# System libs OpenCV / ffmpeg need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as UID 1000. Create a matching user so the app
# can write its temp uploads/outputs.
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /app

# 1) CPU-only PyTorch first (small, no CUDA) so ultralytics doesn't pull the GPU build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# 2) The rest of the dependencies.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3) Application code + models (staged by build_space.ps1 into the Space repo).
COPY . .
RUN mkdir -p uploads outputs videos logs && chown -R user:user /app
USER user

# Delivery-API pipeline config. imgsz 1280 = the tested accuracy; drop to 960 in the
# Space "Settings → Variables" (CRICGIRI_IMGSZ) if CPU analysis is too slow for you.
ENV PYTHONUNBUFFERED=1 \
    CRICGIRI_BALL_MODELS=ball_ft_t4.pt,ball_best_leather_new.pt \
    CRICGIRI_IMGSZ=1280 \
    CRICGIRI_CONF=0.05 \
    CRICGIRI_PITCH_LEN=20.12

EXPOSE 7860

# One worker (model singleton; the app serialises inference). Long keep-alive because
# CPU inference on a full clip can take a couple of minutes.
CMD ["uvicorn", "api.delivery_api:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--timeout-keep-alive", "300"]
