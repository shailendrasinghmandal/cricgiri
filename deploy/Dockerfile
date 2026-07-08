# CricGiri Analytics API — production container
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-api.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

RUN mkdir -p uploads outputs/api logs models videos

ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
# Free-tier engine config for the Delivery API: committed small model, lighter imgsz.
ENV CRICGIRI_BALL_MODELS=ball_best.pt
ENV CRICGIRI_IMGSZ=640
ENV CRICGIRI_PITCH_LEN=20.12

EXPOSE 8000

# Serve the Delivery Analytics API (POST /analyze -> delivery JSON). Single worker
# (model singleton; app serialises inference). Long keep-alive for slow CPU inference.
CMD ["sh", "-c", "uvicorn api.delivery_api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 300"]
