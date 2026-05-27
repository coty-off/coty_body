FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# core deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


# ---------------- WORKER ----------------
FROM base AS worker

COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY u2net.onnx /root/.u2net/u2net.onnx

CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "-P", "solo", "--concurrency=1"]


# ---------------- FLOWER ----------------
FROM python:3.11-slim AS flower

WORKDIR /app

COPY requirements-flower.txt .
RUN pip install --no-cache-dir -r requirements-flower.txt

COPY celery_app_flower.py .

CMD ["celery", "-A", "celery_app_flower", "flower", "--port=5555"]