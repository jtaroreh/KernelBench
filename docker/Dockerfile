FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir google-antigravity requests python-dotenv

COPY cloud_worker.py /app/cloud_worker.py

ENTRYPOINT ["python", "/app/cloud_worker.py"]
