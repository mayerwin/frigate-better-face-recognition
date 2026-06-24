# frigate-better-face-recognition
# Single-stage image. insightface (SCRFD + ArcFace) needs a C/C++ toolchain to
# build its extension and the OpenCV runtime libs; everything runs on CPU.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    HOST=0.0.0.0 \
    PORT=8975

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI (client only, no daemon) so the frigate-ext nginx integrator can
# `docker exec` into Frigate to inject its /faces button. Optional at runtime:
# without /var/run/docker.sock mounted it just logs manual-setup hints and the
# tool still serves its own UI. Official Docker apt repo => always current.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

EXPOSE 8975
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8975/api/healthz',timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:build_default_app", "--factory", "--host", "0.0.0.0", "--port", "8975"]
