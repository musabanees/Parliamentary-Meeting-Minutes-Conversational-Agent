# ---- builder stage ----
    FROM python:3.12-slim AS builder

    WORKDIR /app

    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1

    # Only needed if some deps must compile. If not, you can remove this whole RUN.
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
      && rm -rf /var/lib/apt/lists/*

    COPY requirements.txt .

    # Install dependencies into /install so we can copy them into runtime image
    RUN pip install --upgrade pip \
     && pip install --no-cache-dir --prefer-binary --prefix=/install -r requirements.txt


    # ---- runtime stage ----
    FROM python:3.12-slim AS runtime

    WORKDIR /app

    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1

    # Non-root user
    RUN useradd -m appuser

    # Copy installed python packages from builder
    COPY --from=builder /install /usr/local

    # Copy only required app files
    COPY main.py ./main.py
    COPY params.yaml ./params.yaml
    COPY parliament_agent/ ./parliament_agent/
    COPY data/ ./data/

    # If you truly need evaluation inside the container, keep these.
    # Otherwise, comment them out to reduce image size.
    COPY evaluation/scripts/ ./evaluation/scripts/
    COPY evaluation/dataset/ ./evaluation/dataset/

    EXPOSE 8000

    USER appuser

    CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]