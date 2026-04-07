FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all source code
COPY . /app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -lc 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

CMD ["sh", "-lc", "python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
