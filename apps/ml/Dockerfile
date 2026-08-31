FROM python:3.11-slim

WORKDIR /workspace

# LightGBM requires libgomp for OpenMP threading support
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code is bind-mounted at runtime via docker-compose
CMD ["tail", "-f", "/dev/null"]
