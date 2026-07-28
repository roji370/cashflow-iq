FROM python:3.11-slim

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code is bind-mounted at runtime via docker-compose
CMD ["tail", "-f", "/dev/null"]
