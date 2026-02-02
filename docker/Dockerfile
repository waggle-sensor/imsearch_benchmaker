FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-enabled torch stack first (pin versions to satisfy benchmaker)
# disable if you are not using a GPU
RUN pip install --no-cache-dir \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
# First install main requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Re-assert CUDA torch in case anything tried to swap it
# disable if you are not using a GPU
RUN pip install --no-cache-dir --upgrade --force-reinstall \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121


# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["/bin/bash"]

