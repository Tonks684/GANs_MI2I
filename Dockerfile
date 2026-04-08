# ── Base image: PyTorch 2.0.1 + CUDA 11.8 (matches the conda environment) ──
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

LABEL maintainer="Samuel Tonks"
LABEL description="GANs_MI2I — Pix2PixHD virtual fluorescence staining"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer-cached)
COPY pyproject.toml ./
# Install CPU-agnostic dependencies; torch/torchvision are already in the base image
RUN pip install --no-cache-dir \
        tifffile \
        scikit-image \
        scipy \
        pandas \
        tqdm \
        pillow \
        dominate \
        pyyaml \
        zarr \
        tensorboard \
        wandb \
        pytest \
        pytest-cov \
        ruff

# Copy source
COPY . .

# Install the package in editable mode (no-deps: torch already present)
RUN pip install --no-cache-dir --no-deps -e .

# Default: run tests (override CMD for training/inference)
CMD ["pytest", "tests/", "-v", "--tb=short", "--ignore=tests/test_data_loader.py"]
