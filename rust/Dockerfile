# Use Rust base image
FROM rust:bullseye

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt update && apt install -y \
    wget \
    gnupg2 \
    lsb-release \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    clang \
    llvm \
    lld \
    python3 \
    python3-pip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA Repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt update

# Install CUDA Toolkit (includes NVCC)
RUN apt install -y cuda-toolkit-12-3

# Set Environment Variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


RUN echo -e '#!/bin/bash\necho \"GPU 0: Fake GPU (UUID: GPU-fake)\"" > /usr/bin/nvidia-smi && \
    chmod +x /usr/bin/nvidia-smi


# Verify NVCC installation
RUN nvcc --version
