## Dockerfile for friendly-stable-audio-tools

# Pytorch 24.01 -> CUDA 12.3.2, PyTorch 2.2.0
FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update & \
    python -m pip install --upgrade pip

# friendly-stable-audio-tools (latest main)
RUN git clone https://github.com/yukara-ikemiya/friendly-stable-audio-tools.git
RUN cd friendly-stable-audio-tools && \
    pip install .

# To avoid Accelerate import error
RUN pip uninstall -y transformer-engine
