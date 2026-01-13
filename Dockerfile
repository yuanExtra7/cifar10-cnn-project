FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Keep logs unbuffered for easier `docker logs -f`
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install only training-time python deps; torch/torchvision come from base image
COPY requirements.docker.txt /workspace/requirements.docker.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /workspace/requirements.docker.txt

# Copy source code
COPY src/ /workspace/src/
COPY train.py /workspace/train.py
COPY README.md /workspace/README.md

# Default: run the training entry; pass args directly to container.
ENTRYPOINT ["python", "/workspace/train.py"]

