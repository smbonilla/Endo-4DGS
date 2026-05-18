FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential ninja-build git ca-certificates \
        ffmpeg libgl1 libglib2.0-0 libx11-6 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0 \
    && python -m pip install \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install --no-build-isolation -r /app/requirements.txt

# Common 3DGS-style NVS dependencies. These are installed at build time so the
# submitted image can run without network access during evaluation.
RUN mkdir -p /opt/imed_nvs_submodules \
    && git clone https://github.com/leo-frank/diff-gaussian-rasterization-depth \
        /opt/imed_nvs_submodules/diff-gaussian-rasterization-depth \
    && git -C /opt/imed_nvs_submodules/diff-gaussian-rasterization-depth \
        checkout 03f0b7d00383d6e96c22b37325ac9e5450947bf5 \
    && git clone https://gitlab.inria.fr/bkerbl/simple-knn.git \
        /opt/imed_nvs_submodules/simple-knn \
    && git clone https://github.com/fabio-sim/Depth-Anything-ONNX.git \
        /opt/imed_nvs_submodules/depth_anything \
    && git -C /opt/imed_nvs_submodules/depth_anything \
        checkout d6cc8e1e6713a2129683496b328bde675a66370d \
    && python -m pip install -e /opt/imed_nvs_submodules/diff-gaussian-rasterization-depth \
    && python -m pip install -e /opt/imed_nvs_submodules/simple-knn

# IMPORTANT: any additional model weights your method needs must be downloaded
# or copied here at build time. The challenge evaluator may run the container
# without network access.

COPY entrypoint.py nvs_method.py /app/

ENTRYPOINT ["python", "/app/entrypoint.py"]
CMD ["--input", "/input", "--output", "/output"]
