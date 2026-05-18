FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENDO4DGS_REPO=/workspace/Endo-4DGS

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential ninja-build git ca-certificates \
        ffmpeg libgl1 libglib2.0-0 libx11-6 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade \
        pip==24.0 setuptools==69.5.1 wheel==0.43.0

RUN python -m pip install \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace/Endo-4DGS
COPY . /workspace/Endo-4DGS

# Docker build contexts do not always include initialized submodules. Fetch the
# two external modules needed by the iMED baseline when they are missing.
RUN if [ ! -f submodules/diff-gaussian-rasterization-depth/setup.py ]; then \
        rm -rf submodules/diff-gaussian-rasterization-depth && \
        git clone https://github.com/leo-frank/diff-gaussian-rasterization-depth submodules/diff-gaussian-rasterization-depth && \
        git -C submodules/diff-gaussian-rasterization-depth checkout 03f0b7d00383d6e96c22b37325ac9e5450947bf5; \
    fi && \
    if [ ! -d submodules/depth_anything ]; then \
        git clone https://github.com/fabio-sim/Depth-Anything-ONNX.git submodules/depth_anything && \
        git -C submodules/depth_anything checkout d6cc8e1e6713a2129683496b328bde675a66370d; \
    fi

RUN python -m pip install \
        numpy==1.24.4 \
        scipy==1.11.4 \
        matplotlib \
        tqdm \
        imageio[ffmpeg] \
        opencv-python-headless==4.7.0.72 \
        lpips \
        plyfile \
        pytorch_msssim \
        open3d \
        torchmetrics \
        onnx \
        onnxruntime \
        onnxruntime-gpu \
    && python -m pip install --no-build-isolation mmcv==1.6.0 \
    && python -m pip install -e submodules/diff-gaussian-rasterization-depth \
    && python -m pip install -e submodules/simple-knn

RUN chmod +x /workspace/Endo-4DGS/imed_nvs_baseline.py

ENTRYPOINT ["python", "/workspace/Endo-4DGS/imed_nvs_baseline.py"]
CMD ["--help"]
