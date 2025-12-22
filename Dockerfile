# 基础镜像请根据本地 CUDA 驱动版本调整（示例基于 CUDA 12.1）
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# 基本依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv git curl build-essential \
    libssl-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

# pip 安装（在容器构建时会安装 PyTorch 等）
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# 拷贝代码（在构建镜像时可覆盖）
COPY . /workspace

ENV PYTHONPATH=/workspace:${PYTHONPATH}
CMD ["/bin/bash"]