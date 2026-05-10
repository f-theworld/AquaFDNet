# syntax=docker/dockerfile:1
# Linux + GPU + TensorRT（两阶段）：
# - builder: 具备 nvcc 的 CUDA devel 镜像，用于编译 FDConv 插件 .so
# - runtime: TensorRT py3 运行镜像，承载 trtexec / Python 推理
#
# 设计目的：避免依赖 `tensorrt:*devel` 标签（在部分 NGC 账号/区域不可用）。
#
# 版本策略（避免再踩坑）：
# - 先装 torch/torchvision（cu121），再装 lpips；否则 lpips 会从默认 PyPI 拉 CPU/错误 CUDA 的 torch。
# - numpy 钉死 1.26.4（与 torch 2.2.2 预编译轮一致），禁止升到 NumPy 2。
# - opencv / onnx / matplotlib / timm / piq / lpips 使用精确版本（与 FDConv/AquaFDNet 顶层 import 一致）。

# ARG BUILD_IMAGE=nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG BUILD_IMAGE=nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ARG RUNTIME_IMAGE=nvcr.io/nvidia/tensorrt:24.02-py3

FROM ${RUNTIME_IMAGE} AS trt_runtime_files

FROM ${BUILD_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/tensorrt/include /opt/tensorrt/lib
COPY --from=trt_runtime_files /usr/include/x86_64-linux-gnu/NvInfer*.h /opt/tensorrt/include/
COPY --from=trt_runtime_files /usr/include/x86_64-linux-gnu/NvOnnxParser*.h /opt/tensorrt/include/
COPY --from=trt_runtime_files /usr/lib/x86_64-linux-gnu/libnvinfer*.so* /opt/tensorrt/lib/
COPY --from=trt_runtime_files /usr/lib/x86_64-linux-gnu/libnvonnxparser*.so* /opt/tensorrt/lib/

WORKDIR /workspace
COPY . /workspace

RUN chmod +x docker/build_plugin.sh \
    && PLUGIN_DIR=/workspace/plugin TRT_ROOT=/opt/tensorrt OUT_SO=/tmp/libfdconv_trt_plugin.so docker/build_plugin.sh

FROM ${RUNTIME_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ① PyTorch 必须从官方 cu121 索引安装（与 TensorRT 24.02 所用 CUDA 12.x 驱动兼容）。
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.2.2" \
    "torchvision==0.17.2"

# ② 其余 Python 依赖：精确版本 + numpy 1.x；末尾再次钉 numpy，抵消偶发依赖链升级。
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "opencv-python-headless==4.9.0.80" \
    "matplotlib==3.8.4" \
    "timm==0.9.16" \
    "onnx==1.16.2" \
    "piq==0.8.0" \
    "lpips==0.1.4" \
    && pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.26.4"

# ③ 构建期自检：版本 + ONNX 导出常用依赖（避免镜像建好一跑才缺包）。
RUN python3 -c "\
import numpy as np; \
import torch; \
import tensorrt as trt; \
import cv2; \
import matplotlib; \
import timm; \
from timm.models.layers import trunc_normal_; \
assert not np.__version__.startswith('2.'), np.__version__; \
assert torch.__version__.startswith('2.2.'), torch.__version__; \
_ = trunc_normal_; \
print('deps_ok numpy=', np.__version__, 'torch=', torch.__version__, 'tensorrt=', trt.__version__, 'timm=', timm.__version__)"

WORKDIR /workspace
COPY . /workspace

# ④ 仓库代码可导入（与推荐命令 `python3 /workspace/export_full_onnx.py` 一致；勿依赖 plugin 下重复副本）
RUN cd /workspace && PYTHONPATH=/workspace python3 -c "from AquaFDNet import AquaFDNet; print('import_ok AquaFDNet')"

RUN mkdir -p /opt/fdconv
COPY --from=builder /tmp/libfdconv_trt_plugin.so /opt/fdconv/libfdconv_trt_plugin.so
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcudnn*.so.8* /usr/lib/x86_64-linux-gnu/
RUN ldconfig

ENV LD_LIBRARY_PATH="/opt/fdconv:${LD_LIBRARY_PATH:-}"

CMD ["bash", "-lc", "trtexec --help | head -n 5 && python3 -c \"import tensorrt as trt; print('tensorrt', trt.__version__)\""]
