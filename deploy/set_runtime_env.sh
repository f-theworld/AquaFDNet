#!/usr/bin/env bash
# Linux: 将 TensorRT / CUDA 库加入 LD_LIBRARY_PATH（按端侧安装路径修改 TRT_ROOT / CUDA_HOME）。
export TRT_ROOT="${TRT_ROOT:-/usr/src/tensorrt}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "[set_runtime_env] LD_LIBRARY_PATH head: ${LD_LIBRARY_PATH:0:120}..."
