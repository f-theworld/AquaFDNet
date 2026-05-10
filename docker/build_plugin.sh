#!/usr/bin/env bash
# 在 Linux + CUDA nvcc 环境下编译 FDConv TensorRT 插件为 .so（与 Windows 下 nvcc 链接目标一致）。
# 适用于 NGC `nvcr.io/nvidia/tensorrt:*-py3-devel` 等已含 TensorRT/CUDA 的镜像。
set -euo pipefail

PLUGIN_DIR="${PLUGIN_DIR:-/workspace/plugin}"
OUT_DIR="${OUT_DIR:-/opt/fdconv}"
OUT_SO="${OUT_SO:-${OUT_DIR}/libfdconv_trt_plugin.so}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

discover_trt_include() {
  local p
  for p in \
    "${TRT_ROOT:-}/include" \
    "/usr/src/tensorrt/include" \
    "/usr/include/x86_64-linux-gnu" \
    "/usr/local/tensorrt/include"; do
    if [[ -n "$p" && -f "$p/NvInfer.h" ]]; then
      echo "$p"
      return 0
    fi
  done
  # 兜底：在 /usr 下找 NvInfer.h
  p="$(find /usr -maxdepth 5 -name NvInfer.h 2>/dev/null | head -1 || true)"
  if [[ -n "$p" ]]; then
    dirname "$p"
    return 0
  fi
  echo "ERROR: NvInfer.h not found; set TRT_ROOT or use NGC tensorrt *-devel image." >&2
  return 1
}

discover_trt_lib() {
  local p
  for p in \
    "${TRT_ROOT:-}/lib" \
    "/usr/src/tensorrt/lib" \
    "/usr/lib/x86_64-linux-gnu" \
    "/usr/local/tensorrt/lib"; do
    if [[ -n "$p" && -f "$p/libnvinfer.so" ]]; then
      echo "$p"
      return 0
    fi
  done
  p="$(find /usr -maxdepth 5 -name 'libnvinfer.so*' 2>/dev/null | head -1 || true)"
  if [[ -n "$p" ]]; then
    dirname "$p"
    return 0
  fi
  echo "ERROR: libnvinfer.so not found." >&2
  return 1
}

TRT_INC="$(discover_trt_include)"
TRT_LIB="$(discover_trt_lib)"

mkdir -p "$(dirname "$OUT_SO")"
echo "[build_plugin] CUDA_HOME=$CUDA_HOME"
echo "[build_plugin] TRT_INC=$TRT_INC"
echo "[build_plugin] TRT_LIB=$TRT_LIB"
echo "[build_plugin] OUT_SO=$OUT_SO"

cd "$PLUGIN_DIR"

nvcc -std=c++17 -shared -Xcompiler -fPIC \
  complex_freq_conv_plugin.cpp complex_freq_conv_kernels.cu \
  -I"$TRT_INC" \
  -I"$CUDA_HOME/include" \
  -L"$TRT_LIB" \
  -L"$CUDA_HOME/lib64" \
  -lnvinfer -lnvinfer_plugin -lcudart -lcufft -lcublas -lcudnn \
  -o "$OUT_SO"

echo "[build_plugin] OK: $OUT_SO"
ls -la "$OUT_SO"
