#!/usr/bin/env bash
# 使用 trtexec 从 ONNX + 静态插件构建引擎（请在端侧执行，与目标 TRT 版本一致）。
set -euo pipefail

: "${TRT_ROOT:?Set TRT_ROOT to TensorRT root (contains bin/trtexec)}"
: "${ONNX_PATH:?Set ONNX_PATH}"
: "${PLUGIN_SO:?Set PLUGIN_SO path to libfdconv_trt_plugin.so}"
: "${ENGINE_OUT:?Set ENGINE_OUT path for .plan/.engine file}"

INPUT_SHAPE="${INPUT_SHAPE:-input:1x3x256x256}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

TRTEXEC="${TRT_ROOT}/bin/trtexec"
if [[ ! -x "$TRTEXEC" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" || ! -x "$TRTEXEC" ]]; then
  echo "ERROR: trtexec not found. Set TRT_ROOT or PATH." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${TRT_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "[build_engine_linux] TRTEXEC=$TRTEXEC"
echo "[build_engine_linux] ONNX=$ONNX_PATH PLUGIN=$PLUGIN_SO OUT=$ENGINE_OUT SHAPE=$INPUT_SHAPE"

exec "$TRTEXEC" \
  "--onnx=$ONNX_PATH" \
  "--staticPlugins=$PLUGIN_SO" \
  "--saveEngine=$ENGINE_OUT" \
  --verbose
