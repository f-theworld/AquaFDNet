#!/usr/bin/env bash
# 在 Linux / NGC TensorRT-devel 容器中编译插件为 .so（调用仓库 docker/build_plugin.sh）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PLUGIN_DIR="${PLUGIN_DIR:-$ROOT/plugin}"
export OUT_DIR="${OUT_DIR:-$ROOT/deploy/dist}"
export OUT_SO="${OUT_SO:-$OUT_DIR/libfdconv_trt_plugin.so}"
mkdir -p "$(dirname "$OUT_SO")"
exec bash "$ROOT/docker/build_plugin.sh"
