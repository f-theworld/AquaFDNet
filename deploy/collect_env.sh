#!/usr/bin/env bash
# 采集 Linux/Jetson 部署指纹（GPU、驱动、CUDA、TensorRT）。
# 用法: ./collect_env.sh [out.json]
set -euo pipefail

OUT_JSON="${1:-}"

collect() {
  python3 <<'PY'
import json, os, platform, subprocess, sys
from datetime import datetime, timezone

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""

data = {
    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    "os_description": platform.platform(),
    "machine": platform.machine(),
    "nvidia_smi": run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true"),
    "nvcc_version": run("nvcc --version 2>/dev/null | tail -1 || true"),
    "tensorrt_root": os.environ.get("TRT_ROOT", ""),
    "cuda_home": os.environ.get("CUDA_HOME", ""),
}
try:
    import tensorrt as trt
    data["tensorrt_python"] = trt.__version__
except Exception as e:
    data["tensorrt_python"] = f"import_failed: {e}"
print(json.dumps(data, indent=2))
PY
}

JSON="$(collect)"

echo "$JSON"

if [[ -n "$OUT_JSON" ]]; then
  mkdir -p "$(dirname "$OUT_JSON")"
  echo "$JSON" >"$OUT_JSON"
  echo "[collect_env] wrote $OUT_JSON" >&2
fi
