"""
端侧最小推理：仅依赖 PyTorch(CUDA) + TensorRT，加载 .plan 与插件，跑一次随机输入并打印输出统计。
不依赖训练权重；用于无 ckpt 环境下的冒烟与延迟粗测。
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import sys
import time
from pathlib import Path


def _load_min_align(repo_root: Path):
    path = repo_root / "plugin" / "min_align_test.py"
    spec = importlib.util.spec_from_file_location("min_align_test_deploy", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=str, default="", help="仓库根目录（默认为本文件上两级）")
    p.add_argument("--engine", type=str, required=True, help="TensorRT 引擎文件路径（.plan）")
    p.add_argument("--plugin", type=str, required=True, help="fdconv 插件路径（.dll / .so）")
    p.add_argument("--input-shape", type=int, nargs=4, default=[1, 3, 256, 256], metavar=("N", "C", "H", "W"))
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sync", action="store_true", help="每次推理后 cuda synchronize（便于计时）")
    args = p.parse_args()

    repo = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    engine_path = Path(args.engine).resolve()
    plugin_path = Path(args.plugin).resolve()

    if not engine_path.is_file():
        raise FileNotFoundError(engine_path)
    if not plugin_path.is_file():
        raise FileNotFoundError(plugin_path)

    os.environ.setdefault("TENSORRT_ROOT", os.environ.get("TENSORRT_ROOT", ""))
    os.environ.setdefault("CUDA_PATH", os.environ.get("CUDA_PATH", ""))

    mat = _load_min_align(repo)
    mat.setup_windows_dll_search_path(plugin_path)

    if os.name != "nt":
        pass

    ctypes.CDLL(str(plugin_path))

    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    b, c, h, w = args.input_shape
    x = torch.randn(b, c, h, w, device="cuda", dtype=torch.float32)

    engine = mat.load_trt_engine_from_plan(engine_path)

    t0 = time.perf_counter()
    outs = mat.run_trt(engine, x)
    if args.sync:
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    for name, arr in outs.items():
        a = np.asarray(arr, dtype=np.float32)
        print(f"output={name} shape={a.shape} min={a.min():.6g} max={a.max():.6g} mean={a.mean():.6g}")
    print(f"infer_ok=1 wall_ms={elapsed_ms:.4f}")


if __name__ == "__main__":
    main()
