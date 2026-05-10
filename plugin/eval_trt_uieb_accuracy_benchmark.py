"""
与仓库根目录 `accuracy_performance_comparison.py` 对齐的 UIEB 子集基准：
- 同名配对 test_raw / test_reference，256 INTER_AREA，/255，batch=1
- 指标：PSNR / SSIM(piq) / LPIPS(alex) 相对参考图；延迟 P50/P95；峰值显存（torch 统计口径）

本脚本使用 **TensorRT 全图引擎**（`--precision fp32|fp16` 选择 `engines_matrix` 下对应 `.plan` + 插件 DLL）。

注意：在显存较紧的机器上，**先跑完 TRT 再加载 LPIPS**，避免与 FDConv 插件 workspace 争抢导致 `runComplexFreqConvMvpI32 failed`。

用法（请在仓库的 `plugin` 目录下执行，或自行传入绝对路径）：
  cd <path-to-repo>/plugin
  python eval_trt_uieb_accuracy_benchmark.py --precision fp32 \\
    --input-dir ../data/UIEB/test_raw --ref-dir ../data/UIEB/test_reference
  python eval_trt_uieb_accuracy_benchmark.py --precision fp16 \\
    --input-dir ../data/UIEB/test_raw --ref-dir ../data/UIEB/test_reference

默认 `--input-dir` / `--ref-dir` 与上例相同（相对当前工作目录）；数据集请按该相对路径放置或显式覆盖。
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_PLUGIN = Path(__file__).resolve().parent
_REPO = _PLUGIN.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_PLUGIN) not in sys.path:
    sys.path.insert(0, str(_PLUGIN))

from min_align_test import (  # noqa: E402
    load_trt_engine_from_plan,
    pick_aligned_output,
    run_trt,
    setup_windows_dll_search_path,
)


def load_img_rgb01(path: Path, size: int, device: torch.device) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def psnr_01(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).clamp(min=1e-10)
    return (20 * torch.log10(torch.tensor(1.0, device=a.device) / torch.sqrt(mse))).item()


def ssim_01(a: torch.Tensor, b: torch.Tensor) -> float:
    import piq

    return float(piq.ssim(a, b, data_range=1.0).item())


def main() -> int:
    ap = argparse.ArgumentParser(description="TRT full-graph UIEB accuracy/latency benchmark (FP32/FP16)")
    ap.add_argument(
        "--input-dir",
        type=str,
        default="../data/UIEB/test_raw",
        help="UIEB test_raw 图像目录（默认相对当前工作目录，常在 plugin/ 下运行）",
    )
    ap.add_argument(
        "--ref-dir",
        type=str,
        default="../data/UIEB/test_reference",
        help="UIEB test_reference 参考图目录（默认相对当前工作目录）",
    )
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--precision",
        type=str,
        choices=("fp32", "fp16"),
        default="fp32",
        help="选择默认引擎：fp32/fp16 对应 engines_matrix 下 AquaFDNet_full_*.plan",
    )
    ap.add_argument(
        "--engine",
        type=str,
        default="",
        help="显式指定 .plan 路径（若设置则覆盖 --precision 的默认引擎）",
    )
    ap.add_argument("--plugin", type=str, default="./fdconv_trt_plugin.dll")
    ap.add_argument(
        "--out-json",
        type=str,
        default="",
        help="默认 ./logs/trt_uieb_accuracy_{fp32|fp16}.json",
    )
    args = ap.parse_args()

    if not args.engine:
        args.engine = (
            "./engines_matrix/AquaFDNet_full_fp32.plan"
            if args.precision == "fp32"
            else "./engines_matrix/AquaFDNet_full_fp16.plan"
        )
    if not args.out_json:
        args.out_json = f"./logs/trt_uieb_accuracy_{args.precision}.json"

    device = torch.device("cuda")
    script_dir = _PLUGIN
    engine_path = (script_dir / args.engine).resolve()
    plugin_path = (script_dir / args.plugin).resolve()
    out_json = (script_dir / args.out_json).resolve()
    input_dir = Path(args.input_dir)
    ref_dir = Path(args.ref_dir)

    if not engine_path.is_file():
        print(f"missing engine: {engine_path}", file=sys.stderr)
        return 2
    if not plugin_path.is_file():
        print(f"missing plugin: {plugin_path}", file=sys.stderr)
        return 2

    in_map = {p.name: p for p in input_dir.iterdir() if p.is_file()}
    gt_map = {p.name: p for p in ref_dir.iterdir() if p.is_file()}
    names = sorted(set(in_map.keys()) & set(gt_map.keys()))
    if not names:
        raise RuntimeError("no paired names between input-dir and ref-dir")
    random.seed(args.seed)
    random.shuffle(names)
    names = names[: max(1, min(args.num_samples, len(names)))]

    setup_windows_dll_search_path(plugin_path)
    ctypes.CDLL(str(plugin_path))
    engine = load_trt_engine_from_plan(engine_path)

    expect_shape = (1, 3, args.img_size, args.img_size)
    x0 = load_img_rgb01(in_map[names[0]], args.img_size, device).detach().contiguous()
    torch.cuda.synchronize()
    outs0 = run_trt(engine, x0)
    if not outs0:
        raise RuntimeError("TRT returned no outputs")
    _, arr0 = pick_aligned_output(outs0, expect_shape)
    out_shape = tuple(int(v) for v in np.asarray(arr0).shape)

    times_ms: list[float] = []
    preds_cpu: list[torch.Tensor] = []
    refs_cpu: list[torch.Tensor] = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # ----- 仅 TRT：避免与 LPIPS 同时占显存 -----
    with torch.no_grad():
        for i in range(min(args.warmup, len(names))):
            xw = load_img_rgb01(in_map[names[i]], args.img_size, device).detach().contiguous()
            torch.cuda.synchronize()
            _ = run_trt(engine, xw)
            torch.cuda.synchronize()

        for n in names:
            x = load_img_rgb01(in_map[n], args.img_size, device).detach().contiguous()
            y = load_img_rgb01(gt_map[n], args.img_size, device).detach().contiguous()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            trt_outs = run_trt(engine, x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            _, arr = pick_aligned_output(trt_outs, out_shape)
            pred = torch.from_numpy(np.asarray(arr).astype(np.float32)).to(device)
            pred = torch.clamp(pred, 0.0, 1.0)

            times_ms.append((t1 - t0) * 1000.0)
            preds_cpu.append(pred.detach().cpu())
            refs_cpu.append(y.detach().cpu())
            del pred, y, x, trt_outs, arr
            torch.cuda.empty_cache()

    peak_mb_trt = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)

    del engine, outs0, arr0, x0
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # ----- 指标：此时再加载 LPIPS -----
    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    psnr_list: list[float] = []
    ssim_list: list[float] = []
    lpips_list: list[float] = []

    with torch.no_grad():
        for pred_c, ref_c in zip(preds_cpu, refs_cpu, strict=True):
            p = pred_c.to(device)
            r = ref_c.to(device)
            psnr_list.append(psnr_01(p, r))
            ssim_list.append(ssim_01(p, r))
            lpips_list.append(float(lpips_model(p, r).item()))
            del p, r

    peak_mb_total = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)

    p50 = float(np.percentile(times_ms, 50))
    p95 = float(np.percentile(times_ms, 95))

    report = {
        "backend": f"tensorrt_{args.precision}_full",
        "precision": args.precision,
        "engine": str(engine_path),
        "plugin": str(plugin_path),
        "num_samples": len(names),
        "warmup": args.warmup,
        "seed": args.seed,
        "img_size": args.img_size,
        "p50_ms": p50,
        "p95_ms": p95,
        "peak_mem_mb_trt_phase": peak_mb_trt,
        "peak_mem_mb_after_metrics": peak_mb_total,
        "peak_mem_mb_reported": max(peak_mb_trt, peak_mb_total),
        "psnr_mean": float(np.mean(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "lpips_mean": float(np.mean(lpips_list)),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
