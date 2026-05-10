"""
导出 AquaFDNet full ONNX。

默认导出 **动态 batch（N 维）** 的 ONNX（方案 B），便于 TensorRT 使用
min/opt/max profile 构建引擎。

静态 batch=1 的旧行为：加 --static。
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from AquaFDNet import AquaFDNet
from FDConv import FDConv


def parse_args():
    p = argparse.ArgumentParser(description="Export AquaFDNet full ONNX")
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "uieb_full" / "best_model.pth"),
        help="权重路径",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="输出 ONNX 路径；默认：动态 -> AquaFDNet_full_dyn.onnx，静态 -> AquaFDNet_full.onnx",
    )
    p.add_argument(
        "--static",
        action="store_true",
        help="导出固定 N=1（无 dynamic_axes），兼容旧 TRT 静态引擎流程",
    )
    p.add_argument(
        "--dummy-batch",
        type=int,
        default=1,
        help="torch.onnx.export 使用的示例 batch（仅影响 trace，不改变动态维）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.dummy_batch < 1:
        raise ValueError("--dummy-batch must be >= 1")

    repo = Path(__file__).resolve().parent
    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    out = args.output.strip()
    if not out:
        out = "AquaFDNet_full.onnx" if args.static else "AquaFDNet_full_dyn.onnx"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AquaFDNet(
            use_physics=True,
            use_fdconv=True,
            use_spatial=True,
            use_adaptive_fusion=True,
        )
        .to(device)
        .eval()
    )
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state, strict=True)

    def enable_custom_export(m):
        if isinstance(m, FDConv):
            m.export_custom_freqconv = True

    model.apply(enable_custom_export)

    dummy = torch.randn(args.dummy_batch, 3, 256, 256, device=device)
    export_kw = dict(
        model=model,
        args=(dummy,),
        f=str(out_path),
        opset_version=17,
        input_names=["input"],
        output_names=["enhanced", "reconstructed", "A", "t"],
        custom_opsets={"custom.fdconv": 1},
        do_constant_folding=True,
    )
    if not args.static:
        export_kw["dynamic_axes"] = {
            "input": {0: "N"},
            "enhanced": {0: "N"},
            "reconstructed": {0: "N"},
            "A": {0: "N"},
            "t": {0: "N"},
        }

    with torch.no_grad():
        torch.onnx.export(**export_kw)

    print(f"ok wrote {out_path} static={bool(args.static)}")


if __name__ == "__main__":
    main()
