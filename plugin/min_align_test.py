import argparse
import ctypes
import os
from pathlib import Path

import numpy as np
import torch


def setup_windows_dll_search_path(plugin_path: Path):
    if os.name != "nt":
        return

    dll_dirs = []
    trt_root = os.environ.get("TENSORRT_ROOT", "")
    cuda_root = os.environ.get("CUDA_PATH", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    if trt_root:
        dll_dirs.extend(
            [
                Path(trt_root) / "bin",
                Path(trt_root) / "lib",
            ]
        )
    if cuda_root:
        dll_dirs.extend(
            [
                Path(cuda_root) / "bin",
                Path(cuda_root) / "libnvvp",
            ]
        )
    if conda_prefix:
        dll_dirs.append(Path(conda_prefix) / "Library" / "bin")

    dll_dirs.append(plugin_path.parent)

    valid_dirs = []
    for d in dll_dirs:
        if d.exists():
            valid_dirs.append(str(d))
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(d))

    # Keep PATH in sync for transitive dependencies that still rely on PATH lookup.
    if valid_dirs:
        os.environ["PATH"] = ";".join(valid_dirs) + ";" + os.environ.get("PATH", "")


def parse_args():
    p = argparse.ArgumentParser("Minimal TRT vs PyTorch alignment test")
    p.add_argument(
        "--onnx",
        type=str,
        default="../AquaFDNet_full.onnx",
        help="ONNX 路径；动态 batch（方案 B）请先用 export_full_onnx.py 生成 ../AquaFDNet_full_dyn.onnx 并传入",
    )
    p.add_argument("--engine", type=str, default="./AquaFDNet_full_fp32.plan")
    p.add_argument("--plugin", type=str, default="./fdconv_trt_plugin.dll")
    p.add_argument("--ckpt", type=str, default="../runs/uieb_full/best_model.pth")
    p.add_argument("--input-shape", type=int, nargs=4, default=[1, 3, 256, 256])
    p.add_argument("--batch", type=int, default=1, help="Expected batch size; must match --input-shape[0]")
    p.add_argument(
        "--trt-min-batch",
        type=int,
        default=1,
        help="构建引擎时 optimization profile 的 N 最小值（动态 ONNX）",
    )
    p.add_argument(
        "--trt-opt-batch",
        type=int,
        default=None,
        help="profile 的 N 最优值；默认与 --input-shape 的 N 相同",
    )
    p.add_argument(
        "--trt-max-batch",
        type=int,
        default=None,
        help="profile 的 N 最大值；默认与 --input-shape 的 N 相同（静态 profile）",
    )
    p.add_argument("--workspace-gb", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=123)
    # Keep default behavior aligned with current plugin MVP (FBM skipped in export path),
    # and expose an explicit switch for A/B diagnostics.
    p.add_argument("--disable-fbm-ref", action="store_true", default=True)
    p.add_argument("--enable-fbm-ref", action="store_true", default=False)
    # 逐步 dump：如 1-25 会设置 FDCONV_DUMP_RANGE / FDCONV_PT_DUMP_RANGE，且参考前向走导出分支以对齐 TRT 语义
    p.add_argument(
        "--dump-range",
        type=str,
        default="",
        help='FDConv 全局 call 索引区间，如 "1-20" 或 "5:10"；需重编插件后 FDCONV_DUMP_RANGE 生效',
    )
    return p.parse_args()


def trt_to_torch_dtype(trt_dtype):
    import tensorrt as trt
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    if trt_dtype not in mapping:
        raise RuntimeError(f"Unsupported TRT dtype: {trt_dtype}")
    return mapping[trt_dtype]


def trt_datatype_str(trt, dt) -> str:
    """将 TRT DataType 转为可读字符串（兼容不同 TensorRT Python 绑定）。"""
    if dt is None:
        return "None"
    name = getattr(dt, "name", None)
    if isinstance(name, str) and name:
        return name
    return repr(dt)


def load_trt_engine_from_plan(plan_path: Path):
    """反序列化已构建的 TensorRT 引擎（需已加载自定义插件 DLL，若图中含 plugin）。"""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    data = plan_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        raise RuntimeError(f"deserialize_cuda_engine failed: {plan_path}")
    return engine


def inspect_trt_engine(engine) -> dict:
    """
    打印/序列化引擎的 I/O dtype；若 API 可用则用 IEngineInspector 列出各层 ONELINE（含精度线索）。

    返回 dict，便于写入 JSON；不抛异常（失败时写入 *_error 字段）。
    """
    import tensorrt as trt

    report: dict = {"io_tensors": [], "layers_oneline": [], "custom_layer_hits": [], "inspector_error": None}

    try:
        if hasattr(engine, "num_io_tensors"):
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                dt = engine.get_tensor_dtype(name)
                report["io_tensors"].append(
                    {
                        "name": name,
                        "mode": str(mode),
                        "dtype": trt_datatype_str(trt, dt),
                    }
                )
        else:
            for i in range(engine.num_bindings):
                report["io_tensors"].append(
                    {
                        "index": i,
                        "name": engine.get_binding_name(i),
                        "is_input": bool(engine.binding_is_input(i)),
                        "dtype": trt_datatype_str(trt, engine.get_binding_dtype(i)),
                    }
                )
    except Exception as e:
        report["io_error"] = str(e)

    if not hasattr(engine, "create_engine_inspector"):
        report["inspector_error"] = "ICudaEngine.create_engine_inspector 不可用（TensorRT 版本过旧）"
        return report

    try:
        inspector = engine.create_engine_inspector()
    except Exception as e:
        report["inspector_error"] = f"create_engine_inspector: {e}"
        return report

    n_layers = getattr(engine, "num_layers", None)
    if n_layers is None:
        report["inspector_error"] = "engine.num_layers 不可用，无法枚举层"
        return report

    fmt = None
    for fmt_name in ("ONELINE", "JSON", "TEXT"):
        fmt = getattr(trt.LayerInformationFormat, fmt_name, None)
        if fmt is not None:
            break
    if fmt is None:
        report["inspector_error"] = "LayerInformationFormat 无可用枚举（ONELINE/JSON/TEXT）"
        return report

    try:
        for layer_idx in range(int(n_layers)):
            try:
                line = inspector.get_layer_information(layer_idx, fmt)
            except Exception as e:
                line = f"<layer {layer_idx}: {e}>"
            report["layers_oneline"].append(line)
            low = line.lower()
            if "complexfreqconv" in low or "plugin" in low or "fdconv" in low:
                report["custom_layer_hits"].append({"layer_index": layer_idx, "oneline": line})
    except Exception as e:
        report["inspector_error"] = str(e)

    return report


def load_pytorch_model(repo_root: Path, ckpt_path: Path, disable_fbm_ref: bool):
    if str(repo_root) not in os.sys.path:
        os.sys.path.insert(0, str(repo_root))

    from AquaFDNet import AquaFDNet
    from FDConv import FDConv

    device = torch.device("cuda")
    model = AquaFDNet(
        use_physics=True,
        use_fdconv=True,
        use_spatial=True,
        use_adaptive_fusion=True,
    ).to(device).eval()

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state, strict=True)

    if disable_fbm_ref:
        for m in model.modules():
            if isinstance(m, FDConv) and hasattr(m, "FBM"):
                m.FBM = torch.nn.Identity()

    return model


def build_or_load_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float,
    input_shape,
    trt_min_batch: int,
    trt_opt_batch: int,
    trt_max_batch: int,
    *,
    fp16: bool = False,
):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)

    if engine_path.exists():
        engine_bytes = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Failed to deserialize existing engine.")
        return engine

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )

    _, c, h, w = (int(v) for v in input_shape)
    min_shape = (int(trt_min_batch), c, h, w)
    opt_shape = (int(trt_opt_batch), c, h, w)
    max_shape = (int(trt_max_batch), c, h, w)
    if not (min_shape[0] <= opt_shape[0] <= max_shape[0]):
        raise ValueError(
            f"TRT profile batches invalid: min={min_shape[0]} opt={opt_shape[0]} max={max_shape[0]}"
        )
    profile = builder.create_optimization_profile()
    profile.set_shape("input", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network failed.")

    engine_path.write_bytes(bytes(serialized))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Failed to deserialize built engine.")
    return engine


def run_trt(engine, x_cuda: torch.Tensor):
    import tensorrt as trt

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context.")

    stream = torch.cuda.Stream()

    # New API (TRT 10+): name-based I/O
    if hasattr(engine, "num_io_tensors"):
        io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(input_names) != 1:
            raise RuntimeError(f"Expected single input, got: {input_names}")

        in_name = input_names[0]
        context.set_input_shape(in_name, tuple(x_cuda.shape))
        context.set_tensor_address(in_name, int(x_cuda.data_ptr()))

        outputs = {}
        for n in output_names:
            shape = tuple(context.get_tensor_shape(n))
            dtype = trt_to_torch_dtype(engine.get_tensor_dtype(n))
            out = torch.empty(shape, dtype=dtype, device="cuda")
            context.set_tensor_address(n, int(out.data_ptr()))
            outputs[n] = out

        with torch.cuda.stream(stream):
            ok = context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        stream.synchronize()
        return {k: v.detach().cpu().numpy() for k, v in outputs.items()}

    # Old API (TRT 8/9): binding-based I/O
    n_bindings = engine.num_bindings
    bindings = [0] * n_bindings
    input_indices = [i for i in range(n_bindings) if engine.binding_is_input(i)]
    output_indices = [i for i in range(n_bindings) if not engine.binding_is_input(i)]
    if len(input_indices) != 1:
        raise RuntimeError(f"Expected single input binding, got {input_indices}")

    in_idx = input_indices[0]
    context.set_binding_shape(in_idx, tuple(x_cuda.shape))
    bindings[in_idx] = int(x_cuda.data_ptr())

    outputs = {}
    for i in output_indices:
        shape = tuple(context.get_binding_shape(i))
        dtype = trt_to_torch_dtype(engine.get_binding_dtype(i))
        out = torch.empty(shape, dtype=dtype, device="cuda")
        bindings[i] = int(out.data_ptr())
        outputs[engine.get_binding_name(i)] = out

    with torch.cuda.stream(stream):
        ok = context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v2 failed.")
    stream.synchronize()
    return {k: v.detach().cpu().numpy() for k, v in outputs.items()}


def pick_aligned_output(trt_outs, ref_shape):
    # Prefer semantic name when available.
    for k, v in trt_outs.items():
        if "enhanced" in k.lower():
            return k, v

    # Fallback: choose output that matches reference shape.
    candidates = [(k, v) for k, v in trt_outs.items() if tuple(v.shape) == tuple(ref_shape)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Deterministic fallback when multiple tensors share same shape.
        return candidates[0]

    shape_info = {k: tuple(v.shape) for k, v in trt_outs.items()}
    raise RuntimeError(
        f"No TRT output matches REF shape {tuple(ref_shape)}. "
        f"Available TRT outputs: {shape_info}"
    )


def main():
    args = parse_args()
    if args.batch < 1:
        raise ValueError(f"--batch must be >=1, got {args.batch}")
    if len(args.input_shape) != 4:
        raise ValueError(f"--input-shape expects 4 dims, got {args.input_shape}")
    if int(args.input_shape[0]) != int(args.batch):
        raise ValueError(
            f"--batch ({args.batch}) must match --input-shape[0] ({args.input_shape[0]}). "
            f"Example: --batch {args.batch} --input-shape {args.batch} 3 256 256"
        )

    run_b = int(args.input_shape[0])
    opt_b = int(args.trt_opt_batch) if args.trt_opt_batch is not None else run_b
    max_b = int(args.trt_max_batch) if args.trt_max_batch is not None else run_b
    min_b = int(args.trt_min_batch)
    if min_b > run_b or max_b < run_b:
        raise ValueError(
            f"Run batch N={run_b} must satisfy trt_min_batch <= N <= trt_max_batch "
            f"({min_b}..{max_b}); adjust --trt-min-batch/--trt-max-batch or --input-shape."
        )
    if not (min_b <= opt_b <= max_b):
        raise ValueError(
            f"--trt-min/opt/max-batch must satisfy min<=opt<=max, got {min_b},{opt_b},{max_b}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    script_dir = Path(__file__).resolve().parent
    onnx_path = (script_dir / args.onnx).resolve()
    engine_path = (script_dir / args.engine).resolve()
    plugin_path = (script_dir / args.plugin).resolve()
    ckpt_path = (script_dir / args.ckpt).resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not plugin_path.exists():
        raise FileNotFoundError(f"Plugin dll not found: {plugin_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    setup_windows_dll_search_path(plugin_path)

    # Load plugin library before parsing/running TRT.
    ctypes.CDLL(str(plugin_path))

    # Fixed input
    b, c, h, w = args.input_shape
    x_cuda = torch.randn(b, c, h, w, device="cuda", dtype=torch.float32)

    dr = (args.dump_range or "").strip()
    if dr:
        for sep in ("-", ":"):
            if sep in dr:
                a, b = dr.split(sep, 1)
                lo, hi = int(a.strip()), int(b.strip())
                if lo < 1 or hi < lo:
                    raise ValueError(f"Invalid --dump-range {dr!r}, expect lo-hi with lo>=1 and hi>=lo")
                os.environ["FDCONV_DUMP"] = "1"
                os.environ["FDCONV_PT_DUMP"] = "1"
                os.environ["FDCONV_DUMP_RANGE"] = f"{lo}-{hi}"
                os.environ["FDCONV_PT_DUMP_RANGE"] = f"{lo}-{hi}"
                break
        else:
            raise ValueError(f"Invalid --dump-range {dr!r}, use e.g. 1-20")

    # PyTorch reference（可选：与 ONNX 导出一致的 ComplexFreqConvBlockFn，便于与 TRT 同 call 对齐 dump）
    disable_fbm_ref = args.disable_fbm_ref and (not args.enable_fbm_ref)
    model = load_pytorch_model(script_dir.parent, ckpt_path, disable_fbm_ref=disable_fbm_ref)
    if dr:
        from FDConv import ComplexFreqConvBlockFn, FDConv

        ComplexFreqConvBlockFn.reset_dump_counters()
        for m in model.modules():
            if isinstance(m, FDConv):
                m.export_custom_freqconv = True
        _orig_onnx_export = torch.onnx.is_in_onnx_export
        torch.onnx.is_in_onnx_export = lambda: True
        try:
            with torch.no_grad():
                ref_out = model(x_cuda)[0].detach().cpu().numpy()  # enhanced
        finally:
            torch.onnx.is_in_onnx_export = _orig_onnx_export
    else:
        with torch.no_grad():
            ref_out = model(x_cuda)[0].detach().cpu().numpy()  # enhanced

    # TRT run
    engine = build_or_load_engine(
        onnx_path,
        engine_path,
        args.workspace_gb,
        args.input_shape,
        trt_min_batch=min_b,
        trt_opt_batch=opt_b,
        trt_max_batch=max_b,
    )
    trt_outs = run_trt(engine, x_cuda)
    if len(trt_outs) == 0:
        raise RuntimeError("No TRT outputs.")
    selected_name, trt_first = pick_aligned_output(trt_outs, ref_out.shape)

    # Align shape if needed
    if trt_first.shape != ref_out.shape:
        raise RuntimeError(f"Shape mismatch: TRT {trt_first.shape} vs REF {ref_out.shape}")

    abs_diff = np.abs(trt_first.astype(np.float32) - ref_out.astype(np.float32))
    print(f"run_ok=1")
    print(f"max_abs_err={abs_diff.max():.8f}")
    print(f"mean_abs_err={abs_diff.mean():.8f}")
    print(f"selected_output={selected_name}")
    print(f"trt_outputs={list(trt_outs.keys())}")


if __name__ == "__main__":
    main()

