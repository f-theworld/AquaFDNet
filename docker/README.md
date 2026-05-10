# Linux + GPU + TensorRT（Docker）

本目录提供在 **Linux 容器** 内编译 **FDConv TensorRT 插件（`.so`）** 并运行 **TensorRT / trtexec** 的基线镜像；与 Windows 下的 `fdconv_trt_plugin.dll` **二进制不通用**，需在 Linux 重编。

当前 Dockerfile 使用**两阶段**构建：

- `builder`：`nvidia/cuda:*devel`（含 `nvcc`），负责编译插件；
- `runtime`：`nvcr.io/nvidia/tensorrt:*py3`，负责推理运行。

这样即使服务器宿主机没有 `nvcc`，也能在镜像构建阶段编出 `.so`。

## 前置条件

- Linux 主机（或 WSL2 后端）已安装 **Docker**、**NVIDIA 驱动**、**[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**。
- 能拉取 NGC 镜像（部分环境需 `docker login nvcr.io`）。

## 默认基础镜像（可覆盖）

`docker/Dockerfile` 默认使用两类基础镜像：

- `BUILD_IMAGE=nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04`
- `RUNTIME_IMAGE=nvcr.io/nvidia/tensorrt:24.02-py3`

可通过构建参数覆盖（需与目标机驱动及你打算使用的 TensorRT/CUDA 大版本一致）：

```bash
docker build -f docker/Dockerfile \
  --build-arg BUILD_IMAGE=nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04 \
  --build-arg RUNTIME_IMAGE=nvcr.io/nvidia/tensorrt:24.02-py3 \
  -t aquafdnet-trt:dev .
```

## 构建镜像（推荐从仓库根执行）

```bash
docker compose -f docker/compose.yaml build
```

## 进入容器（挂载当前仓库）

```bash
docker compose -f docker/compose.yaml run --rm trt-dev bash
```

容器内插件路径：`**/opt/fdconv/libfdconv_trt_plugin.so**`（已加入 `LD_LIBRARY_PATH`）。

如果你想确认最终运行容器里不需要 `nvcc`（正常现象）：

```bash
docker compose -f docker/compose.yaml run --rm trt-dev bash -lc \
  'which nvcc || echo "nvcc not in runtime image (expected)"; ls -l /opt/fdconv/libfdconv_trt_plugin.so'
```

## 引擎（`.plan`）与 ONNX

- **在 Windows 上构建的 `.plan` 不建议直接拷进 Linux 使用**（CUDA/TRT 版本、驱动、架构绑定不同）。推荐在容器内用 **trtexec** 从 `AquaFDNet_full.onnx` 重建。
- 示例（在挂载的仓库根 `/workspace` 下执行，按需要改 workspace 大小与输出路径）：

```bash
trtexec --onnx=AquaFDNet_full.onnx \
  --saveEngine=plugin/engines_matrix/AquaFDNet_full_fp32_linux.plan \
  --staticPlugins=/opt/fdconv/libfdconv_trt_plugin.so \
  --memPoolSize=workspace:4096
```

## Python 脚本（`plugin/min_align_test.py` 等）

镜像默认 **未安装 PyTorch**（避免与 NGC CUDA 小版本错配）。若要在容器内跑 PT vs TRT 对齐，请先安装与镜像 CUDA 主版本匹配的 wheel，例如（版本号请按容器内 `nvcc --version` / NGC Release Notes 调整）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

运行 TRT 时 `**--plugin` 需指向 `.so**`，例如：

```bash
cd /workspace/plugin
python min_align_test.py --plugin /opt/fdconv/libfdconv_trt_plugin.so \
  --engine ./engines_matrix/AquaFDNet_full_fp32_linux.plan
```

## 故障排查

- **构建时拉不到 `nvcr.io/nvidia/tensorrt:*`**：先 `docker login nvcr.io`（用户名 `$oauthtoken`，密码为 NGC API Key）。
- **构建时 `NvInfer*.h` 拷贝失败**：说明 `RUNTIME_IMAGE` 的 TensorRT 头文件路径变化，执行 `docker run --rm <runtime_image> bash -lc 'ls /usr/include/x86_64-linux-gnu | rg NvInfer'` 后按实际路径调整 Dockerfile 的 `COPY --from=...`。
- **插件 `nvcc` 链接失败**：检查 `libnvinfer` / `libcudnn` 是否在 `TRT_LIB`、`CUDA_HOME/lib64` 下；必要时在 devel 镜像内安装缺失包。
- **反序列化 `.plan` 失败**：在**当前镜像**内用同一 ONNX **重新 build** 引擎。

