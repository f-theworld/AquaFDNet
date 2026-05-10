# AquaFDNet

Repository for an underwater image enhancement–oriented network and its deployment stack. The **FD** in the name relates to **FDConv** (frequency-domain convolution), **TensorRT** custom plugins, and ONNX/engine workflows.

## Availability & reproducibility

The **companion paper is not yet formally published**. To respect the review and publication process, **the full training code, the top-level `AquaFDNet` Python package expected by `docker/Dockerfile` (see the `import AquaFDNet` build check), and official checkpoints are not published in this tree yet.**

**After the paper appears, we will release the remaining model-side artifacts**—architecture and training code, pretrained weights, export defaults, and configuration—so that the Docker Quick Start, ONNX export, and `plugin/` evaluation flows can be executed end-to-end without relying on private copies.

Until that release, this repository provides **ONNX/TensorRT Python helpers under `plugin/`, container definitions under `docker/`, and CUDA reference snippets under `CUDA/`** for early inspection. **The TensorRT custom-operator C++/CUDA sources are not shipped here during the pre-publication period** (see [Custom operator open source](#custom-operator-tensorrt-plugin-open-source) below).

> **Note:** `docker compose … build` runs `PYTHONPATH=/workspace python3 -c "from AquaFDNet import AquaFDNet"` and compiles the FDConv TensorRT plugin from sources under `plugin/`. **Both the model package and the plugin sources are required for a full image build** unless you adapt the Dockerfile to inject a prebuilt `.so`.

## Custom operator (TensorRT plugin) open source

The deployment stack relies on a **TensorRT plugin** implementing the frequency-domain custom op used in ONNX (`ComplexFreqConvBlock` / FDConv path). **The C++/CUDA implementation of that operator is withheld from this public tree until publication**, so that implementation-level details stay aligned with the paper’s disclosure timeline.

**We will open-source the operator**—the plugin sources (`complex_freq_conv_plugin.cpp`, `complex_freq_conv_plugin.h`, `complex_freq_conv_kernels.cu`, and any small build glue) and the same layout expected by `docker/build_plugin.sh`—**when the companion work is formally published**, in the same repository (or a tagged release) and under the project’s chosen license once a `LICENSE` file is added. That drop, together with the model-side release above, restores **source-level** reproducibility for `nvcc` → `libfdconv_trt_plugin.so` and end-to-end TRT alignment.

Until then, the scripts in `plugin/` document export and evaluation **interfaces**; they do not bundle the operator binary or its CUDA/C++ sources.

## Repository layout (current checkout)


| Path                     | Description                                                                                                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docker/`                | Linux + GPU + TensorRT two-stage image recipes; see [docker/README.md](docker/README.md). **Image build expects plugin sources under `plugin/`** once the open-source operator drop is available.       |
| `docker/Dockerfile`      | Override `BUILD_IMAGE` / `RUNTIME_IMAGE` as needed; runtime stage installs `/opt/fdconv/libfdconv_trt_plugin.so` when the builder stage succeeds.                                                       |
| `docker/compose.yaml`    | Sample compose: mount repo to `/workspace`, GPU reservation, `torch-cache` volume, optional dataset mounts.                                                                                             |
| `docker/build_plugin.sh` | Compiles the FDConv TensorRT plugin to `libfdconv_trt_plugin.so` from `plugin/*.cpp` / `plugin/*.cu` **when those sources are present**.                                                                |
| `plugin/`                | Python tooling only: `export_full_onnx.py`, `min_align_test.py`, `eval_trt_uieb_accuracy_benchmark.py`. **Operator C++/CUDA sources are not in this checkout** (pending the open-source release above). |
| `CUDA/`                  | C++/CUDA sources in the style of PyTorch `ATen` CUDA spectral ops (cuFFT, Hermitian symmetry fill, etc.). Reference material for FFT alignment—not a standalone executable.                             |


**Planned after publication (full train → export → TRT from source):**

- Top-level `AquaFDNet` package and related modules consumed by `export_full_onnx.py`
- Official checkpoints and dataset/eval defaults referenced by the scripts
- **TensorRT FDConv / `ComplexFreqConvBlock` plugin C++/CUDA sources** under `plugin/`, so `docker/build_plugin.sh` matches public trees again

## Quick start: Docker + TensorRT

Prerequisites: Linux or WSL2, Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), and access to NGC images where applicable.

**Prerequisites for a successful `docker compose … build`:** (1) post-publication **model code** at the repo root for the `import AquaFDNet` check, and (2) post-publication **plugin C++/CUDA sources** under `plugin/` for the `nvcc` build step—or your own Dockerfile changes to skip build and copy in a compatible `libfdconv_trt_plugin.so`.

From the **repository root** (once the above artifacts exist in your tree):

```bash
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm trt-dev bash
```

In-container plugin path: `/opt/fdconv/libfdconv_trt_plugin.so` (on `LD_LIBRARY_PATH`).

For build args, **trtexec** `.plan` generation from ONNX, Python alignment tests, and troubleshooting, see **[docker/README.md](docker/README.md)**.

## CUDA directory

`CUDA/SpectralOps.cpp` and `CUDA/SpectralOps.cu` follow PyTorch’s native CUDA spectral implementation patterns (cuFFT plan cache, conjugate symmetry kernels). Compare against the matching `pytorch/aten` sources for your PyTorch version if you patch or rebuild.

## License and citation

If the repository adds a `LICENSE` or paper BibTeX, prefer those. Otherwise follow upstream dependency licenses and in-repo notices.

---

Operational detail lives in **docker/README.md**; this file is the high-level map.