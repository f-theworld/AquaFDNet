# 将 TensorRT / CUDA /（可选）Conda 库加入 PATH。与 plugin/run_trtexec_smoke.ps1 思路一致。
# 传入路径或通过环境变量：TENSORRT_ROOT、CUDA_PATH、CONDA_PREFIX。
# 用法: . .\set_runtime_env.ps1 [-TensorRtRoot ...] [-CudaRoot ...] [-CondaPrefix ...]
param(
    [string]$TensorRtRoot = "",
    [string]$CudaRoot = "",
    [string]$CondaPrefix = "",
    [string]$PluginDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $TensorRtRoot) { $TensorRtRoot = $env:TENSORRT_ROOT }
if (-not $CudaRoot) { $CudaRoot = $env:CUDA_PATH }
if (-not $CondaPrefix) { $CondaPrefix = $env:CONDA_PREFIX }

if (-not $TensorRtRoot -or -not (Test-Path -LiteralPath $TensorRtRoot)) {
    throw "TensorRT root missing: pass -TensorRtRoot or set `$env:TENSORRT_ROOT."
}
if (-not $CudaRoot -or -not (Test-Path -LiteralPath $CudaRoot)) {
    throw "CUDA root missing: pass -CudaRoot or set `$env:CUDA_PATH."
}

if (-not $PluginDir) {
    $PluginDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\plugin" | Resolve-Path
}

$parts = @(
    (Join-Path $TensorRtRoot "bin"),
    (Join-Path $TensorRtRoot "lib"),
    (Join-Path $CudaRoot "bin"),
    (Join-Path $CudaRoot "libnvvp")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

if ($CondaPrefix -and (Test-Path -LiteralPath (Join-Path $CondaPrefix "Library\bin"))) {
    $parts += (Join-Path $CondaPrefix "Library\bin")
}
if ($PluginDir -and (Test-Path -LiteralPath $PluginDir.Path)) {
    $parts += $PluginDir.Path
}

$env:Path = (($parts -join ";") + ";" + $env:Path)
$env:TENSORRT_ROOT = $TensorRtRoot
$env:CUDA_PATH = $CudaRoot
if ($CondaPrefix) { $env:CONDA_PREFIX = $CondaPrefix }

Write-Host "[set_runtime_env] TENSORRT_ROOT=$TensorRtRoot"
Write-Host "[set_runtime_env] CUDA_PATH=$CudaRoot"
Write-Host "[set_runtime_env] PluginDir=$($PluginDir.Path)"
