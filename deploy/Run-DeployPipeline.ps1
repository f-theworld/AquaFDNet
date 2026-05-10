# 一键：采集环境 ->（可选）构建插件 ->（可选）构建引擎 -> 打包 dist -> 验证最小推理。
# TensorRT/CUDA 路径：参数或环境变量 TENSORRT_ROOT、CUDA_PATH、CONDA_PREFIX。
param(
    [string]$RepoRoot = "",
    [switch]$SkipBuildPlugin,
    [switch]$SkipBuildEngine,
    [switch]$WithAlign,
    [string]$TensorRtRoot = "",
    [string]$CudaRoot = "",
    [string]$CondaPrefix = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $here "..")).Path
}

if (-not $TensorRtRoot) { $TensorRtRoot = $env:TENSORRT_ROOT }
if (-not $CudaRoot) { $CudaRoot = $env:CUDA_PATH }
if (-not $CondaPrefix) { $CondaPrefix = $env:CONDA_PREFIX }

if (-not $SkipBuildPlugin -or -not $SkipBuildEngine) {
    if (-not $TensorRtRoot -or -not (Test-Path -LiteralPath $TensorRtRoot)) {
        throw "TensorRT root required for build steps: pass -TensorRtRoot or set `$env:TENSORRT_ROOT (or use -SkipBuildPlugin -SkipBuildEngine)."
    }
    if (-not $CudaRoot -or -not (Test-Path -LiteralPath $CudaRoot)) {
        throw "CUDA root required for build steps: pass -CudaRoot or set `$env:CUDA_PATH (or use -SkipBuildPlugin -SkipBuildEngine)."
    }
}

$py = $PythonExe
if (-not $py -and $CondaPrefix -and (Test-Path -LiteralPath (Join-Path $CondaPrefix "python.exe"))) {
    $py = Join-Path $CondaPrefix "python.exe"
}
if (-not $py) { $py = "python" }

Write-Host "=== [1/5] collect_env ==="
& (Join-Path $here "collect_env.ps1")

if (-not $SkipBuildPlugin) {
    $bp = Join-Path $RepoRoot "plugin\build_plugin.ps1"
    if (-not (Test-Path -LiteralPath $bp)) {
        throw "Missing $bp — add plugin/build_plugin.ps1 or use -SkipBuildPlugin."
    }
    Write-Host "=== [2/5] build_plugin (Windows) ==="
    & $bp -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot
} else {
    Write-Host "=== [2/5] build_plugin SKIPPED ==="
}

if (-not $SkipBuildEngine) {
    Write-Host "=== [3/5] build_engine ==="
    Push-Location (Join-Path $RepoRoot "plugin")
    try {
        & (Join-Path $here "build_engine_windows.ps1") -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot -CondaPrefix $CondaPrefix
    } finally {
        Pop-Location
    }
} else {
    Write-Host "=== [3/5] build_engine SKIPPED ==="
}

Write-Host "=== [4/5] package_release ==="
& (Join-Path $here "package_release.ps1") -RepoRoot $RepoRoot -PythonExe $py

Write-Host "=== [5/5] verify_deploy ==="
$vargs = @{ RepoRoot = $RepoRoot; PythonExe = $py }
if ($WithAlign) { $vargs.WithAlign = $true }
& (Join-Path $here "verify_deploy.ps1") @vargs

Write-Host "[Run-DeployPipeline] done"
