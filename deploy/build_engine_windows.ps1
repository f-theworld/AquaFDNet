# 在 Windows 端从 ONNX + 插件构建引擎并做一次 trtexec 冒烟（封装 plugin/run_trtexec_smoke.ps1）。
# 路径可通过参数或环境变量 TENSORRT_ROOT、CUDA_PATH、CONDA_PREFIX 提供。
param(
    [string]$TensorRtRoot = "",
    [string]$CudaRoot = "",
    [string]$CondaPrefix = "",
    [string]$OnnxPath = "..\AquaFDNet_full.onnx",
    [string]$PluginPath = ".\fdconv_trt_plugin.dll",
    [string]$EnginePath = ".\AquaFDNet_full_fp32.plan",
    [string]$InputShape = "input:1x3x256x256",
    [switch]$UseShapes,
    [switch]$SkipSmoke
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

$pluginDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\plugin" | Resolve-Path
$smoke = Join-Path $pluginDir "run_trtexec_smoke.ps1"
if (-not (Test-Path -LiteralPath $smoke)) {
    throw "Missing $smoke — add plugin/run_trtexec_smoke.ps1 to the repo or skip this step."
}

$smokeArgs = @{
    TensorRtRoot = $TensorRtRoot
    CudaRoot     = $CudaRoot
    CondaPrefix  = $CondaPrefix
    OnnxPath     = $OnnxPath
    PluginPath   = $PluginPath
    EnginePath   = $EnginePath
    InputShape   = $InputShape
    SkipBuild    = $false
    WarmUp       = 2
    Iterations   = 2
    Duration     = 0
}
if ($UseShapes) { $smokeArgs.UseShapes = $true }
if ($SkipSmoke) {
    $smokeArgs.WarmUp = 0
    $smokeArgs.Iterations = 1
}

Write-Host "[build_engine_windows] invoking $smoke"
& $smoke @smokeArgs
