# 验证打包产物：环境采集 + 最小 TRT 推理；可选与 PyTorch 对齐（需 ckpt 与完整 CUDA 环境）。
param(
    [string]$RepoRoot = "",
    [string]$DistDir = "",
    [string]$PythonExe = "",
    [switch]$WithAlign,
    [switch]$Benchmark,
    [string]$CkptRelative = "runs\uieb_full\best_model.pth"
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")).Path
}
if (-not $DistDir) {
    $DistDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "dist"
}

$dist = Resolve-Path $DistDir
$onnx = Join-Path $dist "AquaFDNet_full.onnx"
$eng = Join-Path $dist "AquaFDNet_full_fp32.plan"
$plg = Join-Path $dist "fdconv_trt_plugin.dll"

foreach ($p in @($onnx, $eng, $plg)) {
    if (-not (Test-Path -LiteralPath $p)) {
        throw "verify_deploy: missing $p (run package_release.ps1 first)"
    }
}

$collect = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "collect_env.ps1"
Write-Host "=== collect_env ==="
& $collect

$py = $PythonExe
if (-not $py) {
    if ($env:CONDA_PREFIX -and (Test-Path -LiteralPath (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        $py = Join-Path $env:CONDA_PREFIX "python.exe"
    } else { $py = "python" }
}

$infer = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "run_infer_minimal.py"
Write-Host "=== run_infer_minimal ==="
& $py $infer --repo-root $RepoRoot --engine $eng --plugin $plg --input-shape 1 3 256 256 --seed 123 --sync

if ($Benchmark) {
    $bm = Join-Path $RepoRoot "plugin\benchmark_modes.ps1"
    if (-not (Test-Path -LiteralPath $bm)) { throw "benchmark_modes.ps1 not found under plugin/" }
    Write-Host "=== benchmark_modes (short) ==="
    Push-Location (Join-Path $RepoRoot "plugin")
    try {
        & $bm -Repeats 2 -WarmUp 5 -Iterations 10
    } finally {
        Pop-Location
    }
}

if ($WithAlign) {
    $ckpt = Join-Path $RepoRoot $CkptRelative
    if (-not (Test-Path -LiteralPath $ckpt)) {
        throw "WithAlign requires ckpt: $ckpt"
    }
    $pluginDir = Join-Path $RepoRoot "plugin"
    Write-Host "=== min_align_test (TRT vs PyTorch) ==="
    Push-Location $pluginDir
    try {
        & $py .\min_align_test.py --onnx $onnx --engine $eng --plugin $plg --ckpt $ckpt --seed 123
    } finally {
        Pop-Location
    }
}

Write-Host "[verify_deploy] OK"
