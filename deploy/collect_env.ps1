# 采集端侧 / 本机部署指纹：OS、架构、GPU、CUDA、TensorRT（供 manifest 与跨机对齐）。
# 用法: .\collect_env.ps1 [-OutJson path\to\env_fingerprint.json]
param(
    [string]$OutJson = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

function Try-Command($Name, [object[]]$CmdArgs) {
    try {
        $o = & $Name @CmdArgs 2>$null
        return ($o | Out-String).Trim()
    } catch {
        return ""
    }
}

$isWin = $PSVersionTable.Platform -eq "Win32NT" -or $env:OS -match "Windows"
$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
$rid = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription

$gpu = ""
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (Try-Command "nvidia-smi" @("--query-gpu=name,driver_version", "--format=csv,noheader"))
}

$cudaNvcc = ""
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $cudaNvcc = (Try-Command "nvcc" @("--version"))
}

$trtVer = ""
$py = $PythonExe
if (-not $py) {
    $condaPy = ""
    if ($env:CONDA_PREFIX) {
        $cp = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $cp) { $condaPy = $cp }
    }
    $cands = @($condaPy, "python") | Where-Object { $_ }
    foreach ($c in $cands) {
        if ($c -eq "python" -and (Get-Command python -ErrorAction SilentlyContinue)) { $py = "python"; break }
        if ($c -ne "python" -and (Test-Path $c)) { $py = $c; break }
    }
}
if ($py) {
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        $trtVer = (& $py -c "import tensorrt as trt; print(trt.__version__)" 2>&1 | Out-String).Trim()
    } catch {
        $trtVer = "tensorrt_query_failed"
    }
    $ErrorActionPreference = $oldEap
}

$obj = [ordered]@{
    collected_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    os_description     = $rid
    process_architecture = $arch
    is_windows         = $isWin
    nvidia_smi_gpu     = $gpu
    nvcc_version       = $cudaNvcc
    tensorrt_python    = $trtVer
    tensorrt_root      = $env:TENSORRT_ROOT
    cuda_path          = $env:CUDA_PATH
    conda_prefix       = $env:CONDA_PREFIX
}

$json = ($obj | ConvertTo-Json -Depth 4)
if (-not $OutJson) {
    Write-Output $json
}

if ($OutJson) {
    $dir = Split-Path -Parent $OutJson
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $json | Set-Content -Path $OutJson -Encoding utf8
    Write-Host "[collect_env] wrote $OutJson"
}
