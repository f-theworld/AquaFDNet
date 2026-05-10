# 将 ONNX、引擎、插件及环境指纹打包到 deploy/dist（便于拷贝到端侧）。
param(
    [string]$RepoRoot = "",
    [string]$DistDir = "",
    [string]$OnnxName = "AquaFDNet_full.onnx",
    [string]$EngineName = "AquaFDNet_full_fp32.plan",
    [string]$PluginName = "fdconv_trt_plugin.dll",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")).Path
}
if (-not $DistDir) {
    $DistDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "dist"
}

$pluginDir = Join-Path $RepoRoot "plugin"
$srcOnnx = Join-Path $RepoRoot $OnnxName
$srcEngine = Join-Path $pluginDir $EngineName
$srcPlugin = Join-Path $pluginDir $PluginName

foreach ($p in @($srcOnnx, $srcEngine, $srcPlugin)) {
    if (-not (Test-Path -LiteralPath $p)) {
        throw "Missing artifact: $p"
    }
}

New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

Copy-Item -LiteralPath $srcOnnx -Destination (Join-Path $DistDir $OnnxName) -Force
Copy-Item -LiteralPath $srcEngine -Destination (Join-Path $DistDir $EngineName) -Force
Copy-Item -LiteralPath $srcPlugin -Destination (Join-Path $DistDir $PluginName) -Force

$envScript = Join-Path $DistDir "set_runtime_env.ps1"
$envBody = @"
# 由 package_release.ps1 生成；在 dist 目录下执行: . .\set_runtime_env.ps1
`$here = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$env:AQUAFDNET_DIST = `$here
Write-Host "[env] AQUAFDNET_DIST=`$here"
"@
Set-Content -Path $envScript -Value $envBody -Encoding utf8

$fp = Join-Path $DistDir "env_fingerprint.json"
$collect = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "collect_env.ps1"
& $collect -OutJson $fp -PythonExe $PythonExe

$manifest = [ordered]@{
    packaged_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    repo_root       = $RepoRoot
    artifacts       = @{
        onnx   = $OnnxName
        engine = $EngineName
        plugin = $PluginName
    }
}
$manifestPath = Join-Path $DistDir "manifest.json"
($manifest | ConvertTo-Json -Depth 6) | Set-Content -Path $manifestPath -Encoding utf8

if (Test-Path -LiteralPath $fp) {
    $finger = Get-Content -Raw -Path $fp | ConvertFrom-Json
    $manifest2 = [ordered]@{
        packaged_at_utc = $manifest.packaged_at_utc
        repo_root       = $manifest.repo_root
        artifacts       = $manifest.artifacts
        env_fingerprint = $finger
    }
    ($manifest2 | ConvertTo-Json -Depth 8) | Set-Content -Path $manifestPath -Encoding utf8
}

Write-Host "[package_release] OK -> $DistDir"
Write-Host "  manifest: $manifestPath"
