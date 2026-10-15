param(
    [string]$SourceTle = "",
    [switch]$NoDownload,
    [switch]$AllowCurrentEpochs,
    [int]$Trials = 20,
    [double]$Duration = 60.0,
    [int]$TargetSamples = 1200,
    [int]$TargetConcurrentSamples = 300,
    [string]$PythonExe = "D:\Anaconda\python.exe",
    [string]$MatlabExe = "D:\MATLAB\R2024a\bin\matlab.exe",
    [switch]$SkipNs3
)

$ErrorActionPreference = "Stop"

function ConvertTo-MatlabLiteral {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defaultSource = Join-Path $repoRoot "data\raw\starlink.tle"
if ([string]::IsNullOrWhiteSpace($SourceTle) -and (Test-Path $defaultSource)) {
    $SourceTle = $defaultSource
    $NoDownload = $true
}
if (-not [string]::IsNullOrWhiteSpace($SourceTle) -and -not [IO.Path]::IsPathRooted($SourceTle)) {
    $SourceTle = Join-Path $repoRoot $SourceTle
}

if (-not (Test-Path $MatlabExe)) {
    throw "MATLAB executable not found: $MatlabExe"
}
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$matlabArgs = @(
    "addpath(" + (ConvertTo-MatlabLiteral (Join-Path $repoRoot "matlab")) + ")"
)
$callArgs = @()
if (-not [string]::IsNullOrWhiteSpace($SourceTle)) {
    $callArgs += "'SourceTle'"
    $callArgs += ConvertTo-MatlabLiteral $SourceTle
}
$callArgs += "'PythonExe'"
$callArgs += ConvertTo-MatlabLiteral $PythonExe
$callArgs += "'Trials'"
$callArgs += $Trials.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$callArgs += "'Duration'"
$callArgs += $Duration.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
$callArgs += "'TargetSamples'"
$callArgs += $TargetSamples.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$callArgs += "'TargetConcurrentSamples'"
$callArgs += $TargetConcurrentSamples.ToString([System.Globalization.CultureInfo]::InvariantCulture)
if ($NoDownload) {
    $callArgs += "'NoDownload'"
    $callArgs += "true"
}
if ($AllowCurrentEpochs) {
    $callArgs += "'AllowCurrentEpochs'"
    $callArgs += "true"
}
$matlabArgs += "run_paper_simulation(" + ($callArgs -join ", ") + ")"
$batch = ($matlabArgs -join "; ") + ";"

Write-Host "MATLAB batch command:"
Write-Host $batch
& $MatlabExe -batch $batch
if ($LASTEXITCODE -ne 0) {
    throw "MATLAB simulation workflow failed."
}

if (-not $SkipNs3) {
    & (Join-Path $PSScriptRoot "run_ns3_3gpp_wsl.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "NS-3 communication simulation failed."
    }
}

$manifest = [ordered]@{
    repo_root = $repoRoot
    source_tle = $SourceTle
    python_exe = $PythonExe
    matlab_exe = $MatlabExe
    trials = $Trials
    duration_s = $Duration
    target_samples = $TargetSamples
    target_concurrent_samples = $TargetConcurrentSamples
    no_download = [bool]$NoDownload
    allow_current_epochs = [bool]$AllowCurrentEpochs
    skip_ns3 = [bool]$SkipNs3
    reports_dir = (Join-Path $repoRoot "reports")
}
$manifestPath = Join-Path $repoRoot "reports\full_run_manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Full simulation completed."
Write-Host "Reports directory: $($manifest.reports_dir)"
