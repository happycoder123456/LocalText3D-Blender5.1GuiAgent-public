$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv yet. Installing Shap-E for Windows 11 + Python 3.13..."
    & (Join-Path $PSScriptRoot "run_all.ps1") @args
    exit $LASTEXITCODE
}

Write-Host "Starting local worker on http://127.0.0.1:8765"
Write-Host "This address is this computer only. Do not port-forward it."
Write-Host "Leave this window open. In Blender, set Engine to Shap-E."
& $py -m worker serve @args
