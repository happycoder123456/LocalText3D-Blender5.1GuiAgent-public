$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Test-Python {
    param([string[]]$Parts)
    $exe = Get-Command $Parts[0] -ErrorAction SilentlyContinue
    if (-not $exe) { return $false }
    try {
        if ($Parts.Length -gt 1) {
            $ver = & $Parts[0] $Parts[1] -c "import sys; print(sys.version_info[0])"
        } else {
            $ver = & $Parts[0] -c "import sys; print(sys.version_info[0])"
        }
        return ($LASTEXITCODE -eq 0 -and $ver -eq "3")
    } catch {
        return $false
    }
}

$script = Join-Path $PSScriptRoot "run_all.py"
if (Test-Python @("py", "-3.13")) {
    Write-Host "Using py -3.13"
    & py -3.13 $script @args
    exit $LASTEXITCODE
}
if (Test-Python @("py", "-3")) {
    Write-Host "Using py -3"
    & py -3 $script @args
    exit $LASTEXITCODE
}
if (Test-Python @("python")) {
    Write-Host "Using python"
    & python $script @args
    exit $LASTEXITCODE
}
throw "Python 3.13 not found. Install 64-bit Python from python.org and tick Add python.exe to PATH."
