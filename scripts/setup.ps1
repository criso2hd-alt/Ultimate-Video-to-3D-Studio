# Creates .venv with Python 3.12, installs the app, and fetches the depth model.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Test-Python312 {
    param([string[]]$Command)
    try {
        $prev = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
        & $Command[0] @($Command[1..($Command.Length - 1)]) -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>&1 | Out-Null
        $ErrorActionPreference = $prev
        return $LASTEXITCODE -eq 0
    } catch { $ErrorActionPreference = $prev; return $false }
}

$Py = $null
if (Test-Python312 @("py", "-3.12")) { $Py = @("py", "-3.12") }
elseif (Test-Python312 @("python")) { $Py = @("python") }
if ($null -eq $Py) { throw "Python 3.12 (64-bit) is required. Install it from python.org and run this again." }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Py[0] @($Py[1..($Py.Length - 1)]) -m venv .venv
}
& .venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
& .venv\Scripts\python.exe -m pip install -e ".[test,build]"
& .venv\Scripts\python.exe scripts\get_model.py
Write-Host "Done. Run .\scripts\run.ps1" -ForegroundColor Green
