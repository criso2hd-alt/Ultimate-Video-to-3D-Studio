$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Not set up yet. Run .\scripts\setup.ps1 first." }
Set-Location -LiteralPath $Root
& $Python main.py @args
