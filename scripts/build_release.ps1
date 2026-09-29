# Builds the portable Windows folder: release\UltimateVideo3DStudio\UltimateVideo3DStudio.exe
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python = ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run .\scripts\setup.ps1 first." }
if (-not (Test-Path "ultimate_video_3d\assets\onnx\Depth-Anything-V2-Small-hf.fp16.onnx")) {
    throw "The depth model is missing. Run: $Python scripts\get_model.py"
}
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed; not building." }
& $Python -m PyInstaller --noconfirm --clean --distpath release --workpath build UltimateVideo3DStudio.spec
# The licence and readme travel with the app, beside the exe.
Copy-Item -LiteralPath LICENSE, README.md -Destination "release\UltimateVideo3DStudio" -Force
Write-Host "Built release\UltimateVideo3DStudio" -ForegroundColor Green
