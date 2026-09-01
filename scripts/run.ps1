# Launch the ReQAgnIze kiosk on Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path ".venv\Scripts\Activate.ps1") { . ".venv\Scripts\Activate.ps1" }
python -m app.main
