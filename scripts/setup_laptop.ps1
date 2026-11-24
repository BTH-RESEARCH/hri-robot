# Windows laptop setup. Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\setup_laptop.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path .venv)) {
    Write-Host "==> Creating .venv"
    python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip wheel
& $py -m pip install -e ".[dev]" -f https://k2-fsa.github.io/icefall/piper_phonemize.html

if (-not (Test-Path .env)) { Copy-Item .env.example .env }

Write-Host "==> Models"
& .\.venv\Scripts\hri.exe --profile laptop models download

& .\.venv\Scripts\hri.exe --profile laptop info
Write-Host ""
Write-Host "Done. Next: .\.venv\Scripts\Activate.ps1 ; notepad .env (API keys) ; hri chat --fake-emotion sad"
