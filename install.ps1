# Windows: creates .venv next to this file and installs CPU PyTorch + the rest.
# Usage:  powershell -ExecutionPolicy Bypass -File install.ps1   (add -Cuda to use your NVIDIA GPU build instead)
param([switch]$Cuda)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -m venv .venv
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $py -m pip install --upgrade pip
if ($Cuda) { & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124 }
else       { & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu }
& $py -m pip install -r requirements.txt
& $py -c "import torch, transformers, librosa; print('ok', torch.__version__, transformers.__version__, librosa.__version__)"
Write-Host "Done. Run:  .venv\Scripts\python.exe scripts\analyze.py <file>"
