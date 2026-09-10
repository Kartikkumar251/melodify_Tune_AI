# BeatFlow AI (Melodyfy) - Dependency Installer
# Usage: powershell -ExecutionPolicy Bypass -File install_deps.ps1

Write-Host "Installing BeatFlow AI Dependencies..." -ForegroundColor Cyan
python -m pip install -r "$PSScriptRoot\requirements.txt"
Write-Host "Installation complete!" -ForegroundColor Green
