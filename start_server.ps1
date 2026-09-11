# BeatFlow AI (Melodyfy) - Server Watchdog Launcher
# Usage: powershell -ExecutionPolicy Bypass -File start_server.ps1

$python = "python"
$script = "$PSScriptRoot\run.py"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  BeatFlow AI - Server Watchdog Launcher  " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow

while ($true) {
    Write-Host "[$(Get-Date -f 'HH:mm:ss')] Starting BeatFlow API Server..." -ForegroundColor Green
    Set-Location $PSScriptRoot
    & $python $script
    $code = $LASTEXITCODE
    Write-Host "[$(Get-Date -f 'HH:mm:ss')] Server exited (code $code). Restarting in 3s..." -ForegroundColor Yellow
    Start-Sleep 3
}
