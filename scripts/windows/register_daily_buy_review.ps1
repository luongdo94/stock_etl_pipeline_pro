# scripts/windows/register_daily_buy_review.ps1
# Registers a Windows Scheduled Task that asks ChatGPT (Codex CLI, your ChatGPT login) to review every BUY CANDIDATE
# of the newest score snapshot (scripts/tools/daily_buy_review.py). It must run on THIS machine: the Codex login is here.
# Schedule it after the ETL (Airflow: 07:30 Europe/Berlin, about 5-10 minutes) - the default 08:15 leaves a margin.
#     powershell -ExecutionPolicy Bypass -File .\scripts\windows\register_daily_buy_review.ps1 [-Time 08:15] [-Python C:\path\python.exe]
# Remove it again with:  Unregister-ScheduledTask -TaskName "StockETL-BuyReview" -Confirm:$false
param(
    [string]$Time = "08:15",
    [string]$Python = ""
)

$project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) {
    $venvPy = Join-Path $project ".venv\Scripts\python.exe"
    $Python = if (Test-Path $venvPy) { $venvPy } else { (Get-Command python -ErrorAction Stop).Source }
}
$log = Join-Path $project "logs\buy_review.log"
New-Item -ItemType Directory -Force (Join-Path $project "logs") | Out-Null

$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c set PYTHONIOENCODING=utf-8 && `"$Python`" scripts\tools\daily_buy_review.py >> `"$log`" 2>&1" `
    -WorkingDirectory $project
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday,Saturday -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RunOnlyIfNetworkAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "StockETL-BuyReview" -Action $action -Trigger $trigger -Settings $settings `
    -Description "ChatGPT review of the BUY CANDIDATEs ($project)" -Force | Out-Null

Write-Host "Registered 'StockETL-BuyReview': Mon-Sat at $Time using $Python" -ForegroundColor Green
Write-Host "Log: $log"
