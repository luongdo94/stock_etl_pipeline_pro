# register_daily_etl.ps1
# Registers a Windows Scheduled Task that runs the ETL (and the daily score snapshot that feeds the
# Track Record tab) every weekday morning. Run once from the project folder:
#     powershell -ExecutionPolicy Bypass -File .\register_daily_etl.ps1 [-Time 07:00] [-Python C:\path\python.exe]
# Remove it again with:  Unregister-ScheduledTask -TaskName "StockETL-Daily" -Confirm:$false
param(
    [string]$Time = "07:00",
    [string]$Python = "",
    [switch]$NoSync
)

$project = $PSScriptRoot
if (-not $Python) {
    $venvPy = Join-Path $project ".venv\Scripts\python.exe"
    $Python = if (Test-Path $venvPy) { $venvPy } else { (Get-Command python -ErrorAction Stop).Source }
}
$runArgs = "run.py" + $(if ($NoSync) { " --no-sync" } else { "" })
$log = Join-Path $project "logs\scheduled_etl.log"
New-Item -ItemType Directory -Force (Join-Path $project "logs") | Out-Null

# cmd wrapper so stdout/stderr are appended to a log file
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c set PYTHONIOENCODING=utf-8 && `"$Python`" $runArgs >> `"$log`" 2>&1" `
    -WorkingDirectory $project
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday,Saturday -At $Time
# IgnoreNew: a slow run must never overlap the next trigger (the ETL also holds its own file lock)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -RunOnlyIfNetworkAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "StockETL-Daily" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Stock ETL pipeline + daily score snapshot ($project)" -Force | Out-Null

Write-Host "Registered 'StockETL-Daily': Mon-Sat at $Time using $Python" -ForegroundColor Green
Write-Host "Log: $log"
