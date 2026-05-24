# run_fast_etl.ps1
# Runs the Stock ETL Pipeline via run.py in Fast Mode with No-Sync to Supabase Cloud natively on Windows.

Write-Host "🚀 Starting Stock ETL Pipeline (Fast Mode / No-Sync)..." -ForegroundColor Cyan
Write-Host "--------------------------------------------------------" -ForegroundColor Gray

# Ensure Unicode characters (like emojis) are printed correctly
$env:PYTHONIOENCODING="utf-8"

# Check virtual environment
if (Test-Path ".venv") {
    Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
    .\.venv\Scripts\Activate.ps1
    $PYTHON_EXEC = "python"
} else {
    $PYTHON_EXEC = "python"
}

# Run the ETL script
& $PYTHON_EXEC run.py --fast --no-sync

Write-Host "--------------------------------------------------------" -ForegroundColor Gray
Write-Host "✅ ETL Run Finished." -ForegroundColor Green
