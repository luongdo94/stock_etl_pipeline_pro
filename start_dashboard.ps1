# start_dashboard.ps1
# Starts the Stock Dashboard natively on Windows.

$PORT = 8503
Write-Host "🚀 Starting Stock Dashboard on port $PORT..." -ForegroundColor Cyan

# Check for existing process on port 8503
$nets = netstat -ano | Select-String "LISTENING" | Select-String ":$PORT\b"
if ($nets) {
    # Extract PID (last column)
    $pidStr = ($nets.ToString() -split '\s+')[-1]
    if ($pidStr -and $pidStr -ne "0") {
        $existingPid = [int]$pidStr
        Write-Host "🔄 Stopping existing process on port $PORT (PID: $existingPid)..." -ForegroundColor Yellow
        Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

# Force Local Mode for local development
$env:SUPABASE_REMOTE_MODE = "false"
$env:PYTHONIOENCODING = "utf-8"

# Check virtual environment
if (Test-Path ".venv") {
    Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
    .\.venv\Scripts\Activate.ps1
    $PYTHON_EXEC = "python"
} else {
    $PYTHON_EXEC = "python"
}

# Run app.py using Streamlit
& $PYTHON_EXEC -m streamlit run app.py --server.port $PORT
