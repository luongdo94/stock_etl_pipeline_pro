# 🪟 Migration Guide: macOS → Windows

## 📋 Overview

This guide helps you migrate the Honest Quant Intelligence Platform from macOS to Windows.

---

## 🎯 Quick Migration Steps

### Option A: Git Clone (Recommended)

```bash
# On Windows (PowerShell or Git Bash)
git clone <your-repo-url>
cd stock_etl_pipeline_kiro
```

### Option B: Manual Transfer

1. **Compress on macOS:**
   ```bash
   # Exclude large files
   tar -czf stock_etl_backup.tar.gz \
     --exclude='.venv' \
     --exclude='warehouse/*.duckdb' \
     --exclude='warehouse/*.duckdb.wal' \
     --exclude='.cache' \
     --exclude='__pycache__' \
     --exclude='.pytest_cache' \
     --exclude='logs' \
     --exclude='.git' \
     .
   ```

2. **Transfer to Windows:**
   - USB drive
   - Cloud storage (Google Drive, Dropbox)
   - Network share

3. **Extract on Windows:**
   ```powershell
   # Using 7-Zip or Windows built-in
   tar -xzf stock_etl_backup.tar.gz
   ```

---

## 🔧 Windows Setup

### 1. Install Prerequisites

#### Python 3.9+
```powershell
# Download from python.org or use winget
winget install Python.Python.3.9

# Verify
python --version
```

#### Git (if using Option A)
```powershell
winget install Git.Git
```

#### Visual Studio Build Tools (for some Python packages)
```powershell
# Download from: https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++"
```

### 2. Create Virtual Environment

```powershell
# Navigate to project directory
cd stock_etl_pipeline_kiro

# Create venv
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate (Command Prompt)
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# If any package fails, try:
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

### 4. Transfer Database

#### Option 1: Copy Database Files
```powershell
# On macOS, compress database
cd warehouse
tar -czf stock_dw_backup.tar.gz stock_dw.duckdb

# Transfer to Windows, then extract
tar -xzf stock_dw_backup.tar.gz
```

#### Option 2: Use Supabase/Cloud Storage
```powershell
# If you have cloud sync enabled
python run.py --only-sync  # Download from cloud
```

#### Option 3: Rebuild from Scratch
```powershell
# Full ETL run (takes time but ensures clean state)
python run.py --full
```

### 5. Configure Environment

```powershell
# Copy .env file from macOS or create new
# Edit .env with Windows paths if needed

# Example .env for Windows:
SUPABASE_URL=your_url
SUPABASE_SERVICE_KEY=your_key
COHERE_API_KEY=your_key
```

---

## 🔄 Path Differences (macOS vs Windows)

### File Paths

| macOS | Windows |
|-------|---------|
| `/Users/luongdo/project` | `C:\Users\luongdo\project` |
| `warehouse/stock_dw.duckdb` | `warehouse\stock_dw.duckdb` |
| Forward slashes `/` | Backslashes `\` (or forward `/` works too) |

### Python handles this automatically in most cases!

```python
# ✅ Cross-platform (works on both)
import os
db_path = os.path.join('warehouse', 'stock_dw.duckdb')

# ✅ Also works on both
from pathlib import Path
db_path = Path('warehouse') / 'stock_dw.duckdb'
```

---

## 🐳 Docker Alternative (Easiest!)

If you want identical environment on both systems:

### 1. Install Docker Desktop for Windows
```powershell
winget install Docker.DockerDesktop
```

### 2. Use Docker Compose
```powershell
# Start services
docker-compose up -d

# Access Airflow
# http://localhost:8080
```

### 3. Run Dashboard in Container
```dockerfile
# Create Dockerfile if not exists
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```powershell
# Build and run
docker build -t stock-dashboard .
docker run -p 8501:8501 -v ${PWD}/warehouse:/app/warehouse stock-dashboard
```

---

## ✅ Verification Checklist

### After Migration, Test:

```powershell
# 1. Check Python
python --version

# 2. Check dependencies
pip list | Select-String "streamlit|duckdb|pandas"

# 3. Test database connection
python -c "import duckdb; conn = duckdb.connect('warehouse/stock_dw.duckdb', read_only=True); print('✅ DB OK'); conn.close()"

# 4. Test dashboard
streamlit run app.py

# 5. Test ETL
python run.py --fast

# 6. Check insider data
python quick_check_insider.py
```

---

## 🚨 Common Windows Issues & Fixes

### Issue 1: Long Path Names
```powershell
# Enable long paths in Windows
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Or use Git Bash which handles long paths better
```

### Issue 2: Line Endings (CRLF vs LF)
```powershell
# Configure Git to handle line endings
git config --global core.autocrlf true
```

### Issue 3: Permission Errors
```powershell
# Run PowerShell as Administrator for some operations
# Right-click PowerShell → "Run as Administrator"
```

### Issue 4: Firewall Blocking Streamlit
```powershell
# Add firewall rule
New-NetFirewallRule -DisplayName "Streamlit" -Direction Inbound -Program "C:\Path\To\Python\python.exe" -Action Allow
```

### Issue 5: DuckDB Lock Issues
```powershell
# Close all Python processes
Get-Process python | Stop-Process -Force

# Remove lock files
Remove-Item warehouse\*.wal -Force
```

---

## 📊 Performance Considerations

### Windows-Specific Optimizations:

1. **Disable Windows Defender for Project Folder** (speeds up file I/O)
   ```
   Settings → Virus & threat protection → Exclusions
   Add: C:\Users\luongdo\stock_etl_pipeline_kiro
   ```

2. **Use SSD for Database**
   - Store `warehouse/` folder on SSD, not HDD

3. **Increase Virtual Memory**
   ```
   Settings → System → About → Advanced system settings
   → Performance Settings → Advanced → Virtual memory
   Set to 1.5x your RAM size
   ```

---

## 🔄 Ongoing Sync Between Machines

### Option 1: Git + Cloud Database
```bash
# On macOS: Push code changes
git add .
git commit -m "Update"
git push

# Sync database to cloud
python run.py --only-sync

# On Windows: Pull changes
git pull

# Download database from cloud
python run.py --only-sync
```

### Option 2: Supabase for Database
```python
# Both machines use Supabase as source
# Set in .env:
SUPABASE_REMOTE_MODE=true
```

### Option 3: Network Share
```powershell
# Share warehouse folder on network
# Access from both machines
```

---

## 🎯 Recommended Workflow

### Development on Windows:

1. **Use WSL2 (Windows Subsystem for Linux)** - Best of both worlds!
   ```powershell
   # Install WSL2
   wsl --install
   
   # Install Ubuntu
   wsl --install -d Ubuntu
   
   # Inside WSL, clone project
   cd ~
   git clone <repo>
   cd stock_etl_pipeline_kiro
   
   # Setup as if on Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   
   # Run dashboard (accessible from Windows browser)
   streamlit run app.py
   ```

2. **Use VS Code with Remote-WSL Extension**
   - Edit files in Windows
   - Run code in Linux environment
   - Best compatibility!

---

## 📝 Migration Checklist

- [ ] Install Python 3.9+
- [ ] Install Git (if using)
- [ ] Transfer project files
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Transfer/rebuild database
- [ ] Copy .env file
- [ ] Test database connection
- [ ] Test dashboard
- [ ] Test ETL pipeline
- [ ] Verify insider data
- [ ] Setup Airflow (if needed)
- [ ] Configure firewall
- [ ] Add antivirus exclusions
- [ ] Test all features

---

## 🆘 Troubleshooting

### Get Help:
```powershell
# Check Python environment
python -c "import sys; print(sys.executable); print(sys.version)"

# Check installed packages
pip list

# Check database
python quick_check_insider.py

# Check logs
Get-Content logs\pipeline_*.log -Tail 50
```

### Common Commands (PowerShell vs Bash):

| Task | macOS/Linux | Windows PowerShell |
|------|-------------|-------------------|
| List files | `ls -la` | `Get-ChildItem` or `dir` |
| View file | `cat file.txt` | `Get-Content file.txt` |
| Find text | `grep "text" file` | `Select-String "text" file` |
| Environment vars | `export VAR=value` | `$env:VAR="value"` |
| Kill process | `kill -9 PID` | `Stop-Process -Id PID -Force` |

---

## 🎉 Success Indicators

After migration, you should see:

✅ Dashboard runs: `http://localhost:8501`  
✅ Database accessible: `warehouse/stock_dw.duckdb`  
✅ Insider data visible in Stock Scanner tab  
✅ ETL pipeline runs without errors  
✅ All charts and visualizations working  

---

## 📞 Support

If you encounter issues:

1. Check this guide first
2. Review error messages carefully
3. Check Windows Event Viewer for system errors
4. Verify all paths use correct separators
5. Ensure antivirus isn't blocking Python

---

**Good luck with your migration! 🚀**

*Last Updated: 2026-05-15*  
*Tested on: Windows 10/11, Python 3.9+*
