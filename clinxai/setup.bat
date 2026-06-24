@echo off
TITLE ClinXAI - Automated Setup & Launch
COLOR 0A
echo ==================================================
echo       ClinXAI - Automated Setup System
echo ==================================================
echo.

REM ===== STEP 1: Check Python Installation =====
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
echo [OK] Python is installed.
python --version
echo.

REM ===== STEP 2: Check pip Installation =====
echo [2/5] Checking pip installation...
pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] pip is not installed. Installing pip...
    python -m ensurepip --upgrade
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install pip!
        pause
        exit /b 1
    )
)
echo [OK] pip is installed.
pip --version
echo.

REM ===== STEP 3: Upgrade pip =====
echo [3/5] Upgrading pip to latest version...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.
echo.

REM ===== STEP 4: Install/Update Dependencies =====
echo [4/5] Analyzing and installing dependencies...
echo This may take a few minutes on first run...
echo.

if exist backend\requirements.txt (
    echo Installing from backend\requirements.txt...
    pip install -r backend\requirements.txt --quiet
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Some packages may have failed. Retrying with verbose output...
        pip install -r backend\requirements.txt
    )
    echo [OK] All dependencies installed.
) else (
    echo [WARNING] requirements.txt not found! Installing core packages...
    pip install fastapi uvicorn python-multipart torch torchvision numpy pandas scikit-learn xgboost shap lime pillow matplotlib opencv-python torchxrayvision scikit-image --quiet
)
echo.

REM ===== STEP 5: Launch Application =====
echo [5/5] Launching ClinXAI System...
echo.
echo ==================================================
echo       All checks passed! Starting servers...
echo ==================================================
echo.

REM Start Backend Server
echo [Backend] Starting FastAPI server on port 8000...
start "ClinXAI Backend" cmd /k "echo [BACKEND SERVER] && echo URL: http://127.0.0.1:8000 && echo Docs: http://127.0.0.1:8000/docs && echo. && uvicorn backend.main:app --host 127.0.0.1 --port 8000"

REM Wait for backend to initialize
timeout /t 3 >nul

REM Start Frontend Server
echo [Frontend] Starting HTTP server on port 5500...
start "ClinXAI Frontend" cmd /k "echo [FRONTEND SERVER] && echo URL: http://127.0.0.1:5500 && echo. && python -m http.server 5500 --directory frontend"

REM Wait for frontend to initialize
echo Waiting for servers to initialize...
timeout /t 5 >nul

REM Open Browser
echo.
echo [Browser] Launching application...
start http://127.0.0.1:5500/
echo.

echo ==================================================
echo       ✓ ClinXAI is now running!
echo ==================================================
echo.
echo Backend API:  http://127.0.0.1:8000
echo API Docs:     http://127.0.0.1:8000/docs
echo Frontend:     http://127.0.0.1:5500
echo.
echo Close the server windows to stop the application.
echo ==================================================
pause
