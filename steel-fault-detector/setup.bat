@echo off
echo ========================================
echo Steel Plate Fault Detector - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Creating Python virtual environment...
cd backend
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)

echo [4/4] Creating sample model (untrained)...
python train_model.py

echo.
echo ========================================
echo Backend setup complete!
echo ========================================
echo.
echo To start the backend server:
echo   cd backend
echo   venv\Scripts\activate
echo   python app.py
echo.
echo ========================================
echo Now setting up Frontend...
echo ========================================
echo.

cd ..\frontend

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    echo Then run: cd frontend ^&^& npm install ^&^& npm start
    pause
    exit /b 0
)

echo Installing Node.js dependencies...
npm install
if errorlevel 1 (
    echo [ERROR] Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the application:
echo.
echo 1. Start the Backend (Terminal 1):
echo    cd steel-fault-detector\backend
echo    venv\Scripts\activate
echo    python app.py
echo.
echo 2. Start the Frontend (Terminal 2):
echo    cd steel-fault-detector\frontend
echo    npm start
echo.
echo The app will open at http://localhost:3000
echo ========================================
pause
