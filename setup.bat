@echo off
echo ============================================================
echo  IBM AICTE -- Interview Trainer Agent
echo  Setup Script (Windows)
echo ============================================================

echo.
echo [1/4] Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

echo.
echo [2/4] Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [3/4] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [4/4] Setting up .env file...
if not exist .env (
    copy .env.example .env
    echo .env file created from .env.example
    echo IMPORTANT: Open .env and fill in your IBM Cloud credentials!
) else (
    echo .env already exists.
)

echo.
echo ============================================================
echo  Setup complete!
echo  
echo  Next steps:
echo  1. Open .env and add your IBM watsonx.ai credentials
echo  2. Run: start.bat
echo  3. Open: http://localhost:8000
echo ============================================================
pause
