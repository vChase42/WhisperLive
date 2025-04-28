@echo off
setlocal enabledelayedexpansion

echo [*] Checking required files...

if not exist ".env" (
    echo [ERROR] .env file not found!
    PAUSE
    exit /b 1
)

if not exist "b2b-client.py" (
    echo [ERROR] b2b_client.py file not found!
    PAUSE
    exit /b 1
)

if not exist "run_server.py" (
    echo [ERROR] run_server.py file not found!
    PAUSE
    exit /b 1
)

echo [*] Checking if Python is installed...

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    PAUSE
    exit /b 1
)

echo [*] Building Docker server image...

docker build -f Dockerfile.server -t b2b_server .
if errorlevel 1 (
    echo [ERROR] Docker build failed!
    exit /b 1
)

echo [*] Creating virtual environment...

python -m venv client_venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    exit /b 1
)

echo [*] Activating virtual environment and installing client requirements...

call client_venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements/client.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python packages!
    exit /b 1
)

echo [*] Setup completed successfully.
pause
