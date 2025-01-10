@echo off
REM Navigate to the directory containing your project
cd /d "%~dp0"

REM Activate the virtual environment
call .\venv\Scripts\activate

REM Run the Python script
python run_server.py

REM Pause the terminal (optional, to keep it open after execution)
pause