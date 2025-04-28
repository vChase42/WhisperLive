@echo off
setlocal enabledelayedexpansion

echo [*] Launching Docker server...

docker run --gpus all -p 9090:9090 --rm --name b2b_server -d b2b_server

echo [*] Activating virtual environment and starting client...

call client_venv\Scripts\activate
python b2b-client.py
