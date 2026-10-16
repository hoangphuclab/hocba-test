@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_predict.bat path\to\transcript.jpg
  exit /b 2
)
python "%~dp0runtime.py" "%~1" --preview "%~dp0prediction_overlay.jpg" --json "%~dp0prediction.json"
pause
