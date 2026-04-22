@echo off
setlocal
cd /d "%~dp0"

echo Starting Speech Transcription Platform 2.0v ...
if "%WHISPERX_HF_TOKEN%"=="" (
  echo [Notice] WHISPERX_HF_TOKEN is not set. Speaker diarization will fall back to UNKNOWN.
)

if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" app.py --server-name 127.0.0.1 --server-port 7860
  goto end
)

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 app.py --server-name 127.0.0.1 --server-port 7860
  goto end
)

where python >nul 2>nul
if %errorlevel%==0 (
  python app.py --server-name 127.0.0.1 --server-port 7860
  goto end
)

echo [ERROR] Python was not found in PATH.
echo Please install Python 3.10 - 3.13, or open README.md and follow the setup steps.
pause
exit /b 1

:end
if errorlevel 1 (
  echo.
  echo [ERROR] Web UI failed to start. Check the README for environment setup steps.
  pause
  exit /b 1
)
