@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Downloading large-v2 model parts from GitHub Release...
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\download_and_reconstruct_large_v2.ps1" -Owner bobshen0721 -Repo new_whisperx -Tag model-large-v2
if errorlevel 1 (
  echo.
  echo [ERROR] Model download or reconstruction failed.
  pause
  exit /b 1
)

echo.
echo [2/2] Model is ready:
echo %~dp0models\faster-whisper-large-v2\model.bin
pause
