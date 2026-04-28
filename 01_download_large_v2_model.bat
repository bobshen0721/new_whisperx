@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0tools\download_and_reconstruct_large_v2.ps1" (
  echo [ERROR] Required file was not found:
  echo %~dp0tools\download_and_reconstruct_large_v2.ps1
  echo.
  echo Please extract the entire GitHub ZIP first, then run this BAT from the extracted folder.
  echo Do not run it directly from inside the ZIP preview window.
  pause
  exit /b 1
)

if not exist "%~dp0models\faster-whisper-large-v2\release-manifest.json" (
  echo [ERROR] Model manifest was not found:
  echo %~dp0models\faster-whisper-large-v2\release-manifest.json
  echo.
  echo Please re-download the project ZIP from GitHub and extract the entire folder.
  pause
  exit /b 1
)

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
