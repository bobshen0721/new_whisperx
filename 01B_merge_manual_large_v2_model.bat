@echo off
setlocal
cd /d "%~dp0"

set "DEFAULT_PART_DIR=%~dp0release-assets\large-v2"
set "MANIFEST_PATH=%~dp0models\faster-whisper-large-v2\release-manifest.json"
set "MODEL_DIR=%~dp0models\faster-whisper-large-v2"
set "RECONSTRUCT_SCRIPT=%~dp0tools\reconstruct_large_v2.ps1"

if not exist "%RECONSTRUCT_SCRIPT%" (
  echo [ERROR] Required file was not found:
  echo %RECONSTRUCT_SCRIPT%
  echo.
  echo Please extract the entire GitHub ZIP first, then run this BAT from the extracted folder.
  pause
  exit /b 1
)

if not exist "%MANIFEST_PATH%" (
  echo [ERROR] Model manifest was not found:
  echo %MANIFEST_PATH%
  echo.
  echo Please re-download the project ZIP from GitHub and extract the entire folder.
  pause
  exit /b 1
)

if not exist "%DEFAULT_PART_DIR%" (
  mkdir "%DEFAULT_PART_DIR%" >nul 2>nul
)

echo Manual large-v2 model merge
echo.
echo Download these files from GitHub Release:
echo   faster-whisper-large-v2.model.bin.part01
echo   faster-whisper-large-v2.model.bin.part02
echo.
echo Recommended folder:
echo   %DEFAULT_PART_DIR%
echo.
echo Put both part files in that folder, then press Enter.
echo If you saved them somewhere else, paste that folder path below.
echo.
set "PART_DIR="
set /p "PART_DIR=Part folder [Enter = recommended folder]: "
if "%PART_DIR%"=="" set "PART_DIR=%DEFAULT_PART_DIR%"
set "PART_DIR=%PART_DIR:"=%"

if not exist "%PART_DIR%" (
  echo.
  echo [ERROR] Folder not found:
  echo %PART_DIR%
  pause
  exit /b 1
)

if not exist "%PART_DIR%\faster-whisper-large-v2.model.bin.part01" (
  echo.
  echo [ERROR] Missing part file:
  echo %PART_DIR%\faster-whisper-large-v2.model.bin.part01
  pause
  exit /b 1
)

if not exist "%PART_DIR%\faster-whisper-large-v2.model.bin.part02" (
  echo.
  echo [ERROR] Missing part file:
  echo %PART_DIR%\faster-whisper-large-v2.model.bin.part02
  pause
  exit /b 1
)

echo.
echo [1/2] Verifying parts and merging model.bin...
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%RECONSTRUCT_SCRIPT%" -ManifestPath "%MANIFEST_PATH%" -PartDirectory "%PART_DIR%" -ModelDirectory "%MODEL_DIR%"
if errorlevel 1 (
  echo.
  echo [ERROR] Model reconstruction failed.
  echo Please check that both part files were fully downloaded from the Release page.
  pause
  exit /b 1
)

echo.
echo [2/2] Model is ready:
echo %MODEL_DIR%\model.bin
pause
