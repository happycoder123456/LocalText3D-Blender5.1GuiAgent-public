@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Local Text to 3D
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Local Text to 3D
echo Installing engines if needed, then starting the worker.
echo Worker listens on http://127.0.0.1:8765 on this computer only. Do not port-forward it.
echo Leave this window open.
echo.
echo Tip: for TRELLIS on Windows, run:
echo   py -3.13 scripts\run_all.py --with-trellis
echo.

set "RUNNER="
where py >nul 2>&1
if not errorlevel 1 (
  py -3.13 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "RUNNER=py313"
)
if not defined RUNNER (
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "RUNNER=py3"
  )
)
if not defined RUNNER (
  where python >nul 2>&1
  if not errorlevel 1 (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "RUNNER=python"
  )
)

if not defined RUNNER (
  echo.
  echo ERROR: Python was not found.
  echo Install 64-bit Python 3.13 from https://www.python.org/downloads/
  echo During setup, tick "Add python.exe to PATH".
  echo Then double-click START.bat again.
  echo.
  echo You do not need Linux or WSL.
  echo.
  pause
  exit /b 1
)

if "%RUNNER%"=="py313" (
  echo Using: py -3.13
  py -3.13 scripts\run_all.py %*
) else if "%RUNNER%"=="py3" (
  echo Using: py -3
  py -3 scripts\run_all.py %*
) else (
  echo Using: python
  python scripts\run_all.py %*
)

set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo Setup or worker exited with error code %EC%.
  echo Read the messages above, then press any key to close.
  echo.
  pause
)
exit /b %EC%
