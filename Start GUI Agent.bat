@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LocalText3D GUI Agent
if not exist ".venv-agent\Scripts\python.exe" goto :missing

echo GUI Agent sidecar on http://127.0.0.1:8766 on this computer only. Do not port-forward it.
echo Leave this window open.
".venv-agent\Scripts\python.exe" -m agent serve %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo Agent exited with error code %EC%. Read the messages above.
  pause
)
exit /b %EC%

:missing
echo Missing .venv-agent. Run "Setup GUI Agent.bat" first.
pause
exit /b 1
