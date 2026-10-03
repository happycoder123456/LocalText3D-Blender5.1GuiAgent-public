@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
if not exist ".venv-agent\Scripts\python.exe" goto :missing

echo GUI Agent sidecar on http://127.0.0.1:8766 on this computer only. Do not port-forward it.
".venv-agent\Scripts\python.exe" -m agent serve %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo Agent exited with error code %EC%.
  pause
)
exit /b %EC%

:missing
echo Run: py -3 scripts\setup_agent.py
echo Then re-run this script.
pause
exit /b 1
