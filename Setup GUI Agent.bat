@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Setup GUI Agent

where py >nul 2>&1
if errorlevel 1 goto :try_python
py -3.13 -c "import sys" >nul 2>&1
if not errorlevel 1 goto :run_py313
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto :run_py3

:try_python
where python >nul 2>&1
if not errorlevel 1 goto :run_python
echo ERROR: Python was not found. Install 64-bit Python 3.13 from python.org
echo and tick "Add python.exe to PATH".
pause
exit /b 1

:run_py313
py -3.13 scripts\setup_agent.py %*
set "EC=%ERRORLEVEL%"
goto :done

:run_py3
py -3 scripts\setup_agent.py %*
set "EC=%ERRORLEVEL%"
goto :done

:run_python
python scripts\setup_agent.py %*
set "EC=%ERRORLEVEL%"

:done
if not "%EC%"=="0" echo Setup exited with error code %EC%.
pause
exit /b %EC%
