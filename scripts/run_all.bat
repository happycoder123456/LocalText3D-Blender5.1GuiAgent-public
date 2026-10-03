@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

rem NOTE: %ERRORLEVEL% inside a parenthesised block is expanded at parse time,
rem so each runner jumps to its own label and captures the code at top level.
where py >nul 2>&1
if errorlevel 1 goto :try_python
py -3.13 -c "import sys" >nul 2>&1
if not errorlevel 1 goto :run_py313
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto :run_py3

:try_python
where python >nul 2>&1
if not errorlevel 1 goto :run_python

echo.
echo ERROR: Python was not found.
echo Install 64-bit Python 3.13 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH".
echo.
pause
exit /b 1

:run_py313
py -3.13 "%~dp0run_all.py" %*
set "EC=%ERRORLEVEL%"
goto :done

:run_py3
py -3 "%~dp0run_all.py" %*
set "EC=%ERRORLEVEL%"
goto :done

:run_python
python "%~dp0run_all.py" %*
set "EC=%ERRORLEVEL%"
goto :done

:done
if not "%EC%"=="0" (
  echo.
  echo Setup or worker exited with error code %EC%.
  pause
)
exit /b %EC%
