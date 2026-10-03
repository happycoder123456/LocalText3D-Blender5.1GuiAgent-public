@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set ATTN_BACKEND=xformers
set SPCONV_ALGO=native
set XFORMERS_FORCE_DISABLE_TRITON=1
if exist "vendor\TRELLIS" set "PYTHONPATH=%CD%\vendor\TRELLIS;%PYTHONPATH%"

rem %ERRORLEVEL% inside ( ... ) blocks expands at parse time; use labels instead.
if exist ".venv-trellis\Scripts\python.exe" goto :run_trellis
if not exist ".venv\Scripts\python.exe" goto :bootstrap
goto :run_shap_e

:run_trellis
rem PyTorch C++ extensions need ninja on PATH (installed into Scripts by pip).
set "PATH=%CD%\.venv-trellis\Scripts;%PATH%"
echo Starting local worker with TRELLIS+Shap-E on http://127.0.0.1:8765
echo This address is this computer only. Do not port-forward it.
echo Leave this window open, then generate from the Blender Text to 3D panel.
".venv-trellis\Scripts\python.exe" -m worker serve %*
set "EC=%ERRORLEVEL%"
goto :done

:bootstrap
echo No .venv yet. Installing Shap-E, then starting the worker.
call "%~dp0run_all.bat" %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo Failed with error code %EC%.
  pause
)
exit /b %EC%

:run_shap_e
echo Starting local worker on http://127.0.0.1:8765
echo This address is this computer only. Do not port-forward it.
echo Leave this window open, then generate from the Blender Text to 3D panel.
echo For TRELLIS: py -3.13 scripts\setup_trellis.py
".venv\Scripts\python.exe" -m worker serve %*
set "EC=%ERRORLEVEL%"

:done
if not "%EC%"=="0" (
  echo.
  echo Worker exited with error code %EC%.
  pause
)
exit /b %EC%
