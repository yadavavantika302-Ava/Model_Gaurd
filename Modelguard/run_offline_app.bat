@echo off
cd /d "%~dp0"
echo Starting ModelGuard offline app...
echo Open http://127.0.0.1:5000 in your browser.
set "PYTHON_RUNNER=python"
py -3.13 -c "import flask, PIL" >nul 2>&1
if not errorlevel 1 set "PYTHON_RUNNER=py -3.13"
%PYTHON_RUNNER% -m flask --app src.app run --host 127.0.0.1 --port 5000
pause
