@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup_windows.ps1"
set "LUNAR_EXIT=%ERRORLEVEL%"
if /i "%~1"=="--non-interactive" exit /b %LUNAR_EXIT%
pause
exit /b %LUNAR_EXIT%
