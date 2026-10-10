@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed
if not exist "%~dp0.venv\Scripts\python.exe" (
  call "%~dp0setup_windows.cmd" --non-interactive
  if errorlevel 1 goto failed
)
"%~dp0.venv\Scripts\python.exe" -c "import sys, struct; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8 and sys.prefix != sys.base_prefix, 'Use setup_windows.cmd to create a 64-bit Python 3.13 venv'"
if errorlevel 1 goto failed
if /i "%~1"=="--check" goto check
"%~dp0.venv\Scripts\python.exe" "%~dp0desktop_app\main.py"
if errorlevel 1 goto failed
exit /b 0
:check
"%~dp0.venv\Scripts\python.exe" -c "import sys, struct; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8; import lunar_od, scipy, numba, spiceypy, pandas, matplotlib, pyqtgraph, h5py; from PyQt5 import QtWidgets; print('Lunar OD desktop imports: OK')"
if errorlevel 1 exit /b 1
exit /b 0
:failed
echo Lunar OD could not start. Run setup_windows.cmd and check the message above.
pause
exit /b 1
