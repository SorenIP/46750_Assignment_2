@echo off
cd /d "%~dp0"
echo Starting all four DP variants. Leave this window open.
python main.py --DP-variants
echo.
echo Finished. Press any key to close.
pause >nul
