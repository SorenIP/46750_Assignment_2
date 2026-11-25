@echo off
cd /d "%~dp0"
echo Running relaxed-IR variant. Do not close this window.
python main.py --DP-variants budget_balance_relaxed_ir
pause >nul
