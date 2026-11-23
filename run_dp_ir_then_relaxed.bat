@echo off
cd /d "%~dp0"
if not exist results\base\DP_variants mkdir results\base\DP_variants
echo Running budget_balance_ir (stopped automatically after 1.5 h), then the relaxed IR variant. Do not close this window.
start /b cmd /c "ping -n 5400 127.0.0.1 >nul & echo stop> results\base\DP_variants\STOP_budget_balance_ir.txt"
python main.py --DP-variants budget_balance_ir budget_balance_relaxed_ir
pause >nul
