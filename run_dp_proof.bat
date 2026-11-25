@echo off
cd /d "%~dp0"
if not exist results\base\DP_variants mkdir results\base\DP_variants
echo Proof run: smallest possible IR shortfall (stops after 60 min), then min cost at that shortfall (stops at 90 min in total).
echo Do not close this window.
start /b cmd /c "ping -n 3600 127.0.0.1 >nul & echo stop> results\base\DP_variants\STOP_ir_shortfall_only.txt"
start /b cmd /c "ping -n 5400 127.0.0.1 >nul & echo stop> results\base\DP_variants\STOP_relaxed_ir_stage2.txt"
python main.py --DP-variants ir_shortfall_only relaxed_ir_stage2
pause >nul
