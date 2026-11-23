@echo off
rem Double-click this file. It runs the final Question 2 steps (about 5 to 20 minutes).
rem Needs: Python with gurobipy and a full Gurobi licence (as for the earlier runs).
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (python run_q2_final.py) else (py run_q2_final.py)
echo.
echo Finished. Send the folder results\q2_final and results\base\DP_queue\queue_summary.csv to Claude.
pause
