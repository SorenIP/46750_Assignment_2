@echo off
cd /d "%~dp0"
title DP queue - do not close
echo Question 2 job queue. It runs unattended for many hours and restarts itself if python crashes.
echo To end it: create the file queue\STOP_QUEUE.txt. Do not close this window.
:again
python run_dp_queue.py
if exist queue\STOP_QUEUE.txt goto end
echo python ended, restarting in 30 s ...
ping -n 31 127.0.0.1 >nul
goto again
:end
echo Queue ended.
pause >nul
