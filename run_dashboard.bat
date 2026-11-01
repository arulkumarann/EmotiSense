@echo off
REM Starts the EmotiSense dashboard with the trained timeline model, then opens it in your browser.
cd /d "%~dp0"
python -m emotisense dashboard --timeline models\timeline_bilstm_deploy.pt
pause
