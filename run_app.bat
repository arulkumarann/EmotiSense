@echo off
REM Starts EmotiSense (local model + web app) and opens it in your browser.
cd /d "%~dp0"
if not exist "web\dist\index.html" (
  echo Building the interface for the first time...
  pushd web
  call npm install
  call npm run build
  popd
)
python -m emotisense web --open
pause
