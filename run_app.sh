#!/usr/bin/env bash
# Starts EmotiSense (local model + web app) at http://localhost:8000
cd "$(dirname "$0")"
if [ ! -f web/dist/index.html ]; then
  echo "Building the interface for the first time..."
  (cd web && npm install && npm run build)
fi
python -m emotisense web --open
