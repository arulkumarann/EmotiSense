#!/usr/bin/env bash
# Starts the EmotiSense dashboard with the trained timeline model (http://localhost:8501).
cd "$(dirname "$0")"
python -m emotisense dashboard --timeline models/timeline_bilstm_deploy.pt
