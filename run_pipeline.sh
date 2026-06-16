#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/forecasting/src"
python forecasting/scripts/run_capsdac_pipeline.py
python forecasting/scripts/generate_visualization_report.py
pytest
