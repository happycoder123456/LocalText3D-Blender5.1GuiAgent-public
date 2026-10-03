#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Local Text to 3D"
echo "Installing Shap-E if needed, then starting the worker."
echo "Worker listens on http://127.0.0.1:8765 on this computer only. Do not port-forward it."
exec python3 scripts/run_all.py "$@"
