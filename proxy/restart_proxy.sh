#!/usr/bin/env bash
set -euo pipefail

cd /Users/mac/Desktop/pet/proxy
source /Users/mac/Desktop/pet/.venv/bin/activate

# Stop existing proxy if running
pkill -f "/Users/mac/Desktop/pet/proxy/server.py" || true

# Start proxy
python /Users/mac/Desktop/pet/proxy/server.py
