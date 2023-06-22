#!/bin/bash --login
set -e
conda activate RAiDER
exec python -u raider.py "$@"
