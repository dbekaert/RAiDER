#!/bin/bash --login
set -e
conda activate RAiDER
exec python -um raider.py "$@"
