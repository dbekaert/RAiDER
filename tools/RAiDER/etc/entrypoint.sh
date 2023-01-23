#!/bin/bash --login
set -e
conda activate RAiDER
exec raider.py "$@"
