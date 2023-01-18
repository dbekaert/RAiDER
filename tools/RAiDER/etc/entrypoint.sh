#!/bin/bash --login
set -e
conda activate raider
exec raider.py "$@"
