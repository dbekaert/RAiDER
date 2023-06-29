#!/bin/bash --login
set -e
conda activate RAiDER
exec python -um RAiDER.cli "$@"
