#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e

# Check if CATEGORY is equal to 'itt'
if [ "$CATEGORY" = "itt" ]; then
  # Copy and install only if CATEGORY is 'itt'
  cp -r working/external/LLaVA .
  pip install --no-cache-dir -e .
fi
