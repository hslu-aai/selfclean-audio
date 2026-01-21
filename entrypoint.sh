#!/bin/bash

# Fix git directory ownership for Docker mounted volumes
git config --global --add safe.directory /workspace

# Install package in editable mode if not already installed or if code changed
if [ ! -f /tmp/.installed ] || [ requirements.txt -nt /tmp/.installed ]; then
    echo "Installing package in editable mode..."
    pip install -e . --no-deps
    touch /tmp/.installed
fi

# Execute the original command
exec "$@"
