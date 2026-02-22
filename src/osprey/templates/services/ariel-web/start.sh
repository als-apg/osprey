#!/bin/bash
# ARIEL Web startup wrapper - handles dev mode override
set -e

echo "=============================================="
echo "Starting ARIEL Web Interface"
echo "=============================================="

# Development mode override - install local osprey AFTER base packages
if [ "$DEV_MODE" = "true" ]; then
    # Find and install osprey wheel if it exists (mounted at /app/build in dev mode)
    OSPREY_WHEEL=$(find /app/build -maxdepth 1 -name "osprey_framework-*.whl" 2>/dev/null | head -1)

    if [ -n "$OSPREY_WHEEL" ]; then
        echo "Development mode: Installing local osprey wheel..."
        uv pip install --system --no-cache --force-reinstall --no-deps "$OSPREY_WHEEL"
        echo "Osprey overridden with local development version"
    else
        echo "Dev mode enabled but no osprey wheel found, using installed version"
    fi
else
    echo "Using PyPI osprey version"
fi

# Start the FastAPI application using native osprey interface
echo "Starting ARIEL web server from native osprey interface..."
exec uvicorn osprey.interfaces.ariel.app:create_app --factory --host 0.0.0.0 --port 8085
