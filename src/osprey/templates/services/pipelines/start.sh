#!/bin/bash
# Pipelines startup wrapper - installs framework
set -e

echo "=============================================="
echo "Starting Pipelines Container"
echo "=============================================="

# Install project dependencies (including framework)
if [ -f "/pipelines/repo_src/requirements.txt" ]; then
    echo "Installing project dependencies from requirements.txt..."
    pip install --no-cache-dir -r /pipelines/repo_src/requirements.txt
    echo "✓ Project dependencies installed successfully"
else
    echo "WARNING: /pipelines/repo_src/requirements.txt not found"
    echo "Installing framework as fallback (includes Claude Code SDK)..."
    pip install "osprey-framework>=0.9.6"
fi

# Development mode override - install local osprey AFTER everything else
if [ "$DEV_MODE" = "true" ]; then
    # Find and install osprey wheel if it exists
    OSPREY_WHEEL=$(find /pipelines -maxdepth 1 -name "osprey_framework-*.whl" | head -1)

    if [ -n "$OSPREY_WHEEL" ]; then
        echo "🔧 Development mode: Installing local osprey wheel..."
        pip install --no-cache-dir --force-reinstall "$OSPREY_WHEEL"
        echo "✓ Osprey overridden with local development version from wheel"
    else
        echo "⚠️  Dev mode enabled but no osprey wheel found, using PyPI version"
    fi
else
    echo "📦 Using PyPI osprey version"
fi
