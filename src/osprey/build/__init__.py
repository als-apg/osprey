"""Build-time kernel shared across OSPREY layers.

This package holds build-time helpers that lower layers (``services``,
``mcp_server``) and higher layers (``cli``, ``interfaces``) both consume —
Claude Code model/provider resolution, the OTEL telemetry env block, the
channel-finder tier defaults, and the reproducible-render manifest
primitives.

Layering rule: ``build`` may be imported by any layer, but it must **not**
import from ``cli``, ``interfaces``, or ``services`` (nor ``agent_runner``).
It depends only on lower-level kernels (``osprey.errors``, ``osprey.models``,
``osprey.profiles``, ``osprey.utils``). Keeping it a leaf is what lets the
build/runtime layers reuse it without an inversion.
"""
