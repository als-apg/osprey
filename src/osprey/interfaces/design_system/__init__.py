"""OSPREY design system: shared design tokens and theming runtime.

Single source of truth for the design tokens (colors, fonts, spacing, chart
and terminal palettes, ...) consumed by every OSPREY browser interface.
Tokens are hand-authored as DTCG JSON under ``tokens/`` and compiled by
``generator/`` into checked-in CSS/JS artifacts under ``static/``, which
every interface app mounts at ``/design-system``.
"""
