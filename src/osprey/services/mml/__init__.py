"""MATLAB Middle Layer (MML) install service.

Loads a facility's Accelerator Objects export (JSON or ``.mat``), normalises
every family body into one canonical spelling, and feeds the mapping and
emit stages. Nothing facility-specific lives here: families and fields are
discovered structurally.
"""
