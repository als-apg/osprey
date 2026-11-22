"""The framework's simulation engine plug-ins: ``pyat`` and ``injection``.

Each module implements the engine contract of
:mod:`osprey_connectors.simulation.engines` and registers under its
:data:`~osprey_connectors.simulation.engines.ENGINE_GROUP` entry-point group. A
build reaches an engine only through its entry point, never by importing the
module by name. The contract, the group and the calibration math live in
``osprey_connectors``; this package holds implementations only.
"""
