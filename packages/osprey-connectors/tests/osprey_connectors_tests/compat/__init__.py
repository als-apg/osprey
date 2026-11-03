"""Compatibility checks between released frameworks and this connectors tree.

``scan_framework_imports`` and ``contracts`` are scripts as much as modules:
CI runs them with ``python -I`` inside a venv that holds a published framework
and the connectors wheel built at HEAD, where this checkout is not importable.
Both therefore use the standard library and ``osprey_connectors`` only.
"""
