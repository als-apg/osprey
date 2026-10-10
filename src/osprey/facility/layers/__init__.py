"""Layers: the importers that turn an outside export into facility sources.

Each layer reads one outside format and writes record sources under
``data/facility/imported/<layer>/``; the build merges them with the authored
sources field by field.
"""
