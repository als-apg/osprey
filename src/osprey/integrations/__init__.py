"""Adapters that let another system run on this deployment's connectors.

Each subpackage serves one peer and imports that peer's own packages, which are
not dependencies of this one: nothing here is imported unless the peer loads
it.
"""
