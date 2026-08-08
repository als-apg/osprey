"""Route modules for the auth sidecar.

Each module here exposes a module-level ``router``; the application factory
includes the ones that exist (see
:func:`~osprey.services.auth_sidecar.app._include_available_routers`), so a new
route lands by adding a file, not by editing the factory.
"""
