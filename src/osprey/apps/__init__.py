"""Applications: deployment-supplied containers, panels and tool servers.

An application lives in ``apps/<name>/`` of a deployment, or in a package an
installed distribution declares (:mod:`osprey.apps.origin`), and is described by its
``app.yml`` (:mod:`osprey.apps.manifest`), which :func:`osprey.apps.loader.load`
discovers and validates.
"""
