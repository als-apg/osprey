"""The machines a deployment stands up beside the facility, and whose history its archive holds.

:mod:`osprey_connectors.deployment.simulator` and
:mod:`osprey_connectors.deployment.standin` decide whether the virtual
accelerator and the live stand-in are up where a session would dial them, and
:mod:`osprey_connectors.deployment.archive_targets` decides which control
targets the deployment's one ``archiver:`` block serves. Each module is its own
API and lists it in its own ``__all__``.
"""
