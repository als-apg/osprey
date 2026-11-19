"""The machines a deployment stands up beside the facility.

:mod:`osprey_connectors.deployment.simulator` and
:mod:`osprey_connectors.deployment.standin` decide whether the virtual
accelerator and the live stand-in are up where a session would dial them;
which control targets the deployment's one ``archiver:`` block serves is
:mod:`osprey_connectors.service_targets`' ``archive`` row. Each module is its
own API and lists it in its own ``__all__``.
"""
