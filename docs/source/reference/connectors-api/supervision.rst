.. _connectors-api-supervision:

Supervising Connector Processes
===============================

What a supervisor needs to run connectors in child processes: the connector
host's proxy and pool, where a target's child will connect, the EPICS
environment a child is started without, starting and stopping children, and
the policy for a call that failed.

.. automodule:: osprey_connectors.ipc

.. automodule:: osprey_connectors.gateways

.. automodule:: osprey_connectors.epics_env

.. automodule:: osprey_connectors.process

.. automodule:: osprey_connectors.recovery
