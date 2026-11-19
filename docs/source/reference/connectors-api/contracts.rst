.. _connectors-api-contracts:

Connector Contracts
===================

The two interfaces every connector implements, the door a connector opens
around its own write, and how the archive's MongoDB store names a channel and
is reached. What a conforming connector must do is described in
:doc:`/reference/contracts/connectors`; these are the classes and functions
that carry it.

.. automodule:: osprey_connectors.control_system

Writing a control-system connector
----------------------------------

``connect``, ``write_channel``, ``write_multiple_channels`` and
``write_channel_checked`` are final: a subclass that defines one is refused
when its class is created. A connector implements
:meth:`~osprey_connectors.control_system.ControlSystemConnector._open` in place
of ``connect()``, and
:meth:`~osprey_connectors.control_system.ControlSystemConnector._send`, which
answers with a :class:`~osprey_connectors.control_system.SendReport`; the
base class runs the write guard, the limits and the confirming read, and
builds every result. The other protected hooks are optional.

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._open

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._read_many

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._send

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._precheck

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._confirming_read

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._run_check

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._bound

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._current_values

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._send_batch

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._send_and_confirm

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._write_route_open

.. automethod:: osprey_connectors.control_system.ControlSystemConnector._validation_refusal

.. automodule:: osprey_connectors.control_system.write_door

.. automodule:: osprey_connectors.archiver

.. automodule:: osprey_connectors.archive_store
