.. _connectors-api-services:

Directory, alarms and put log
=============================

The channel directory (:ref:`channel-finder-directory`) reads four kinds of
source: a directory of channels and servers, the current alarm state, the
alarm history and the put history. Each kind has a read-only contract, and
this page documents the four contracts and the rule that decides which source
answers on each control target.

These backends ship, each named by its ``type``:

- **Directory:** ``channelfinder``, a ChannelFinder service, and the
  simulator's own list of channels.
- **Alarm state:** ``phoebus_alarm_server``, the Phoebus alarm server through
  its Kafka topic; ``osprey_alarm_relay``, the same state from an OSPREY alarm
  relay over HTTP; and a replayed file for the simulator.
- **Alarm history:** ``phoebus_alarm_logger``, the Phoebus alarm logger.
- **Put history:** ``caputlog``, a caPutLog service.

The shipped services are ones that EPICS facilities run. A facility with a
different service writes a class against the contract of its kind and names
the class by its dotted path where the built-in ``type`` would stand.

.. automodule:: osprey_connectors.service_answers

.. automodule:: osprey_connectors.directory

.. automodule:: osprey_connectors.alarm_state

.. automodule:: osprey_connectors.alarm_log

.. automodule:: osprey_connectors.put_log

.. automodule:: osprey_connectors.service_targets
