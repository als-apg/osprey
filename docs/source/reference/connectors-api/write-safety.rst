.. _connectors-api-write-safety:

Write Safety
============

The records and verdicts a write passes before anything is put: the read-only
run, the write switch, a program's bounded write authorization, held sessions,
the refusal words, which machine a write lands on, and waiting for a readback
to settle. :doc:`/architecture/safety-chain` describes how they fit together.

.. automodule:: osprey_connectors.posture_env

.. automodule:: osprey_connectors.posture_store

.. automodule:: osprey_connectors.write_authorization

.. automodule:: osprey_connectors.refusals

.. automodule:: osprey_connectors.session_hold

.. automodule:: osprey_connectors.control_context

.. automodule:: osprey_connectors.write_target

.. automodule:: osprey_connectors.settle
