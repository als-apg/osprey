.. _connectors-api-errors-logging:

Errors and Logging
==================

The exception taxonomy callers branch on, and the logger every connector writes
through. The class names and ``reason`` codes in
:mod:`osprey_connectors.errors` are covered by the package's compatibility
guarantee. Every exception the package raises inherits
:class:`~osprey_connectors.errors.ConnectorsError`, wherever it is defined, and
keeps the builtin base its callers read.

.. automodule:: osprey_connectors.errors

.. automodule:: osprey_connectors.logger

.. automodule:: osprey_connectors.log_filter
