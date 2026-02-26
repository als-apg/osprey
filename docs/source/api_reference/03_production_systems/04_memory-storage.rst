==============
Memory Storage
==============

User memory infrastructure with persistent storage, data source integration, and structured memory operations.

.. note::
   For implementation guides and examples, see :doc:`../../../developer-guides/05_production-systems/04_memory-storage-service`.

Storage Management
==================

.. currentmodule:: osprey.services.memory_storage.storage_manager

.. autoclass:: MemoryStorageManager
   :members:
   :show-inheritance:

Data Source Integration
=======================

.. currentmodule:: osprey.services.memory_storage.memory_provider

.. autoclass:: UserMemoryProvider
   :members:
   :show-inheritance:

Data Models
===========

.. currentmodule:: osprey.services.memory_storage.models

.. autoclass:: MemoryContent
   :members:
   :show-inheritance:

Utility Functions
=================

.. currentmodule:: osprey.services.memory_storage.storage_manager

.. autofunction:: get_memory_storage_manager

.. seealso::

   :doc:`../../../developer-guides/05_production-systems/04_memory-storage-service`
       Complete implementation guide and examples

   :class:`osprey.data_management.DataSourceProvider`
       Base provider interface implemented by UserMemoryProvider
