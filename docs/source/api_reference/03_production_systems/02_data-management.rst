===============
Data Management
===============

Data orchestration system for integrating heterogeneous data sources into agent workflows with provider discovery, concurrent retrieval, and LLM-optimized formatting.

.. note::
   For implementation guides and examples, see :doc:`../../../developer-guides/05_production-systems/02_data-source-integration`.

.. currentmodule:: osprey.data_management

Management Classes
==================

.. autoclass:: DataSourceManager
   :members:
   :show-inheritance:

.. autoclass:: DataRetrievalResult
   :members:
   :show-inheritance:

Provider Interfaces
===================

.. autoclass:: DataSourceProvider
   :members:
   :show-inheritance:

.. autoclass:: DataSourceContext
   :members:
   :show-inheritance:

Request Models
==============

.. autoclass:: DataSourceRequest
   :members:
   :show-inheritance:

.. autoclass:: DataSourceRequester
   :members:
   :show-inheritance:

Utility Functions
=================

.. autofunction:: get_data_source_manager

.. autofunction:: create_data_source_request

.. seealso::

   :doc:`../../../developer-guides/05_production-systems/02_data-source-integration`
       Complete implementation guide and examples

   :class:`osprey.services.memory_storage.UserMemoryProvider`
       Example core data source provider implementation
