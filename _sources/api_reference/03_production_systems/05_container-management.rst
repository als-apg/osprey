====================
Container Management
====================

Container orchestration and deployment system for managing framework and application services.

.. note::
   For implementation guides and examples, see :doc:`../../../developer-guides/05_production-systems/05_container-and-deployment`.

.. currentmodule:: osprey.deployment

Container Orchestration
=======================

.. automodule:: osprey.deployment.container_manager
   :members:
   :show-inheritance:

Runtime Detection
=================

.. automodule:: osprey.deployment.runtime_helper
   :members:
   :show-inheritance:

Configuration Loading
=====================

.. automodule:: osprey.deployment.loader
   :members:
   :show-inheritance:

Module Constants
================

.. currentmodule:: osprey.deployment.container_manager

.. data:: SERVICES_DIR

   Directory name for service configurations.

   :type: str
   :value: "services"

.. data:: TEMPLATE_FILENAME

   Standard filename for Docker Compose templates.

   :type: str
   :value: "docker-compose.yml.j2"

.. data:: COMPOSE_FILE_NAME

   Standard filename for rendered Docker Compose files.

   :type: str
   :value: "docker-compose.yml"

.. seealso::

   :doc:`../../../developer-guides/05_production-systems/05_container-and-deployment`
       Complete guide to container deployment patterns

   :doc:`../01_core_framework/04_configuration_system`
       Framework configuration system integration
