====================
Container Deployment
====================

**What you'll learn:** How to deploy and manage containerized services using the Osprey Framework's deployment CLI

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Using ``osprey deploy`` CLI for service deployment and orchestration
   - Configuring services in your project's ``config.yml``
   - Managing Jinja2 template rendering with ``docker-compose.yml.j2`` files
   - Understanding build directory management and source code copying
   - Implementing development vs production deployment patterns

   **Prerequisites:** Understanding of Docker/container concepts and :doc:`../../api_reference/01_core_framework/04_configuration_system`

   **Time Investment:** 30-45 minutes for complete understanding

Overview
========

The Osprey Framework provides a container management system for deploying services. The system handles service discovery, Docker Compose template rendering, and container orchestration using Docker or Podman with native compose support.

**Core Features:**

- **Runtime Flexibility**: Automatic detection and use of Docker or Podman
- **Simple Service Configuration**: All services defined in a flat ``services:`` section
- **Template Rendering**: Jinja2 processing of Docker Compose templates with full configuration context
- **Build Management**: Automated build directory creation with source code and configuration copying
- **Container Orchestration**: Docker Compose or Podman Compose integration for multi-service deployment

Architecture
============

The container management system uses a simple, flat directory structure. All services live in your project's ``services/`` directory and are configured the same way.

**Common Services:**

*Framework Infrastructure Services:*
   Core services used across applications:

   - ``jupyter``: Python execution environment with EPICS support
   - ``open-webui``: Web interface for agent interaction
   - ``pipelines``: Processing pipeline infrastructure

*Application-Specific Services:*
   Custom services for your particular application. Examples from the :doc:`../../example-applications/als-assistant`:

   - ``mongo``: MongoDB database for ALS operations data
   - ``pv_finder``: EPICS Process Variable discovery MCP server
   - ``langfuse``: LLM observability and monitoring
   - Any custom services you create

All services are defined in the same ``services:`` section of your ``config.yml``, regardless of whether they're framework infrastructure or application-specific.

Service Configuration
=====================

Services are configured in your project's ``config.yml`` using a simple, flat structure. All services—whether framework infrastructure or application-specific—use the same configuration format.

Basic Configuration Pattern
---------------------------

Here's the standard pattern used by all framework projects:

.. code-block:: yaml

   # config.yml - Your project configuration

   # Define all services in a flat structure
   services:
     # Jupyter - Python execution environment
     jupyter:
       path: ./services/jupyter
       containers:
         read:
           name: jupyter-read
           hostname: jupyter-read
           port_host: 8088
           port_container: 8088
           execution_modes: ["read_only"]
         write:
           name: jupyter-write
           hostname: jupyter-write
           port_host: 8089
           port_container: 8088
           execution_modes: ["write_access"]
       copy_src: true
       render_kernel_templates: true

     # Open WebUI - User interface frontend
     open_webui:
       path: ./services/open-webui
       hostname: localhost
       port_host: 8080
       port_container: 8080

     # Pipelines - Processing infrastructure
     pipelines:
       path: ./services/pipelines
       port_host: 9099
       port_container: 9099
       copy_src: true

     # Application-specific service example (optional)
     # Example: MongoDB for your application data
     mongo:
       name: mongo
       path: ./services/mongo
       port_host: 27017
       port_container: 27017
       copy_src: false

   # Control which services to deploy
   deployed_services:
     - jupyter
     - open_webui
     - pipelines
     # - mongo  # Add your application services as needed

**Key Configuration Options:**

- ``path``: Directory containing the service's Docker Compose template (``docker-compose.yml.j2``)
- ``name``: Container name (defaults to service key if not specified)
- ``hostname``: Container hostname for networking
- ``port_host/port_container``: Port mapping between host and container
- ``copy_src``: Whether to copy ``src/`` directory into the build directory (default: false)
- ``additional_dirs``: Extra directories to copy to build environment (list)
- ``render_kernel_templates``: Process Jupyter kernel templates (for Jupyter services only)
- ``containers``: Multi-container configuration (for services like Jupyter with read/write variants)

Service Directory Organization
------------------------------

Your project organizes services in a flat directory structure:

.. code-block:: text

   your-project/
   ├── services/
   │   ├── docker-compose.yml.j2          # Root network configuration
   │   ├── jupyter/                        # Jupyter service
   │   │   ├── docker-compose.yml.j2
   │   │   ├── Dockerfile
   │   │   ├── custom_start.sh
   │   │   └── python3-epics-readonly/
   │   ├── open-webui/                     # Web UI service
   │   │   ├── docker-compose.yml.j2
   │   │   ├── Dockerfile
   │   │   └── functions/
   │   ├── pipelines/                      # Processing pipeline service
   │   │   ├── docker-compose.yml.j2
   │   │   └── main.py
   │   └── mongo/                          # (Optional) Application services
   │       ├── docker-compose.yml.j2      # E.g., MongoDB for ALS Assistant
   │       └── Dockerfile
   ├── config.yml
   └── src/
       └── your_app/

Each service directory contains:
   - ``docker-compose.yml.j2`` (required): Jinja2 template for Docker Compose
   - ``Dockerfile`` (optional): If the service needs a custom image
   - Other service-specific files (scripts, configs, etc.)

The ``path`` field in your configuration points to these service directories.

Deployment Workflow
===================

The container management system supports both development and production deployment patterns.

.. admonition:: New in v0.7+: Framework CLI Commands
   :class: version-07plus-change

   Service deployment is now managed through the ``osprey deploy`` CLI command.

Development Pattern
-------------------

For development and debugging, start services incrementally:

1. **Configure services incrementally** in ``config.yml``:

   .. code-block:: yaml

      deployed_services:
        - open_webui  # Start with one service

2. **Start in non-detached mode** to monitor logs:

   .. code-block:: bash

      osprey deploy up

3. **Add additional services** after verifying each one works correctly

Production Pattern
------------------

For production deployment:

1. **Configure all required services** in ``config.yml``:

   .. code-block:: yaml

      deployed_services:
        - jupyter
        - open_webui
        - pipelines

2. **Start all services in detached mode**:

   .. code-block:: bash

      osprey deploy up --detached

3. **Verify services are running**:

   .. code-block:: bash

      podman ps

Development Mode
----------------

**Development mode** enables testing Osprey framework changes in containers without publishing to PyPI. When enabled with the ``--dev`` flag, containers use your locally installed Osprey instead of the PyPI version.

**When to Use:**

- Testing framework modifications before release
- Debugging framework internals within container environments
- Contributing to framework development
- Validating framework changes across containerized services

**How It Works:**

The deployment system automatically:

1. Locates your locally installed framework package
2. Copies the framework source code to the build directory
3. Sets the ``DEV_MODE`` environment variable for containers
4. Containers install the local framework copy instead of PyPI version

**Usage:**

.. code-block:: bash

   # Deploy with local framework (foreground)
   osprey deploy up --dev

   # Deploy with local framework (background)
   osprey deploy up --detached --dev

**Verification:**

After deploying in development mode, verify the framework source was copied:

.. code-block:: bash

   # Check for osprey override directory
   ls build/services/jupyter/osprey_override/

   # Check environment variable in container
   podman exec jupyter-read env | grep DEV_MODE

**Fallback Behavior:**

If the local framework cannot be located or copied:

- The system prints a warning message
- Containers fall back to installing from PyPI
- Deployment continues normally

This ensures deployments succeed even if development mode setup fails.

Docker Compose Templates
=========================

Services use Jinja2 templates for Docker Compose file generation. Templates have access to your complete configuration context.

Template Structure
------------------

Templates are located at ``{service_path}/docker-compose.yml.j2``. Here's a complete example:

.. code-block:: yaml

   # services/jupyter/docker-compose.yml.j2
   services:
     jupyter-read:
       container_name: {{services.jupyter.containers.read.name}}
       build:
         context: ./jupyter
         dockerfile: Dockerfile
       restart: unless-stopped
       ports:
         - "{{services.jupyter.containers.read.port_host}}:{{services.jupyter.containers.read.port_container}}"
       volumes:
         - ./jupyter:/jupyter
         - {{project_root}}/{{file_paths.agent_data_dir}}/{{file_paths.executed_python_scripts_dir}}:/home/jovyan/work/executed_scripts/
       environment:
         - NOTEBOOK_DIR=/home/jovyan/work
         - JUPYTER_ENABLE_LAB=yes
         - PYTHONPATH=/jupyter/repo_src
         - TZ={{system.timezone}}
         - HTTP_PROXY=${HTTP_PROXY}
         - NO_PROXY=${NO_PROXY}
       networks:
         - osprey-network

**Template Features:**

- **Configuration Access**: Full configuration available as Jinja2 variables

  - Access services: ``{{services.service_name.option}}``
  - Access file paths: ``{{file_paths.agent_data_dir}}``
  - Access system config: ``{{system.timezone}}``
  - Access project root: ``{{project_root}}``

- **Environment Variables**: Reference host environment via ``${VAR_NAME}``

- **Networking**: All services automatically join the ``osprey-network``

- **Volume Management**: Dynamic volume mounting based on configuration

Template Access Patterns
-------------------------

Common template patterns for accessing configuration:

.. code-block:: yaml

   # Access service configuration
   ports:
     - "{{services.my_service.port_host}}:{{services.my_service.port_container}}"

   # Access nested service config (like Jupyter containers)
   container_name: {{services.jupyter.containers.read.name}}

   # Access file paths
   volumes:
     - {{project_root}}/{{file_paths.agent_data_dir}}:/data

   # Access system configuration
   environment:
     - TZ={{system.timezone}}

   # Access custom configuration
   environment:
     - DATABASE_URL={{database.connection_string}}

Deployment CLI Usage
====================

Deploy services using the ``osprey deploy`` command.

Basic Commands
--------------

.. code-block:: bash

   # Start services in foreground (see logs in terminal)
   osprey deploy up

   # Start services in background (detached mode)
   osprey deploy up --detached

   # Stop services
   osprey deploy down

   # Restart services
   osprey deploy restart

   # Show service status
   osprey deploy status

   # Clean deployment (remove containers and volumes)
   osprey deploy clean

   # Rebuild containers from scratch
   osprey deploy rebuild

Service Status Display
----------------------

The ``status`` command displays detailed information about all deployed services in a formatted table with visual indicators:

.. code-block:: bash

   osprey deploy status

**Example Output:**

.. code-block:: text

   Service Deployment Status

   ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
   ┃ Service       ┃ Project     ┃ Status         ┃ Ports          ┃ Image          ┃
   ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
   │ open-webui    │ weather-ag… │ ● Running      │ 8080→8080      │ ghcr.io/...    │
   │ pipelines     │ weather-ag… │ ● Running      │ 9099→9099      │ local/...      │
   │ jupyter-read  │ weather-ag… │ ● Running      │ 8088→8088      │ local/...      │
   │ jupyter-write │ weather-ag… │ ● Running      │ 8089→8088      │ local/...      │
   │ mongo         │ als-assist… │ ● Stopped      │ 27017→27017    │ mongo:latest   │
   └───────────────┴─────────────┴────────────────┴────────────────┴────────────────┘

.. admonition:: New in v0.8.2: Project Tracking
   :class: version-08plus-change

   The status display now includes project ownership tracking using Docker labels. This enables multi-project deployments where you can identify which project/agent owns each container.

The status display includes:

- **Service**: Container name
- **Project**: Project/agent name (from ``project_name`` in config.yml)
- **Status**: Running (●) or Stopped (●) with visual indicator
- **Ports**: Port mappings (host→container)
- **Image**: Container image used
- **Health**: Health check status (if configured)

**Multi-Project Status:**

You can check status for specific projects using the ``--project`` flag:

.. code-block:: bash

   # Check status of specific project
   osprey deploy status --project ~/projects/weather-agent

   # Check multiple projects
   osprey deploy status --project ~/projects/agent1
   osprey deploy status --project ~/projects/agent2

Container Labels and Filtering
------------------------------

All deployed containers are automatically labeled with project metadata using Docker labels. This enables advanced container management and filtering.

**Container Labels:**

Each container gets three automatic labels:

- ``osprey.project.name`` - Project identifier (from ``project_name`` in config.yml)
- ``osprey.project.root`` - Absolute path to project directory
- ``osprey.deployed.at`` - ISO 8601 timestamp of deployment

**Query Containers by Project:**

.. code-block:: bash

   # List all Osprey containers
   podman ps --filter label=osprey.project.name

   # List containers for specific project
   podman ps --filter label=osprey.project.name=weather-agent

   # Inspect container labels
   podman inspect jupyter-read | grep osprey

**Use Cases:**

- **Multi-project deployments**: Run multiple agent projects simultaneously
- **Container identification**: Quickly identify which project owns containers
- **Automation**: Script container management based on project labels
- **Debugging**: Filter logs and status by project

Command Options
---------------

.. code-block:: bash

   # Use custom configuration file
   osprey deploy up --config my-config.yml

   # Deploy with local framework (development mode)
   osprey deploy up --dev

   # Expose services to all network interfaces (use with caution!)
   osprey deploy up --expose

   # Restart in detached mode
   osprey deploy restart --detached

   # Rebuild and start in detached mode with local framework
   osprey deploy rebuild --detached --dev

Deployment Workflow Details
----------------------------

When you run ``osprey deploy up``, the container manager follows this workflow:

1. **Configuration Loading**: Load and merge configuration files
2. **Service Discovery**: Read ``deployed_services`` list to identify active services
3. **Build Directory Creation**: Create clean build directories for each service
4. **Template Processing**: Render Jinja2 templates with complete configuration context
5. **File Copying**: Copy service files, source code, and additional directories
6. **Configuration Flattening**: Generate self-contained config files for containers
7. **Container Orchestration**: Execute Docker/Podman Compose with generated files

**Generated Build Directory:**

.. code-block:: text

   build/services/
   ├── docker-compose.yml           # Root network configuration
   ├── jupyter/
   │   ├── docker-compose.yml       # Rendered Jupyter service config
   │   ├── Dockerfile
   │   ├── custom_start.sh
   │   ├── python3-epics-readonly/
   │   │   └── kernel.json          # Rendered kernel config
   │   ├── python3-epics-write/
   │   │   └── kernel.json
   │   ├── repo_src/                # Copied source code (if copy_src: true)
   │   │   ├── your_app/
   │   │   └── requirements.txt
   │   └── config.yml               # Flattened configuration
   ├── open-webui/
   │   ├── docker-compose.yml
   │   └── ...
   └── pipelines/
       ├── docker-compose.yml
       ├── repo_src/                # Copied source code
       └── config.yml

Container Networking
====================

Services communicate through container networks using service names as hostnames.

Service Communication
---------------------

Container-to-container communication uses service names:

- **OpenWebUI to Pipelines**: ``http://pipelines:9099``
- **Pipelines to Jupyter**: ``http://jupyter-read:8088``
- **Application to MongoDB** (ALS Assistant): ``mongodb://mongo:27017``
- **Application to PV Finder** (ALS Assistant): ``http://pv-finder:8051``

Host Access from Containers
----------------------------

For containers to access services running on the host (like Ollama):

- Use ``host.docker.internal`` instead of ``localhost`` (Docker on macOS/Windows)
- Use ``host.containers.internal`` on Linux with Podman
- Example: ``http://host.docker.internal:11434`` for Ollama

.. code-block:: yaml

   # In docker-compose.yml.j2
   environment:
     - OLLAMA_BASE_URL=http://host.docker.internal:11434  # Use host.containers.internal for Podman

.. _network-binding-security:

Network Binding and Security
----------------------------

.. versionchanged:: 0.10.7
   Services now bind to ``127.0.0.1`` (localhost only) by default. Previous versions bound to ``0.0.0.0``, exposing services to all network interfaces.

By default, all deployed services bind to **localhost only** (``127.0.0.1``). This means services are only accessible from the machine running the containers, not from external hosts on the network.

.. danger::

   **Versions v0.10.6 and earlier** bound services to ``0.0.0.0`` by default, exposing them to all network interfaces. If you were running an earlier version:

   - Stop all Osprey-related containers immediately
   - Update your Osprey installation
   - Redeploy from scratch so the new secure defaults take effect
   - Rotate any API keys (OpenAI, Anthropic, CBorg, etc.) that were configured as environment variables
   - Consider running a security scan on machines that were deployed on public networks

Exposing Services to the Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you intentionally need services accessible from external hosts (e.g., cloud deployments, shared lab servers), use the ``--expose`` flag:

.. code-block:: bash

   # Expose services to all network interfaces (0.0.0.0)
   osprey deploy up --expose

   # Combine with other flags
   osprey deploy up --detached --expose
   osprey deploy rebuild --expose

.. warning::

   The ``--expose`` flag binds services to ``0.0.0.0``, making them accessible from **any machine on the network**. Only use this when:

   - You have proper authentication configured on all exposed services
   - Your deployment is behind a firewall or in an isolated network
   - You understand the security implications of exposing open ports

   **Do not use this flag in development** unless you specifically need network access from other machines.

You can also set the bind address persistently in your ``config.yml``:

.. code-block:: yaml

   # config.yml
   deployment:
     bind_address: "127.0.0.1"  # Default: localhost only
     # bind_address: "0.0.0.0"  # Caution: exposes to all interfaces

The ``--expose`` CLI flag overrides the ``config.yml`` setting for that invocation.

Docker Compose Deployments Only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``--expose`` flag only applies to ``osprey deploy up`` (Docker/Podman Compose deployments). If you manage containers through other orchestration systems (e.g., Kubernetes, Rancher), network binding is controlled by your orchestrator's configuration instead.

Port Mapping
------------

Services expose ports to the host system for external access:

.. code-block:: yaml

   # Host access through mapped ports
   services:
     open_webui:
       ports:
         - "8080:8080"  # Access at http://localhost:8080

Common port mappings:

- **OpenWebUI**: ``8080:8080`` → ``http://localhost:8080``
- **Jupyter Read**: ``8088:8088`` → ``http://localhost:8088``
- **Jupyter Write**: ``8089:8088`` → ``http://localhost:8089``
- **Pipelines**: ``9099:9099`` → ``http://localhost:9099``

Advanced Configuration
======================

Environment Variables
---------------------

The container manager automatically loads environment variables from ``.env``:

.. code-block:: bash

   # .env file - Services will have access to these variables
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   CBORG_API_KEY=your_key_here
   PROJECT_ROOT=/absolute/path/to/project
   LOCAL_PYTHON_VENV=/path/to/venv/bin/python

These variables are:

1. Available to Docker Compose via ``${VAR_NAME}`` syntax
2. Can be passed to containers via ``environment:`` sections
3. Used by configuration system for variable expansion

Build Directory Customization
------------------------------

Change the build output directory:

.. code-block:: yaml

   # config.yml
   build_dir: "./custom_build"

Source Code Integration
-----------------------

Control whether services get a copy of your ``src/`` directory:

.. code-block:: yaml

   services:
     pipelines:
       copy_src: true  # Copies src/ to build/services/pipelines/repo_src/

This is useful for:

- Pipelines server (needs access to your application code)
- Jupyter containers (needs your application for interactive development)
- Services that execute your application code

Services that don't need source code (databases, UI-only services) should set ``copy_src: false``.

Additional Directories
----------------------

Copy extra directories into service build environments:

.. code-block:: yaml

   services:
     jupyter:
       additional_dirs:
         # Simple directory copy
         - docs

         # Custom source -> destination mapping
         - src: "_agent_data"
           dst: "agent_data"

         # Copy framework source (useful during development)
         - src: ../osprey/src/osprey
           dst: osprey_src/src/osprey

This is commonly used for:

- Documentation that services need
- Data directories
- Configuration files
- Osprey source during development (before osprey is pip-installable)

Build Directory Management
==========================

The container manager creates complete, self-contained build environments for each service.

Build Process
-------------

For each deployed service, the build process:

1. **Clean Build Directory**: Remove existing build directory for clean deployment
2. **Render Docker Compose Template**: Process Jinja2 template with configuration
3. **Copy Service Files**: Copy all files from service directory (except ``.j2`` templates)
4. **Copy Source Code**: If ``copy_src: true``, copy entire ``src/`` directory
5. **Copy Additional Directories**: Copy any directories specified in ``additional_dirs``
6. **Create Flattened Configuration**: Generate self-contained ``config.yml`` for the container
7. **Process Kernel Templates**: If ``render_kernel_templates: true``, render Jupyter kernel configs

**Source Code Handling:**

When ``copy_src: true``:

- Source code is copied to ``build/services/{service}/repo_src/``
- Global ``requirements.txt`` is automatically copied
- Project's ``pyproject.toml`` is copied as ``pyproject_user.toml``
- Containers set ``PYTHONPATH`` to include the copied source

**Configuration Flattening:**

Each service gets a ``config.yml`` with:

- All imports resolved and merged
- Complete, self-contained configuration
- Registry paths adjusted for container environment
- No import directives (everything is flattened)

Working Examples
================

Complete Control Assistant Example
----------------------------------

This example shows a complete working configuration from the Control Assistant tutorial:

.. code-block:: yaml

   # config.yml
   project_name: "my-control-assistant"
   build_dir: ./build
   project_root: /home/user/my-control-assistant
   registry_path: ./src/my_control_assistant/registry.py

   services:
     jupyter:
       path: ./services/jupyter
       containers:
         read:
           name: jupyter-read
           hostname: jupyter-read
           port_host: 8088
           port_container: 8088
           execution_modes: ["read_only"]
         write:
           name: jupyter-write
           hostname: jupyter-write
           port_host: 8089
           port_container: 8088
           execution_modes: ["write_access"]
       copy_src: true
       render_kernel_templates: true

     open_webui:
       path: ./services/open-webui
       hostname: localhost
       port_host: 8080
       port_container: 8080

     pipelines:
       path: ./services/pipelines
       port_host: 9099
       port_container: 9099
       copy_src: true

   deployed_services:
     - jupyter
     - open_webui
     - pipelines

   system:
     timezone: ${TZ:-America/Los_Angeles}

   file_paths:
     agent_data_dir: _agent_data
     executed_python_scripts_dir: executed_scripts

Deploy this configuration:

.. code-block:: bash

   osprey deploy up --detached

   # Services available:
   # - OpenWebUI: http://localhost:8080
   # - Jupyter Read: http://localhost:8088
   # - Jupyter Write: http://localhost:8089
   # - Pipelines: http://localhost:9099

Troubleshooting
===============

.. tab-set::

   .. tab-item:: Common Issues

      **Services fail to start:**

      1. Check individual service logs: ``podman logs <container_name>``
      2. Verify configuration syntax in ``config.yml``
      3. Ensure required environment variables are set in ``.env``
      4. Try starting services individually to isolate issues
      5. Check that service paths exist and contain ``docker-compose.yml.j2``

      **Port conflicts:**

      1. Check for processes using required ports: ``lsof -i :8080``
      2. Update port mappings in service configurations
      3. Ensure no other containers are using the same ports
      4. Verify ``deployed_services`` doesn't have duplicate services

      **Container networking issues:**

      1. Verify service names match configuration
      2. Use container network names (e.g., ``pipelines``) not ``localhost``
      3. Check firewall settings if accessing from external systems
      4. Ensure all services use the ``osprey-network``

      **Template rendering errors:**

      1. Verify Jinja2 syntax in template files (``{{variable}}`` not ``{variable}``)
      2. Check that configuration values exist before accessing them
      3. Review template paths in error messages
      4. Inspect generated files in ``build/`` directory

      **Service not found in configuration:**

      - Verify service is defined in ``services:`` section
      - Check service name matches between ``services:`` and ``deployed_services:``
      - Ensure ``deployed_services`` list uses correct service names

      **Template file not found:**

      - Verify ``docker-compose.yml.j2`` exists at the ``path`` location
      - Check that the service ``path`` is correct relative to your project root
      - Ensure you haven't accidentally specified a directory that doesn't exist

      **Copy source failures:**

      - Verify ``src/`` directory exists if ``copy_src: true``
      - Check permissions on source directories
      - Ensure additional_dirs paths exist

      **Development mode issues:**

      - Verify osprey is installed in your active virtual environment
      - Check that ``osprey_override/`` directory exists in build after deployment
      - Confirm ``DEV_MODE=true`` is set in container environment
      - If osprey source not found, containers will fall back to PyPI version
      - Review console output for osprey copy warnings during deployment

   .. tab-item:: Debugging Commands

      **List running containers:**

      .. code-block:: bash

         podman ps

      **View container logs:**

      .. code-block:: bash

         podman logs <container_name>
         podman logs -f <container_name>  # Follow logs in real-time

      **Inspect container configuration:**

      .. code-block:: bash

         podman inspect <container_name>

      **Network inspection:**

      .. code-block:: bash

         podman network ls
         podman network inspect osprey-network

      **Check generated configuration:**

      .. code-block:: bash

         # Review rendered Docker Compose files
         cat build/services/jupyter/docker-compose.yml

         # Check flattened configuration
         cat build/services/pipelines/config.yml

      **Check for port conflicts:**

      .. code-block:: bash

         lsof -i :8080  # Check specific port
         netstat -tulpn | grep :8080  # Alternative method

      **Test network connectivity:**

      .. code-block:: bash

         # Test container-to-container communication
         podman exec pipelines ping jupyter-read
         podman exec pipelines curl http://open-webui:8080

      **Rebuild after configuration changes:**

      .. code-block:: bash

         # Full rebuild (safest after config changes)
         osprey deploy clean
         osprey deploy up --detached

         # Or use rebuild command (clean + up in one step)
         osprey deploy rebuild --detached

      **Verify development mode:**

      .. code-block:: bash

         # Check if osprey override was copied
         ls -la build/services/jupyter/osprey_override/

         # Verify DEV_MODE environment variable in container
         podman exec jupyter-read env | grep DEV_MODE

         # Check osprey installation in container
         podman exec jupyter-read pip show osprey-framework

   .. tab-item:: Quick Reference

      **Common Commands:**

      .. code-block:: bash

         # Start services (localhost only by default)
         osprey deploy up
         osprey deploy up --detached

         # Start services exposed to network (use with caution!)
         osprey deploy up --expose

         # Stop services
         osprey deploy down

         # Check status
         osprey deploy status
         podman ps

         # View logs
         podman logs <container_name>
         podman logs -f <container_name>

         # Clean restart
         osprey deploy clean
         osprey deploy up --detached

      **Common Service Names:**

      - ``jupyter-read`` - Jupyter read-only container
      - ``jupyter-write`` - Jupyter write-access container
      - ``open-webui`` - Web interface
      - ``pipelines`` - Processing pipeline
      - ``mongo`` - MongoDB (ALS Assistant)
      - ``pv-finder`` - PV Finder MCP (ALS Assistant)

      **Common Ports:**

      - ``8080`` - OpenWebUI
      - ``8088`` - Jupyter (read-only)
      - ``8089`` - Jupyter (write-access)
      - ``9099`` - Pipelines
      - ``27017`` - MongoDB
      - ``8051`` - PV Finder

   .. tab-item:: Best Practices

      **Development:**

      - **Start minimal**: Begin with one service, verify it works, then add more
      - **Use foreground mode**: Run ``osprey deploy up`` (not detached) during development to see logs
      - **Test services individually**: Deploy services one at a time to isolate issues
      - **Keep build directory in .gitignore**: Build artifacts shouldn't be version controlled
      - **Use meaningful container names**: Makes logs and debugging easier
      - **Use development mode for framework changes**: Run ``osprey deploy up --dev`` when testing osprey modifications
      - **Verify development mode**: Check console output for osprey copy messages

      **Production:**

      - **Use detached mode**: Run ``osprey deploy up --detached`` for production
      - **Keep services on localhost by default**: Only use ``--expose`` when you need network access and have authentication configured
      - **Monitor container resources**: Use ``podman stats`` to watch resource usage
      - **Implement health checks**: Add health check configurations to your docker-compose templates
      - **Plan restart policies**: Use ``restart: unless-stopped`` in production templates
      - **Regular backups**: Back up data volumes for database services
      - **Document deployment**: Keep notes on deployment procedures and configurations

      **Configuration:**

      - **Keep secrets in .env**: Never commit API keys or passwords to version control
      - **Use absolute paths**: Set ``project_root`` as absolute path in config
      - **Test changes incrementally**: Test configuration changes in development first
      - **Version control configs**: Track ``config.yml`` and templates in git
      - **Document custom modifications**: Comment any non-standard template changes
      - **Validate before deploying**: Check YAML syntax before running deploy commands

      **Template Development:**

      - **Test templates incrementally**: Verify each configuration value exists
      - **Use descriptive variable names**: Clear naming makes templates maintainable
      - **Add comments**: Document non-obvious template logic
      - **Check rendered output**: Review files in ``build/`` after changes
      - **Handle missing values gracefully**: Use Jinja2 defaults: ``{{value|default('fallback')}}``

.. seealso::

   :doc:`../../api_reference/01_core_framework/04_configuration_system`
       Advanced configuration patterns and variable expansion

   :doc:`../../api_reference/03_production_systems/05_container-management`
       Container management API reference
