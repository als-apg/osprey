Installation & Setup
====================

What You'll Learn
~~~~~~~~~~~~~~~~~

This installation guide covers the complete framework setup process:

* **Installing Prerequisites** - Python 3.11+, Node.js 18+, container runtime
* **Installing Claude Code CLI** - The AI orchestration layer
* **Installing the Osprey Framework** - Via PyPI or from source
* **Project Creation** - Generating a new project from templates
* **Configuration** - Setting up ``config.yml`` and environment variables
* **Launching the Agent** - Starting a Claude Code session
* **(Optional) Deploying Services** - Containerized services for enhanced capabilities

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **System Requirements:**

   - **Operating System:** Linux, macOS, or Windows with WSL2
   - **Admin/sudo access:** Required for installing container runtime and Python
   - **Internet connection:** For downloading packages and container images
   - **Disk space:** At least 5GB free for containers and dependencies

   **What You'll Install:**

   - Python 3.11+ (programming language)
   - Node.js 18+ (required for Claude Code CLI)
   - `Claude Code CLI <https://docs.anthropic.com/en/docs/claude-code/overview>`_ (AI orchestration)
   - `uv <https://docs.astral.sh/uv/>`_ (package manager)
   - Docker Desktop 4.0+ OR Podman 4.0+ (container runtime, optional)
   - Osprey Framework (via PyPI or from source)

   **Time estimate:** 30-60 minutes for complete setup

Installation Steps
~~~~~~~~~~~~~~~~~~

**Install Container Runtime (Optional)**

.. note::
   A container runtime is only required if you plan to deploy containerized services
   (Jupyter, simulation IOCs, databases). The core agent workflow does not require containers.

The framework supports both Docker and Podman. Install **either one** (not both required):

.. tab-set::

    .. tab-item:: Docker Desktop (Recommended for macOS/Windows)

        **Installation:**

        `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ is the most widely used container platform, providing an integrated experience with native compose support.

        Download and install Docker Desktop 4.0+ from the `official Docker installation guide <https://docs.docker.com/get-started/get-docker/>`_.

        **Verification:**

        After installation, verify Docker is working:

        .. code-block:: bash

           docker --version
           docker compose version
           docker run hello-world

        Docker Desktop handles the VM setup automatically on macOS/Windows - no additional configuration needed.

    .. tab-item:: Podman (Recommended for Linux/Security-focused deployments)

        **Installation:**

        `Podman <https://podman.io/>`_ is a daemonless container engine that provides enhanced security through rootless operation. Unlike Docker, Podman doesn't require a privileged daemon running as root, offering better privilege separation and a reduced attack surface.

        Install Podman 4.0+ from the `official Podman installation guide <https://podman.io/docs/installation>`_.

        **Verification:**

        After installation, verify Podman is working:

        .. code-block:: bash

           podman --version
           podman run hello-world

        **Podman Machine Setup (macOS/Windows only):**

        On macOS/Windows, initialize and start the Podman machine:

        .. code-block:: bash

           podman machine init
           podman machine start

        **Note:** Linux users can skip this step as Podman runs natively on Linux.

**Runtime Selection:**

The framework automatically detects which runtime is available. To explicitly choose a runtime:

- **Via configuration:** Set ``container_runtime: docker`` or ``container_runtime: podman`` in ``config.yml``
- **Via environment variable:** ``export CONTAINER_RUNTIME=docker`` or ``export CONTAINER_RUNTIME=podman``

If both are installed, Docker is preferred by default.

**Install Claude Code CLI**

Claude Code is the AI orchestration layer that powers Osprey's agent workflow.

**Node.js 18+ Prerequisite:**

Claude Code requires `Node.js 18+ <https://nodejs.org/>`_. Verify or install:

.. code-block:: bash

   # Check if Node.js is installed
   node --version

   # Install via Homebrew (macOS)
   brew install node

   # Or download from https://nodejs.org/

**Install Claude Code:**

.. code-block:: bash

   npm install -g @anthropic-ai/claude-code

**Verification:**

.. code-block:: bash

   claude --version

.. note::
   Claude Code requires an ``ANTHROPIC_API_KEY`` for the orchestration layer. You'll configure this
   in the :ref:`environment variables <environment-variables>` section below.

**Environment Setup**

**Python 3.11+ Requirement**

This framework requires `Python 3.11+ <https://www.python.org/downloads/>`_. Verify you have the correct version:

.. code-block:: bash

   python3.11 --version

**Installing the Framework**

.. tab-set::

    .. tab-item:: PyPI (Recommended)

        Install the latest release from PyPI:

        .. code-block:: bash

           # Using uv (recommended)
           uv pip install osprey-framework

           # Or using pip
           pip install osprey-framework

    .. tab-item:: From Source (Development)

        Clone and install for development:

        .. code-block:: bash

           git clone https://github.com/als-apg/osprey.git
           cd osprey
           uv sync

        This creates a ``.venv`` virtual environment automatically and installs all dependencies.

**Core Dependencies** (installed automatically):

* **AI Orchestration**: `Claude Code SDK <https://docs.anthropic.com/>`_ (``claude-agent-sdk``), `LiteLLM <https://docs.litellm.ai/>`_
* **AI Providers**: 100+ providers via LiteLLM including `OpenAI <https://openai.com/>`_, `Anthropic <https://www.anthropic.com/>`_, `Google <https://ai.google.dev/>`_, `Ollama <https://ollama.com/>`_
* **MCP Servers**: `FastMCP <https://github.com/jlowin/fastmcp>`_ for Model Context Protocol server development
* **CLI & UI**: `Rich <https://rich.readthedocs.io/>`_, `Click <https://click.palletsprojects.com/>`_
* **Configuration**: PyYAML, Jinja2, python-dotenv
* **Data Processing**: NumPy, Pandas, Matplotlib
* **Networking**: requests, websocket-client, aiohttp

**Optional Dependencies** (install with extras):

* **[docs]**: Sphinx and documentation tools -- ``uv sync --extra docs``
* **[dev]**: pytest, ruff, mypy, and development tools -- ``uv sync --extra dev``
* **[ariel]**: PostgreSQL with pgvector for logbook search -- ``uv sync --extra ariel``
* **[ariel-proxy]**: SOCKS tunnel support for HTTP ingestion -- ``uv sync --extra ariel-proxy``
* **[sheets]**: Google Sheets channel-finder backend -- ``uv sync --extra sheets``
* **[screen-capture-linux]**: X11 screen capture support -- ``uv sync --extra screen-capture-linux``
* **[all]**: Everything -- ``uv sync --extra all``

**Creating a New Project**

Once the framework is installed, you can create a new project using either the interactive menu or direct CLI commands:

**Method 1: Interactive Mode (Recommended for New Users)**

Simply run ``osprey`` without any arguments to launch the interactive menu:

.. code-block:: bash

   osprey

The interactive menu will:

1. Guide you through template selection with descriptions
2. Help you choose an AI provider (Cborg, OpenAI, Anthropic, etc.)
3. Let you select from available models
4. Automatically detect API keys from your environment
5. Create a ready-to-use project with smart defaults

**Method 2: Direct CLI Command**

For automation or if you prefer direct commands, use ``osprey init``:

.. code-block:: bash

   # Create a project with the default template
   osprey init my-assistant

   # Or specify a template explicitly
   osprey init my-assistant --template control_assistant

   # Navigate to your project
   cd my-assistant

Available templates:

* ``hello_world`` - Minimal tutorial with one MCP server and mock control system
* ``control_assistant`` (default) - Control system integration with channel finder, archiver, and logbook
* ``lattice_design`` - Accelerator lattice physics with pyAT integration

Both methods create identical project structures - choose whichever fits your workflow.

.. dropdown:: **Understand Your Project Structure**
   :color: info
   :icon: file-directory

   The generated project includes all components needed for a complete AI agent application:

   * **Claude Code configuration** (``.claude/``) - Hooks, rules, agents, skills, and output styles
   * **MCP server configuration** (``.mcp.json``) - Defines which MCP servers Claude Code can access
   * **Agent instructions** (``CLAUDE.md``) - Project-specific instructions for the AI agent
   * **Service configurations** (``services/``) - Container configs for deployed services
   * **Template data** (``data/``) - Channel databases, logbooks, benchmarks (template-specific)
   * **Agent workspace** (``_agent_data/``) - Working directory for agent-generated artifacts
   * **Configuration file** (``config.yml``) - Self-contained application settings
   * **Environment template** (``.env.example``) - API keys and secrets template

   **Want to understand what each component does?**

   The :doc:`Hello World Tutorial <hello-world-tutorial>` provides a detailed walkthrough of this structure - explaining what each file does, how components work together, and how to customize them for your needs.

.. _Configuration:

Configuration & Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generated project includes both a ``config.yml`` configuration file and a ``.env.example`` template for environment variables. Configure both for your environment:

.. tab-set::

    .. tab-item:: config.yml

        **Update config.yml**

        The generated project includes a complete ``config.yml`` file in the project root. All framework settings are pre-configured with sensible defaults. Modify the following settings as needed:

        **1. Project Root Path**

        The ``project_root`` in ``config.yml`` is automatically set to your project directory during ``osprey init``. For advanced use cases (multi-environment deployments), you can override this by setting ``PROJECT_ROOT`` in your ``.env`` file.

        **2. Ollama Base URL**

        Set the base URL for `Ollama <https://ollama.com/>`_:

        - For direct host access: ``localhost:11434``
        - For container-based agents: ``host.containers.internal:11434``

        **3. Deployed Services**

        Ensure the following are uncommented in ``deployed_services``:

        - ``jupyter`` - Environment for editing and running generated code

        **4. API Provider URLs**

        If using `CBorg <https://cborg.lbl.gov/>`_ (LBNL internal only), set the API URL:

        - Global: ``https://api.cborg.lbl.gov/v1``
        - Local: ``https://api-local.cborg.lbl.gov/v1`` (requires local network)

        In ``config.yml``: ``api: providers:cborg:base_url: https://api-local.cborg.lbl.gov/v1``

        **5. Model Providers (External Users)**

        If you don't have CBorg access, configure alternative providers in ``config.yml``. Update the ``provider`` fields under the ``models`` section to use ``openai``, ``anthropic``, ``ollama``, etc. Set corresponding API keys in your ``.env`` file.

        .. dropdown:: Need Support for Additional Providers?
           :color: info
           :icon: people

           We're happy to implement support for additional model providers beyond those currently supported. Many research institutions and national laboratories now operate their own AI/LM services similar to LBNL's CBorg system. If you need integration with your institution's internal AI services or other providers, please reach out to us. We can work with you to add native support for your preferred provider.

    .. tab-item:: Environment Variables

        .. _environment-variables:

        **Environment Variables**

        The framework uses environment variables for **secrets** (API keys) and **machine-specific settings** (file paths, network configuration). This allows you to run the same project on different machines - your laptop, a lab server, or a control room computer - without changing your code or ``config.yml``.

        The generated project includes a ``.env.example`` template with all supported variables.

        **When to use .env vs config.yml:**

        - **Environment variables (.env):** Secrets, absolute paths, proxy settings that change per machine
        - **Configuration file (config.yml):** Application behavior, model choices, settings that stay the same

        **Automatic Setup (if API keys are in your environment):**

        If you already have API keys exported in your shell:

        .. code-block:: bash

           # These are already in your shell environment
           export ANTHROPIC_API_KEY=sk-ant-...
           export CBORG_API_KEY=...

           # When you create a project, the framework automatically creates .env with them!
           osprey init my-agent
           # or use interactive mode: osprey

        The framework will create a ``.env`` file automatically with your detected keys.

        **Manual Setup (if keys are not in environment):**

        If API keys are not in your environment, set them up manually:

        .. code-block:: bash

           # Copy the template
           cp .env.example .env

           # Edit with your values
           nano .env  # or your preferred editor

        **Required Variables:**

        **API Keys** (at least one required):

        ``OPENAI_API_KEY``
           OpenAI API key for GPT models.

           Get from: https://platform.openai.com/api-keys

        ``ANTHROPIC_API_KEY``
           Anthropic API key for Claude models. Also required for Claude Code orchestration.

           Get from: https://console.anthropic.com/

        ``GOOGLE_API_KEY``
           Google API key for Gemini models.

           Get from: https://makersuite.google.com/app/apikey

        ``CBORG_API_KEY``
           CBorg API key (LBNL internal only).

           Get from: https://cborg.lbl.gov/

        **Optional Variables:**

        ``OSPREY_PROJECT``
           Default project directory for CLI commands. Allows working with specific projects without changing directories.

           Example: ``/home/user/projects/my-agent``

           See :doc:`/cli-reference/index` for multi-project workflow examples.

        ``TZ``
           Timezone for timestamp formatting.

           Default: ``America/Los_Angeles``

           Example: ``UTC``, ``Europe/London``, ``Asia/Tokyo``

        ``CONFIG_FILE``
           Override config file location (advanced usage).

           Default: ``config.yml`` in current directory

        **Optional Variables** (for advanced use cases):

        ``PROJECT_ROOT``
           Override the ``project_root`` value from ``config.yml``. Useful for multi-environment deployments or if you move your project directory.

           Example: ``/home/user/my-agent``

        **Network Settings** (for restricted environments):

        ``HTTP_PROXY``
           HTTP proxy server URL. Useful in production environments with firewall restrictions (labs, control rooms, corporate networks).

           Example: ``http://proxy.company.com:8080``

        ``NO_PROXY``
           Comma-separated list of hosts to exclude from proxy.

           Example: ``localhost,127.0.0.1,.internal``

.. note::
   **Security & Multi-Machine Workflow:**

   - The framework automatically loads ``.env`` from your project root
   - **Keep ``.env`` in ``.gitignore``** to protect secrets from version control
   - Environment variables in ``config.yml`` are resolved using ``${VARIABLE_NAME}`` syntax
   - **Best practice:** Keep one ``config.yml`` (in git), but different ``.env`` files per machine (NOT in git)
   - Example: ``.env.laptop``, ``.env.controlroom``, ``.env.server`` - copy the appropriate one to ``.env`` when running on that machine

Documentation
~~~~~~~~~~~~~

**Compile Documentation (Optional)**

If you want to build and serve the documentation locally:

.. code-block:: bash

   # Install documentation dependencies
   uv sync --extra docs

   # Build and serve documentation with auto-reload
   cd docs && uv run sphinx-autobuild source build

Once running, you can view the documentation at http://localhost:8000

Deploying Services (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Services enhance the agent's capabilities but aren't required to get started. The core agent workflow works without any deployed services.

Once you have configured your project, you can optionally start containerized services. The framework CLI orchestrates all services using containers.

**Start Services**

For detailed information about all deployment options, see :doc:`/how-to/deploy-project` or the :doc:`CLI reference </cli-reference/index>`.

.. tab-set::

    .. tab-item:: Development Mode (Recommended for starters)

        **For initial setup and debugging**, start services one by one in non-detached mode:

        1. Comment out all services except one in your ``config.yml`` under ``deployed_services``
        2. Start the first service:

        .. code-block:: bash

           osprey deploy up

        3. Monitor the logs to ensure it starts correctly
        4. Once stable, stop with ``Ctrl+C`` and uncomment the next service
        5. Repeat until all services are working

        This approach helps identify issues early and ensures each service is properly configured before proceeding.

    .. tab-item:: Production Mode

        **Once all services are tested individually**, start everything together in detached mode:

        .. code-block:: bash

           osprey deploy up --detached

        This runs all services in the background, suitable for production deployments where you don't need to monitor individual service logs.

**Other Deployment Commands**

.. code-block:: bash

   osprey deploy down      # Stop all services
   osprey deploy restart   # Restart services
   osprey deploy status    # Show service status
   osprey deploy clean     # Clean deployment
   osprey deploy rebuild   # Rebuild containers

**Verify Services are Running**

Check that services are running properly:

.. code-block:: bash

   # If using Docker
   docker ps

   # If using Podman
   podman ps

Start the Agent
~~~~~~~~~~~~~~~~

With your project created and configured, launch a Claude Code agent session:

.. tab-set::

    .. tab-item:: claude (Direct)

        Run ``claude`` from your project root. Claude Code reads ``.mcp.json`` automatically and
        connects to all configured MCP servers:

        .. code-block:: bash

           cd my-assistant
           claude

        Requires ``ANTHROPIC_API_KEY`` in your environment.

    .. tab-item:: osprey claude chat (Managed)

        The managed launch regenerates Claude Code artifacts and injects provider environment
        variables, which is useful when using non-Anthropic providers:

        .. code-block:: bash

           osprey claude chat

        **Options:**

        - ``--resume`` - Resume a previous conversation
        - ``--print`` - Print-mode (non-interactive output)
        - ``--effort <level>`` - Set effort level (``low``, ``medium``, ``high``, ``max``)

    .. tab-item:: osprey web (Browser)

        Launch a browser-based interface with an integrated terminal:

        .. code-block:: bash

           osprey web

        Opens at ``http://localhost:8087`` by default.

        **Options:**

        - ``--port <port>`` - Custom port
        - ``--host <host>`` - Custom host binding
        - ``--detach`` - Run in the background

.. note::
   On first run, Claude Code will display a trust dialog asking you to review and accept
   the project's MCP server configuration and permissions.

.. dropdown:: **Understanding MCP Servers**
   :color: info
   :icon: server

   Osprey exposes its capabilities to Claude Code via **MCP (Model Context Protocol) servers**.
   The ``.mcp.json`` file in your project root configures which servers are available.

   **Available servers:**

   * ``workspace`` - File management, project context, and workspace operations
   * ``python_executor`` - Sandboxed Python code execution
   * ``control_system`` - Read/write channels on the control system (with human approval for writes)
   * ``ariel`` - Logbook search and knowledge retrieval
   * ``export`` - Data export and artifact generation
   * ``channel_finder_hierarchical`` / ``channel_finder_in_context`` / ``channel_finder_middle_layer`` / ``direct_channel_finder`` - Channel discovery (variant depends on template)

   MCP servers start automatically when Claude Code launches — no manual management required.
   The servers run as subprocesses managed by Claude Code and shut down when the session ends.

Troubleshooting
~~~~~~~~~~~~~~~

**Common Issues:**

- **``claude`` not found**: Install Claude Code with ``npm install -g @anthropic-ai/claude-code``
- **MCP connection failed**: Ensure you're running from the project root where ``.mcp.json`` lives
- **Provider auth error**: Export the required API key (e.g., ``export ANTHROPIC_API_KEY=sk-ant-...``)
- If you encounter connection issues with Ollama, ensure you're using ``host.containers.internal`` instead of ``localhost`` when connecting from containers
- Verify that all required services are uncommented in ``config.yml``
- Check that API keys are properly set in the ``.env`` file
- Ensure container runtime is running (Docker Desktop or Podman machine on macOS/Windows)
- If containers fail to start, check logs with: ``docker logs <container_name>`` or ``podman logs <container_name>``

**Verification Steps:**

1. Check framework version: ``osprey --version``
2. Check Python version: ``python --version`` (should be 3.11.x or higher)
3. Check Claude Code: ``claude --version``
4. Check container runtime version: ``docker --version`` or ``podman --version`` (should be 4.0.0+)
5. Test core framework imports: ``python -c "import osprey; print('Osprey installed successfully')"``
6. Check service status (if deployed): ``docker ps`` or ``podman ps``

**Common Installation Issues:**

- **Python version mismatch**: Ensure you're using Python 3.11+ with ``python3.11 --version``
- **Node.js missing**: Claude Code requires Node.js 18+. Install from `nodejs.org <https://nodejs.org/>`_
- **Package conflicts**: If you get dependency conflicts, try ``uv sync`` in a fresh clone
- **Missing dependencies**: Running ``uv sync`` should install everything needed

Next Steps
~~~~~~~~~~

.. seealso::

   :doc:`hello-world-tutorial`
      Build your first agent with a guided tutorial

   :doc:`control-assistant`
      Production control system assistant with channel finding and comprehensive tooling

   :doc:`/cli-reference/index`
      Complete CLI command reference
