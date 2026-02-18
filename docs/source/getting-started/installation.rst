Installation & Setup
====================

What You'll Learn
~~~~~~~~~~~~~~~~~

This installation guide covers the complete framework setup process:

* **Installing Container Runtime** - Docker Desktop or Podman for containerized services
* **Python 3.11+ Setup** - Virtual environment configuration
* **Framework Installation** - Installing the pip package with all dependencies
* **Project Creation** - Generating a new project from templates
* **Configuration** - Setting up ``config.yml`` and environment variables
* **Service Deployment** - Starting containerized services (Jupyter, OpenWebUI, Pipelines)
* **OpenWebUI Configuration** - Chat interface setup and customization

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **System Requirements:**

   - **Operating System:** Linux, macOS, or Windows with WSL2
   - **Admin/sudo access:** Required for installing container runtime and Python
   - **Internet connection:** For downloading packages and container images
   - **Disk space:** At least 5GB free for containers and dependencies

   **What You'll Install:**

   - Docker Desktop 4.0+ OR Podman 4.0+ (container runtime)
   - Python 3.11+ (programming language)
   - Osprey Framework (pip package)

   **Time estimate:** 30-60 minutes for complete setup

Installation Steps
~~~~~~~~~~~~~~~~~~

**Install Container Runtime**

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

**Environment Setup**

**Python 3.11+ Requirement**

This framework requires `Python 3.11+ <https://www.python.org/downloads/>`_. Verify you have the correct version:

.. code-block:: bash

   python3.11 --version

**Virtual Environment Setup**

To avoid conflicts with your system Python packages, create a virtual environment with Python 3.11+:

.. tab-set::

    .. tab-item:: uv (Recommended)

        `uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager that handles virtual environments automatically:

        .. code-block:: bash

           # uv creates and manages the .venv automatically
           uv venv

    .. tab-item:: pip (Traditional)

        .. code-block:: bash

           python3.11 -m venv venv
           source venv/bin/activate  # On Windows: venv\Scripts\activate

**Installing the Framework**

After setting up your virtual environment, install the framework package:

.. tab-set::

    .. tab-item:: uv (Recommended)

        .. code-block:: bash

           uv pip install osprey-framework

    .. tab-item:: pip (Traditional)

        .. code-block:: bash

           pip install --upgrade pip
           pip install osprey-framework

.. admonition:: New in v0.7+: Pip-Installable Architecture
   :class: version-07plus-change

   The framework is now distributed as a pip-installable package with modular dependencies. You no longer need to clone the repository or manage ``requirements.txt`` files manually.

   **Core Dependencies** (always installed):

   * **Core Framework**: `LangGraph <https://www.langchain.com/langgraph>`_, `LangChain <https://www.langchain.com/>`_, `LiteLLM <https://docs.litellm.ai/>`_
   * **AI Providers**: 100+ providers via LiteLLM including `OpenAI <https://openai.com/>`_, `Anthropic <https://www.anthropic.com/>`_, `Google <https://ai.google.dev/>`_, `Ollama <https://ollama.com/>`_
   * **CLI & UI**: `Rich <https://rich.readthedocs.io/>`_, `Click <https://click.palletsprojects.com/>`_, `prompt_toolkit <https://python-prompt-toolkit.readthedocs.io/>`_
   * **Container Runtime**: Docker Desktop 4.0+ or Podman 4.0+ (installed separately via system package managers)
   * **Configuration**: PyYAML, Jinja2, python-dotenv
   * **Networking**: requests, websocket-client

   **Optional Dependencies** (install with extras):

   * **[scientific]**: SciPy, Seaborn, Scikit-learn, ipywidgets *(for advanced data analysis)*
   * **[docs]**: Sphinx and documentation tools
   * **[dev]**: pytest, ruff, mypy, and development tools

   .. note::
      The following packages are now included in the core installation:

      * **Claude Agent SDK**: Advanced code generation with multi-turn agentic reasoning
      * **NumPy, Pandas, Matplotlib**: Required for data handling and visualization in Python execution capability

   **Installation Examples:**

   .. code-block:: bash

      # Recommended: Core + scientific computing for advanced data analysis
      pip install osprey-framework[scientific]

      # Minimal installation (includes Claude Code SDK)
      pip install osprey-framework

      # Core + documentation
      pip install osprey-framework[docs]

      # Everything (includes docs, dev tools, etc.)
      pip install osprey-framework[all]

**Creating a New Project**

.. admonition:: New in v0.7.7: Interactive Project Creation
   :class: version-07plus-change

   The framework now includes an interactive menu that guides you through project creation with helpful prompts and automatic API key detection. This is the recommended method for new users.

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

   # Create a project with the hello_world_weather template
   osprey init my-weather-agent --template hello_world_weather

   # Navigate to your project
   cd my-weather-agent

Available templates:

* ``minimal`` - Basic skeleton for starting from scratch
* ``hello_world_weather`` - Simple weather agent (recommended for learning)
* ``control_assistant`` - Production control system integration template

Both methods create identical project structures - choose whichever fits your workflow.

.. dropdown:: **Understand Your Project Structure**
   :color: info
   :icon: file-directory

   The generated project includes all components needed for a complete AI agent application:

   * **Application code** (``src/``) - Your capabilities, context classes, and business logic
   * **Service configurations** (``services/``) - Container configs for Jupyter, OpenWebUI, and Pipelines
   * **Configuration file** (``config.yml``) - Self-contained application settings
   * **Environment template** (``.env.example``) - API keys and secrets template

   **Project Structure Example** (using ``hello_world_weather`` template):

   .. code-block::

      my-weather-agent/
      ├── src/
      │   └── my_weather_agent/
      │       ├── __init__.py
      │       ├── mock_weather_api.py      # Data source
      │       ├── context_classes.py       # Data models
      │       ├── registry.py              # Component registration
      │       └── capabilities/
      │           ├── __init__.py
      │           └── current_weather.py   # Business logic
      ├── services/                        # Container configurations
      ├── config.yml                       # Application settings
      └── .env.example                     # API key template

   **Want to understand what each component does?**

   The :doc:`Hello World Tutorial <hello-world-tutorial>` provides a detailed walkthrough of this structure - explaining what each file does, how components work together, and how to customize them for your needs. If you want to understand the architecture before continuing with deployment, jump to the tutorial now.

.. _Configuration:

Configuration & Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generated project includes both a ``config.yml`` configuration file and a ``.env.example`` template for environment variables. Configure both for your environment:

.. tab-set::

    .. tab-item:: config.yml

        **Update config.yml**

        The generated project includes a complete ``config.yml`` file in the project root. All framework settings are pre-configured with sensible defaults. Modify the following settings as needed:

        **1. Project Root Path**

        The ``project_root`` in ``config.yml`` is automatically set to your project directory during ``framework init``. For advanced use cases (multi-environment deployments), you can override this by setting ``PROJECT_ROOT`` in your ``.env`` file.

        **2. Ollama Base URL**

        Set the base URL for `Ollama <https://ollama.com/>`_:

        - For direct host access: ``localhost:11434``
        - For container-based agents (like OpenWebUI pipelines): ``host.containers.internal:11434``
        - See `Ollama Connection`_ for OpenWebUI-specific configuration

        **3. Deployed Services**

        Ensure the following are uncommented in ``deployed_services``:

        - ``jupyter`` - Environment for editing and running generated code
        - ``open_webui`` - Web-based chat interface
        - ``pipelines`` - Core agent runtime environment

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
        - **Configuration file (config.yml):** Application behavior, model choices, capabilities that stay the same

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
           Anthropic API key for Claude models.

           Get from: https://console.anthropic.com/

        ``GOOGLE_API_KEY``
           Google API key for Gemini models.

           Get from: https://makersuite.google.com/app/apikey

        ``CBORG_API_KEY``
           CBorg API key (LBNL internal only).

           Get from: https://cborg.lbl.gov/

        **Optional Variables:**

        ``OSPREY_PROJECT``
           Default project directory for CLI commands (new in v0.7.7). Allows working with specific projects without changing directories.

           Example: ``/home/user/projects/my-agent``

           See :doc:`../developer-guides/02_quick-start-patterns/00_cli-reference` for multi-project workflow examples.

        ``LOCAL_PYTHON_VENV``
           Path to Python virtual environment for local execution mode.

           Default: Uses current active environment

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

Building and Running
~~~~~~~~~~~~~~~~~~~~

Once you have installed the framework, created a project, and configured it, you can start the services. The framework includes a deployment CLI that orchestrates all services using Podman containers.

**Start Services**

The framework CLI provides convenient commands for managing services. For detailed information about all deployment options, see :doc:`../developer-guides/05_production-systems/05_container-and-deployment` or the :doc:`CLI reference <../developer-guides/02_quick-start-patterns/00_cli-reference>`.

.. admonition:: New in v0.7+: Framework CLI Commands
   :class: version-07plus-change

   Service management is now handled through the :doc:`osprey deploy <../developer-guides/02_quick-start-patterns/00_cli-reference>` command instead of running Python scripts directly.

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

**Access OpenWebUI**

Once services are running, access the web interface at:

- OpenWebUI: `http://localhost:8080 <http://localhost:8080>`_

.. _openwebui-configuration:

OpenWebUI Configuration
~~~~~~~~~~~~~~~~~~~~~~~

`OpenWebUI <https://openwebui.com/>`_ is a feature-rich, self-hosted web interface for Language Models that provides a ChatGPT-like experience with extensive customization options. The framework's integration provides real-time progress tracking during agent execution, automatic display of :func:`registered figures <osprey.state.StateManager.register_figure>` and :func:`notebooks <osprey.state.StateManager.register_notebook>`, and session continuity across conversations.

.. _Ollama Connection:

**Ollama Connection:**

The framework automatically configures OpenWebUI to connect to Ollama at ``http://host.docker.internal:11434``. This special hostname allows containers to access services running on the host machine (containers cannot access the host's localhost directly).

.. note::
   On **Linux with Podman**, use ``host.containers.internal`` instead of ``host.docker.internal``. Update the URL in ``services/open-webui/docker-compose.yml`` if needed.

Once Ollama is serving, `OpenWebUI <https://openwebui.com/>`_ will automatically discover all models currently available in your Ollama installation.

**Pipeline Connection:**

The Osprey framework provides a pipeline connection to integrate the agent framework with OpenWebUI.

.. dropdown:: Understanding Pipelines
   :color: info
   :icon: info

   `OpenWebUI Pipelines <https://docs.openwebui.com/features/pipelines/>`_ are a powerful extensibility system that allows you to customize and extend OpenWebUI's functionality. Think of pipelines as plugins that can:

   - **Filter**: Process user messages before they reach the LLM and modify responses after they return
   - **Pipe**: Create custom "models" that integrate external APIs, build workflows, or implement RAG systems
   - **Integrate**: Connect with external services, databases, or specialized AI providers

   Pipelines appear as models with an "External" designation in your model selector and enable advanced functionality like real-time data retrieval, custom processing workflows, and integration with external AI services.

The framework automatically configures the pipeline connection with these settings:

- **URL**: ``http://pipelines:9099`` (default configuration)
- **API Key**: ``0p3n-w3bu!`` (default for local development)

.. note::
   **Production Deployments**: The default ``PIPELINES_API_KEY`` is intended for local development only. For production or shared deployments, override it by adding ``PIPELINES_API_KEY=your-secure-key`` to your ``.env`` file. Use a strong, randomly generated key.

**Note**: The URL uses ``pipelines:9099`` instead of ``localhost:9099`` because OpenWebUI runs inside a container and communicates with the pipelines service through the container network.

.. dropdown:: Manual Configuration (if needed)
   :color: secondary
   :icon: tools

   If you need to manually configure or verify the pipeline connection:

   1. Go to **Admin Panel** → **Settings** (upper panel) → **Connections** (left panel)
   2. Click the **(+)** button in **Manage OpenAI API Connections**
   3. Configure the pipeline connection with these details:

      - **URL**: ``http://pipelines:9099`` (if using default configuration)
      - **API Key**: Found in ``services/osprey/pipelines/docker-compose.yml.j2`` under ``PIPELINES_API_KEY`` (default ``0p3n-w3bu!``)

**Authentication:**

For local development convenience, OpenWebUI authentication is disabled (``WEBUI_AUTH=false``). This means:

- **No login required** - Access the interface immediately at `http://localhost:8080 <http://localhost:8080>`_
- **Full admin access** - All features and settings are available without authentication
- **Faster workflow** - No need to manage passwords or user accounts

.. warning::
   **Security Consideration**: If you deploy OpenWebUI on a shared server or expose it to a network, you should enable authentication for security. To enable authentication:

   1. Open ``services/osprey/open-webui/docker-compose.yml.j2``
   2. Change ``WEBUI_AUTH=false`` to ``WEBUI_AUTH=true``
   3. Redeploy with ``osprey deploy up``
   4. On first visit, you'll be prompted to create an admin account

   For localhost-only deployments (default), authentication is not necessary.



**Additional OpenWebUI Configuration:**

For optimal performance and user experience, consider these additional configuration settings:

.. tab-set::

    .. tab-item:: Model Management
       :name: model-management

        **Making Models Public:**

        To use Ollama models for OpenWebUI features like chat tagging, title generation, and other automated tasks, you must configure them as public models:

        1. Go to **Admin Panel** → **Settings** → **Models**
        2. Find the Ollama model you want to use (e.g., ``mistral:7b``, ``llama3:8b``)
        3. Click the **edit button** (pencil icon) next to the model
        4. Ensure the model is **activated** (enabled)
        5. Set the model visibility to **Public** (not Private)
        6. Click **Save** to apply the changes

        **Deactivating Unused Models:**

        - Deactivate unused (Ollama-)models in **Admin Panel** → **Settings** → **Models** to reduce clutter
        - This helps keep your model selection interface clean and focused on the models you actually use
        - You can always reactivate models later if needed

    .. tab-item:: Chat Augmentation

        OpenWebUI automatically generates titles and tags for conversations, which can interfere with your main agent's processing. It's recommended to use a dedicated local model for this:

        1. Go to **Admin Panel** → **Settings** → **Interface**
        2. Find **Task Model** setting
        3. Change from **Current Model** to any local Ollama model (e.g., ``mistral:7b``, ``llama3:8b``)
        4. This prevents title generation from consuming your main agent's resources

        Note that this model needs to be public as well (see :ref:`model-management` section to the left).

    .. tab-item:: Buttons

        **Adding Custom Function Buttons:**

        OpenWebUI allows you to add custom function buttons to enhance the user interface. For comprehensive information about functions, see the `official OpenWebUI functions documentation <https://docs.openwebui.com/features/plugin/>`_.

        **Installing Functions:**

        1. Navigate to **Admin Panel** → **Functions**
        2. Add a function using the plus sign (UI details may vary between OpenWebUI versions)
        3. Copy and paste function code from our repository's pre-built functions

        **Available Functions in Repository:**

        The framework includes several pre-built functions located in ``services/osprey/open-webui/functions/``:

        - ``execution_history_button.py`` - View and manage execution history
        - ``agent_context_button.py`` - Access agent context information
        - ``memory_button.py`` - Memory management functionality
        - ``execution_plan_editor.py`` - Edit and manage execution plans

        **Activation Requirements:**

        After adding a function:

        1. **Enable the function** - Activate it in the functions interface
        2. **Enable globally** - Use additional options to enable the function globally
        3. **Refresh the page** - The button should appear on your OpenWebUI interface after refresh

        These buttons provide quick access to advanced agent capabilities and debugging tools.

    .. tab-item:: Debugging

        **Real-time Log Viewer:**

        For debugging and monitoring, use the ``/logs`` command in chat to view application logs without accessing container logs directly:

        - ``/logs`` - Show last 100 log entries
        - ``/logs 50`` - Show last 50 log entries
        - ``/logs help`` - Show available options

        This is particularly useful for troubleshooting when OpenWebUI provides minimal feedback by design.

    .. tab-item:: Default Prompts

        **Customizing Default Prompt Suggestions:**

        OpenWebUI provides default prompt suggestions that you can customize for your specific use case:

        **Accessing Default Prompts:**

        1. Go to **Admin Panel** → **Settings** → **Interface**
        2. Scroll down to find **Default Prompt Suggestions** section
        3. Here you can see the built-in OpenWebUI prompt suggestions

        **Customizing Prompts:**

        1. **Remove Default Prompts**: Clear the existing default prompts if they don't fit your workflow
        2. **Add Custom Prompts**: Replace them with prompts tailored to your agent's capabilities
        3. **Use Cases**:

           - **Production**: Set prompts that guide users toward your agent's core functionalities
           - **R&D Testing**: Create prompts that help test specific features or edge cases
           - **Domain-Specific**: Add prompts relevant to your application domain (e.g., ALS operations, data analysis)

        **Example Custom Prompts:**

        - "Analyze the recent beam performance data from the storage ring"
        - "Find PV addresses related to beam position monitors"
        - "Generate a summary of today's logbook entries"
        - "Help me troubleshoot insertion device issues"

        **Benefits:**

        - Guides users toward productive interactions with your agent
        - Reduces cognitive load for new users
        - Enables consistent testing scenarios during development
        - Improves user adoption by showcasing agent capabilities


Troubleshooting
~~~~~~~~~~~~~~~

**Common Issues:**

- If you encounter connection issues with Ollama, ensure you're using ``host.containers.internal`` instead of ``localhost`` when connecting from containers
- Verify that all required services are uncommented in ``config.yml``
- Check that API keys are properly set in the ``.env`` file
- Ensure container runtime is running (Docker Desktop or Podman machine on macOS/Windows)
- If containers fail to start, check logs with: ``docker logs <container_name>`` or ``podman logs <container_name>``

**Verification Steps:**

1. Check Python version: ``python --version`` (should be 3.11.x or higher)
2. Check container runtime version: ``docker --version`` or ``podman --version`` (should be 4.0.0+)
3. Verify virtual environment is active (should see ``(venv)`` in your prompt)
4. Test core framework imports: ``python -c "import langgraph; print('LangGraph installed successfully')"``
5. Test container connectivity: ``docker run --rm alpine ping -c 1 host.containers.internal`` (or use ``podman`` instead)
6. Check service status: ``docker ps`` or ``podman ps``

**Common Installation Issues:**

- **Python version mismatch**: Ensure you're using Python 3.11+ with ``python3.11 -m venv venv``
- **Package conflicts**: If you get dependency conflicts, try creating a fresh virtual environment
- **Missing dependencies**: The main requirements.txt should install everything needed; avoid mixing with system packages

Next Steps
~~~~~~~~~~

.. seealso::

   :doc:`hello-world-tutorial`
      Build your first simple weather agent

   :doc:`control-assistant`
      Production control system assistant with channel finding and comprehensive tooling

   :doc:`../developer-guides/02_quick-start-patterns/00_cli-reference`
      Complete CLI command reference

   :doc:`../api_reference/01_core_framework/04_configuration_system`
      Deep dive into configuration system

   :doc:`../developer-guides/03_core-framework-systems/03_registry-and-discovery`
      Understanding the registry and component discovery
