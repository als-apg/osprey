Installation & Setup
====================

Get OSPREY running in five steps. The whole process takes about 10 minutes.

.. dropdown:: **What you'll have when done**
   :color: info
   :icon: check-circle

   - Node.js, Claude Code, and ``uv`` installed
   - An API key configured for your AI provider
   - The ``osprey`` CLI installed and on your ``PATH``
   - The ability to create projects with ``osprey build``


Step 1: Install Node.js
------------------------

Claude Code requires `Node.js <https://nodejs.org/>`_ 18+.

.. tab-set::

   .. tab-item:: macOS (Homebrew)

      .. code-block:: bash

         brew install node

      If you don't have Homebrew, install it first:

      .. code-block:: bash

         /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   .. tab-item:: Linux

      .. code-block:: bash

         # Ubuntu/Debian
         curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
         sudo apt-get install -y nodejs

         # Or use your distribution's package manager

   .. tab-item:: Windows (WSL2)

      Install Node.js inside your WSL2 environment:

      .. code-block:: bash

         curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
         sudo apt-get install -y nodejs

Verify:

.. code-block:: bash

   node --version
   # You should see v18.x.x or higher


Step 2: Install Claude Code
-----------------------------

.. code-block:: bash

   npm install -g @anthropic-ai/claude-code

Verify:

.. code-block:: bash

   claude --version


Step 3: Install Python tools
------------------------------

OSPREY uses `uv <https://docs.astral.sh/uv/>`_ for fast Python package management.
It handles Python versions and virtual environments automatically.

.. tab-set::

   .. tab-item:: macOS (Homebrew)

      .. code-block:: bash

         brew install uv

   .. tab-item:: Linux / WSL2

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

Verify:

.. code-block:: bash

   uv --version


Step 4: Set up your API key
-----------------------------

Claude Code needs an API key for the AI provider. Set it in your shell profile
so it's always available.

.. tab-set::

   .. tab-item:: CBORG (LBNL users)

      `CBORG <https://api.cborg.lbl.gov>`_ is the LBNL AI proxy. If you don't have
      a key yet, sign up at https://api.cborg.lbl.gov.

      .. code-block:: bash

         echo 'export CBORG_API_KEY="your-key-here"' >> ~/.zshrc
         source ~/.zshrc

      Replace ``your-key-here`` with your actual key. To verify:

      .. code-block:: bash

         echo $CBORG_API_KEY

   .. tab-item:: Anthropic (direct)

      Get an API key from https://console.anthropic.com/.

      .. code-block:: bash

         echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.zshrc
         source ~/.zshrc

   .. tab-item:: Other providers

      OSPREY supports 100+ providers via LiteLLM. Set the appropriate key:

      .. code-block:: bash

         # OpenAI
         export OPENAI_API_KEY="sk-..."

         # Google
         export GOOGLE_API_KEY="..."

      See :doc:`/how-to/configure-providers` for the full list.

.. note::
   Using ``bash`` instead of ``zsh``? Replace ``~/.zshrc`` with ``~/.bashrc``.


Step 5: Install OSPREY
-----------------------

.. tab-set::

   .. tab-item:: Recommended (PyPI)

      Install OSPREY as a standalone CLI tool. ``uv`` creates an isolated
      environment for OSPREY and puts the ``osprey`` command on your ``PATH``.

      .. code-block:: bash

         uv tool install osprey-framework

      This is the right choice for most users. Your own projects stay separate
      from the tool — you'll create them in their own directories with
      ``osprey build``.

      To upgrade later:

      .. code-block:: bash

         uv tool upgrade osprey-framework

   .. tab-item:: From source

      Clone the repository if you want to pin to a specific git ref, track
      ``main`` for unreleased changes, or contribute back to OSPREY.

      .. code-block:: bash

         git clone https://github.com/als-apg/osprey.git
         cd osprey
         uv sync --extra dev

      This creates a ``.venv`` inside the clone with OSPREY installed in
      editable mode plus dev dependencies. Commands in the rest of the docs
      that show ``osprey ...`` should be run as ``uv run osprey ...`` from
      inside the clone, or activate the venv with ``source .venv/bin/activate``.

Verify:

.. code-block:: bash

   osprey --version

OSPREY versions follow the ``YYYY.MM.MICRO`` `CalVer <https://calver.org/>`_
scheme — the first two segments identify the release window, the micro
segment increments for hotfixes. See ``CHANGELOG.md`` for details.

You're done! 🎉
-----------------

OSPREY is installed and ready to use. Here's what to do next:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **Hello World Tutorial**
      :link: hello-world-tutorial
      :link-type: doc

      Build your first agent with a mock control system. One MCP server, zero
      complexity. Takes about 10 minutes.

   .. grid-item-card:: **Guided Build Interview**
      :link: osprey-build-interview
      :link-type: doc

      Set OSPREY up for your own detector, beamline, or accelerator subsystem.
      A guided conversation generates a ready-to-build project profile tailored
      to your system. Takes about 10--15 minutes.


.. dropdown:: **Advanced: Container runtime, services & detailed configuration**
   :color: secondary
   :icon: gear

   The steps above cover the core installation. The following are only needed for
   specific use cases.

   **Container Runtime (Docker or Podman)**

   A container runtime is only required if you plan to deploy containerized services
   (Jupyter, simulation IOCs, databases). The core agent workflow does not require
   containers.

   .. tab-set::

      .. tab-item:: Docker Desktop

         Download from the `Docker website <https://www.docker.com/products/docker-desktop/>`_
         and verify:

         .. code-block:: bash

            docker --version
            docker compose version

      .. tab-item:: Podman

         Install from the `Podman website <https://podman.io/docs/installation>`_
         and verify:

         .. code-block:: bash

            podman --version

         On macOS/Windows, also run:

         .. code-block:: bash

            podman machine init
            podman machine start

   **Deploying Services**

   See :doc:`/how-to/deploy-project` for setting up containerized services like Jupyter
   notebooks, databases, or simulation IOCs.

   **Detailed Configuration**

   See :doc:`/how-to/configure-providers` for provider setup and
   :doc:`/how-to/build-profiles` for the full build profile YAML reference.


Troubleshooting
~~~~~~~~~~~~~~~~

.. dropdown:: Common issues
   :color: warning
   :icon: alert

   **"claude: command not found"**
      Install Claude Code: ``npm install -g @anthropic-ai/claude-code``

   **"osprey: command not found"**
      If you installed via ``uv tool install osprey-framework``, make sure uv's
      tool bin directory is on your ``PATH`` — run ``uv tool update-shell`` once
      and open a new terminal. If you installed from source, either activate the
      venv (``source .venv/bin/activate``) or prefix commands with ``uv run``
      from inside the clone.

   **MCP connection failed**
      Ensure you're running ``claude`` from your project root where ``.mcp.json`` lives.

   **Provider authentication error**
      Check that your API key is exported: ``echo $CBORG_API_KEY`` (or whichever key
      you're using). Re-source your shell profile if needed: ``source ~/.zshrc``

   **Python version mismatch**
      OSPREY requires Python 3.11+. Check with ``python3 --version``. The ``uv`` tool
      can install the right version automatically.

   **Verification checklist:**

   .. code-block:: bash

      node --version       # Should be 18+
      claude --version     # Should print version
      uv --version         # Should print version
      osprey --version     # Should print version
